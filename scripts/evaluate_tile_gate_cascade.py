"""Calibrate a fixed-tile gate on validation frames and evaluate the cascade."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mobilenet_unet import MobileNetV3LargeUNet
from src.tile_gate import MobileNetV3SmallTileGate


TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}


def tile_tensor(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def load_models(project, device, gate_checkpoint=None):
    gate = MobileNetV3SmallTileGate(pretrained=False).to(device)
    gate_path = Path(gate_checkpoint) if gate_checkpoint else project / 'outputs/runs/tile_gate_mobilenetv3_small/best.pt'
    gate.load_state_dict(torch.load(
        gate_path,
        map_location=device, weights_only=True)['model'])
    segmenter = MobileNetV3LargeUNet(pretrained=False).to(device)
    segmenter.load_state_dict(torch.load(
        project / 'outputs/runs/tack_mobilenetv3_large/best.pt',
        map_location=device, weights_only=True)['model'])
    return gate.eval(), segmenter.eval()


def tile_rows(frame_root, split):
    manifest = json.loads((frame_root / 'manifest.json').read_text())
    for record in manifest['records']:
        if record['split'] != split:
            continue
        for tile, (col, row) in TILES.items():
            info = record['tiles'][tile]
            yield record, tile, frame_root / record['image'], frame_root / record['mask'], bool(info['has_crack'])


def tile_scores(frame_root, split, gate, device):
    rows = []
    with torch.no_grad():
        batch, pending = [], []
        for record, tile, image_path, mask_path, positive in tile_rows(frame_root, split):
            with Image.open(image_path) as image:
                col, row = TILES[tile]
                crop = image.crop((col * 512, row * 512, (col + 1) * 512, (row + 1) * 512))
                array = np.array(crop.convert('RGB'), copy=True)
            batch.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255)
            pending.append((record, tile, image_path, mask_path, positive))
            if len(batch) == 48:
                scores = gate(torch.stack(batch).to(device)).sigmoid().cpu().tolist()
                rows.extend([dict(record=r, tile=t, image=i, mask=m, positive=p, score=float(s))
                             for (r, t, i, m, p), s in zip(pending, scores)])
                batch, pending = [], []
        if batch:
            scores = gate(torch.stack(batch).to(device)).sigmoid().cpu().tolist()
            rows.extend([dict(record=r, tile=t, image=i, mask=m, positive=p, score=float(s))
                         for (r, t, i, m, p), s in zip(pending, scores)])
    return rows


def gate_metrics(rows, threshold):
    tp = fp = fn = tn = selected = 0
    for row in rows:
        pred = row['score'] >= threshold
        selected += pred
        if pred and row['positive']: tp += 1
        elif pred: fp += 1
        elif row['positive']: fn += 1
        else: tn += 1
    return {'threshold': threshold, 'recall': tp / max(tp + fn, 1),
            'precision': tp / max(tp + fp, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'false_positive_rate': fp / max(fp + tn, 1),
            'selected_tiles': selected, 'selected_rate': selected / max(len(rows), 1),
            'tiles': len(rows)}


def choose_threshold(rows, target_recall=0.9):
    candidates = sorted({0.0, 1.0, *[round(row['score'], 6) for row in rows]})
    eligible = [gate_metrics(rows, t) for t in candidates]
    eligible = [m for m in eligible if m['recall'] >= target_recall]
    if not eligible:
        return max((gate_metrics(rows, t) for t in candidates), key=lambda m: m['recall'])
    return min(eligible, key=lambda m: (m['selected_rate'], -m['precision']))


def cascade_test(frame_root, rows, segmenter, device, threshold):
    by_group = {}
    for row in rows:
        by_group.setdefault(row['record']['group'], []).append(row)
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    frame_rows = []
    for group, group_rows in by_group.items():
        image_path = group_rows[0]['image']; mask_path = group_rows[0]['mask']
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            truth = torch.from_numpy(np.asarray(mask) == 40)
            selected = [row['tile'] for row in group_rows if row['score'] >= threshold]
            started = time.perf_counter()
            if selected:
                batch = torch.stack([
                    tile_tensor_from_frame(image, TILES[tile][0] * 512, TILES[tile][1] * 512)
                    for tile in selected]).to(device)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                with torch.no_grad(): logits = segmenter(batch)[:, 0]
                if device.type == 'cuda': torch.cuda.synchronize(device)
            else:
                logits = []
            seconds = time.perf_counter() - started
            pred = torch.zeros((1536, 2048), dtype=torch.bool)
            for index, tile in enumerate(selected):
                col, row = TILES[tile]
                pred[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = logits[index].sigmoid().cpu() >= 0.5
            tp = (pred & truth).sum().item(); fp = (pred & ~truth).sum().item(); fn = (~pred & truth).sum().item()
            for key, value in [('tp', tp), ('fp', fp), ('fn', fn)]: totals[key] += value
            frame_rows.append({'group': group, 'selected_tiles': selected, 'tiles': len(selected),
                               'seconds': seconds, 'tp': tp, 'fp': fp, 'fn': fn})
    tp, fp, fn = totals.values()
    summary = {'dice': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1),
               'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), **totals}
    return summary, frame_rows


def tile_tensor_from_frame(image, x, y):
    array = np.array(image.crop((x, y, x + 512, y + 512)).convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-recall', type=float, default=0.9)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-checkpoint', type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    frame_root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    device = torch.device('cuda')
    gate, segmenter = load_models(project, device, args.gate_checkpoint)
    val_rows = tile_scores(frame_root, 'val', gate, device)
    choice = choose_threshold(val_rows, args.target_recall)
    test_rows = tile_scores(frame_root, 'test', gate, device)
    test_gate = gate_metrics(test_rows, choice['threshold'])
    cascade_summary, frame_rows = cascade_test(frame_root, test_rows, segmenter, device, choice['threshold'])
    report = {'gate': 'MobileNetV3-Small fixed 512x512 tile gate',
              'gate_checkpoint': str(args.gate_checkpoint or (project / 'outputs/runs/tile_gate_mobilenetv3_small/best.pt')),
              'segmenter': 'MobileNetV3-Large-U-Net', 'target_validation_recall': args.target_recall,
              'validation_choice': choice, 'test_gate': test_gate,
              'test_cascade': cascade_summary, 'mean_selected_tiles': sum(r['tiles'] for r in frame_rows) / max(len(frame_rows), 1),
              'mean_segment_seconds': sum(r['seconds'] for r in frame_rows) / max(len(frame_rows), 1),
              'frames': frame_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('validation_choice','test_gate','test_cascade','mean_selected_tiles','mean_segment_seconds')}, indent=2))


if __name__ == '__main__': main()
