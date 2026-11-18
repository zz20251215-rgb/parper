"""Evaluate the trained 512x512 segmenter on stitched 2048x1536 frames.

This is the full-coverage reference: all 12 confirmed tiles are inferred and
their logits are placed back at the original frame coordinates. It provides a
fair frame-level baseline before ROI gating is introduced.
"""

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
from src.resnet_unet import ResNet34UNet


TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}


def load_model(model_name, checkpoint, device):
    cls = ResNet34UNet if model_name == 'resnet34' else MobileNetV3LargeUNet
    model = cls(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    return model.eval()


def read_tile(frame, x, y):
    array = np.asarray(frame.crop((x, y, x + 512, y + 512)).convert('RGB'))
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float() / 255


def metrics(pred, truth):
    pred, truth = pred.bool(), truth.bool()
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    return {'dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'iou': tp / max(tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'tp': tp, 'fp': fp, 'fn': fn}


def aggregate_metrics(total):
    tp, fp, fn = total['tp'], total['fp'], total['fn']
    return {'dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'iou': tp / max(tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'tp': tp, 'fp': fp, 'fn': fn}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet34', 'mobilenetv3_large'], default='mobilenetv3_large')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--mode', choices=['all_tiles', 'oracle_positive_tiles'], default='all_tiles')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark')
    frame_root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    if args.checkpoint is None:
        args.checkpoint = project / ('outputs/runs/tack_mobilenetv3_large/best.pt'
                                    if args.model == 'mobilenetv3_large'
                                    else 'outputs/runs/tack_resnet34_20261115_222833/best.pt')
    manifest = json.loads((frame_root / 'manifest.json').read_text())
    records = [r for r in manifest['records'] if r['split'] == args.split]
    if args.limit:
        records = records[:args.limit]
    device = torch.device('cuda')
    model = load_model(args.model, args.checkpoint, device)
    total = {'tp': 0, 'fp': 0, 'fn': 0}
    rows = []
    for record in records:
        with Image.open(frame_root / record['image']) as image, Image.open(frame_root / record['mask']) as mask:
            truth = torch.from_numpy(np.asarray(mask) == 40)
            # Unprocessed tiles are explicit background in the oracle upper bound.
            logits = torch.full((1536, 2048), float('-inf'), device=device)
            selected = list(TILES)
            if args.mode == 'oracle_positive_tiles':
                selected = list(record['positive_tiles'])
            started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            with torch.no_grad():
                batch = torch.stack([read_tile(image, TILES[tile][0] * 512, TILES[tile][1] * 512)
                                     for tile in selected]).to(device)
                batch = model(batch)[:, 0]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            for tile_index, tile in enumerate(selected):
                col, row = TILES[tile]
                logits[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = batch[tile_index]
            pred = logits.sigmoid() >= 0.5
            row = metrics(pred.cpu(), truth)
            row.update({'group': record['group'], 'seconds': elapsed, 'tiles': len(selected),
                        'positive': record['positive']})
            rows.append(row)
            for key in ('tp', 'fp', 'fn'):
                total[key] += row[key]
    summary = aggregate_metrics(total)
    report = {'model': args.model, 'split': args.split, 'mode': args.mode,
              'frame_size': [2048, 1536], 'tile_size': [512, 512],
              'coverage': 'all 12 tiles' if args.mode == 'all_tiles' else 'oracle positive tiles',
              'frames': len(rows), 'summary': summary,
              'mean_seconds': sum(r['seconds'] for r in rows) / max(len(rows), 1),
              'mean_tiles': sum(r['tiles'] for r in rows) / max(len(rows), 1),
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('model', 'split', 'mode', 'frames', 'summary', 'mean_seconds', 'mean_tiles')}, indent=2))


if __name__ == '__main__':
    main()
