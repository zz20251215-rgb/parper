"""Evaluate the dual-path frame-context ranker in the full-frame cascade."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallContextRanker
from src.mobilenet_unet import MobileNetV3LargeUNet
from src.tile_geometry import ORDER, TILES


def tile_tensor(image, tile):
    col, row = TILES[tile]
    array = np.array(image.crop((col * 512, row * 512, (col + 1) * 512,
                                 (row + 1) * 512)).convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranker', type=Path, required=True)
    parser.add_argument('--segmenter', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    project = Path(__file__).resolve().parents[1]
    root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    manifest = json.loads((root / 'manifest.json').read_text())
    device = torch.device('cuda')
    ranker = MobileNetV3SmallContextRanker(False).to(device)
    ranker.load_state_dict(torch.load(args.ranker, map_location=device, weights_only=True)['model'])
    ranker.eval()
    segmenter = MobileNetV3LargeUNet(False).to(device)
    segmenter.load_state_dict(torch.load(args.segmenter, map_location=device, weights_only=True)['model'])
    segmenter.eval()
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    rows = []
    for record in [r for r in manifest['records'] if r['split'] == args.split]:
        started = time.perf_counter()
        with Image.open(root / record['image']) as image_file, Image.open(root / record['mask']) as mask_file:
            image = image_file.convert('RGB')
            truth = torch.from_numpy(np.asarray(mask_file) == 40)
            tiles = torch.stack([tile_tensor(image, tile) for tile in ORDER])
            context = torch.from_numpy(np.array(image.resize((256, 192), Image.Resampling.BILINEAR), copy=True)).permute(2, 0, 1).float() / 255
            with torch.no_grad():
                scores = ranker(tiles.unsqueeze(0).to(device), context.unsqueeze(0).to(device))[0]
                indices = scores.topk(min(args.topk, 12)).indices.tolist()
                selected = [ORDER[index] for index in indices]
                logits = segmenter(tiles[indices].to(device))[:, 0]
            torch.cuda.synchronize(device)
            prediction = torch.zeros((1536, 2048), dtype=torch.bool)
            for index, tile in enumerate(selected):
                col, row = TILES[tile]
                prediction[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = logits[index].sigmoid().cpu() >= 0.5
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            for key, value in [('tp', tp), ('fp', fp), ('fn', fn)]:
                totals[key] += value
            rows.append({'group': record['group'], 'selected_tiles': selected,
                         'seconds': time.perf_counter() - started, 'tp': tp, 'fp': fp, 'fn': fn})
    tp, fp, fn = totals.values()
    seconds = [row['seconds'] for row in rows]
    report = {'ranker': str(args.ranker), 'segmenter': str(args.segmenter), 'split': args.split,
              'topk': args.topk, 'mean_selected_tiles': args.topk,
              'dice': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1),
              'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
              'mean_end_to_end_seconds': sum(seconds) / max(len(seconds), 1),
              'p95_end_to_end_seconds': float(np.percentile(seconds, 95)),
              'totals': totals, 'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
