"""Calibrate image-level no-candidate skip scores on complete frames.

This read-only analysis uses the fixed-grid ranker's tile logits. It does not
change inference behavior or train any model. Thresholds are selected only
from the validation split and the test split is reported only descriptively.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallGridRanker
from scripts.benchmark_roi_cascade import crop_tensor
from src.tile_geometry import ORDER


def score_frames(ranker, root, records, device):
    rows = []
    with torch.inference_mode():
        for record in records:
            with Image.open(root / record['image']) as source:
                image = source.convert('RGB')
                tiles = torch.stack([crop_tensor(image, tile) for tile in ORDER]).to(device)
            logits = ranker(tiles.unsqueeze(0))[0].float().cpu().numpy()
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            rows.append({
                'group': record['group'],
                'split': record['split'],
                'positive': bool(record['positive']),
                'max_probability': float(probabilities.max()),
                'top6_min_probability': float(np.sort(probabilities)[-6]),
                'top6_mean_probability': float(np.sort(probabilities)[-6:].mean()),
            })
    return rows


def sweep(rows, score_key):
    values = sorted({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     *[round(row[score_key], 6) for row in rows]})
    result = []
    for threshold in values:
        positive = [row for row in rows if row['positive']]
        negative = [row for row in rows if not row['positive']]
        predicted_positive = lambda row: row[score_key] >= threshold
        tp = sum(predicted_positive(row) for row in positive)
        fp = sum(predicted_positive(row) for row in negative)
        result.append({
            'threshold': threshold,
            'positive_recall': tp / max(len(positive), 1),
            'negative_false_positive_rate': fp / max(len(negative), 1) if negative else None,
            'predicted_positive_frames': tp + fp,
            'positive_frames': len(positive),
            'negative_frames': len(negative),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    project = Path(__file__).resolve().parents[1]
    root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    device = torch.device('cuda')
    ranker = MobileNetV3SmallGridRanker(False).to(device)
    ranker.load_state_dict(torch.load(args.ranker, map_location=device, weights_only=True)['model'])
    ranker.eval()
    rows = score_frames(ranker, root, manifest['records'], device)
    by_split = {}
    for split in ('train', 'val', 'test'):
        split_rows = [row for row in rows if row['split'] == split]
        by_split[split] = {
            'frames': len(split_rows),
            'positive_frames': sum(row['positive'] for row in split_rows),
            'negative_frames': sum(not row['positive'] for row in split_rows),
            'sweeps': {key: sweep(split_rows, key) for key in (
                'max_probability', 'top6_min_probability', 'top6_mean_probability')},
        }
    report = {
        'protocol': 'fixed-grid-frame-skip-calibration-v1',
        'ranker': str(args.ranker),
        'ranker_sha256': __import__('hashlib').sha256(args.ranker.read_bytes()).hexdigest(),
        'manifest': str(manifest_path),
        'score_definition': {
            'max_probability': 'sigmoid of highest tile logit',
            'top6_min_probability': 'sigmoid of the sixth-highest tile logit',
            'top6_mean_probability': 'mean sigmoid probability of the six highest tile logits',
        },
        'threshold_selection': 'validation split only; test split is descriptive because it has no negative complete frames',
        'splits': by_split,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    val = by_split['val']['sweeps']['max_probability']
    print(json.dumps({'val_max_probability_sweep': val}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
