"""Evaluate candidate coverage and trigger cost for a trained multi-box gate."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_multibox_gate import MultiBoxDataset, collate
from src.multibox_gate import MobileNetV3SmallMultiBoxGate, decode_candidates


def iou(a, b):
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max(a[2], 0) * max(a[3], 0) + max(b[2], 0) * max(b[3], 0) - inter
    return inter / max(union, 1e-8)


def nms(candidates, threshold):
    kept = []
    for score, box in sorted(candidates, key=lambda x: x[0], reverse=True):
        box = box.tolist()
        if all(iou(box, old[1].tolist()) < threshold for old in kept):
            kept.append((score, torch.tensor(box)))
    return kept


def box_coverage(candidates, truth_boxes, expansion):
    covered = 0.0
    for truth in truth_boxes:
        cx, cy, w, h = truth
        best = 0.0
        for _, candidate in candidates:
            bx, by, bw, bh = candidate.tolist()
            bw *= expansion; bh *= expansion
            best = max(best, iou((cx, cy, w, h), (bx, by, bw, bh)))
        covered += best
    return covered / max(len(truth_boxes), 1)


def evaluate(model, dataset, device, threshold, nms_threshold, expansion):
    loader = DataLoader(dataset, 8, collate_fn=collate)
    frame_rows = []; covered_pixels = 0; total_pixels = 0; selected = 0; positive_frames = 0; image_hits = 0
    with torch.no_grad():
        for x, truth_boxes, present, groups in loader:
            output = model(x.to(device)); decoded = decode_candidates(output, threshold, max_candidates=128)
            for index, group in enumerate(groups):
                candidates = nms(decoded[index], nms_threshold)
                selected += len(candidates)
                truth = truth_boxes[index]
                if truth:
                    positive_frames += 1
                    coverage = box_coverage(candidates, truth, expansion)
                    image_hits += coverage > 0.5
                    covered_pixels += coverage * len(truth)
                    total_pixels += len(truth)
                frame_rows.append({'group': group, 'truth_boxes': len(truth), 'candidates': len(candidates),
                                   'coverage': box_coverage(candidates, truth, expansion) if truth else 0.0})
    return {'candidate_threshold': threshold, 'nms_threshold': nms_threshold, 'expansion': expansion,
            'mean_candidates': selected / max(len(dataset), 1),
            'positive_frame_recall': image_hits / max(positive_frames, 1),
            'mean_box_iou_coverage': covered_pixels / max(total_pixels, 1), 'frames': frame_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val'); parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--nms-threshold', type=float, default=0.5); parser.add_argument('--expansion', type=float, default=1.25)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    device = torch.device('cuda'); model = MobileNetV3SmallMultiBoxGate(False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)['model']); model.eval()
    dataset = MultiBoxDataset(project / 'data/processed/TACK_Tunnel_B/full_frames', args.split, (512, 384))
    report = evaluate(model, dataset, device, args.threshold, args.nms_threshold, args.expansion)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__': main()
