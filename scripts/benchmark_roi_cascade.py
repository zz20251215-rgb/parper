"""Benchmark full coverage and ROI ranking with matched warm inference."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import (MobileNetV3SmallEvidenceGridRanker,
                             MobileNetV3SmallGridRanker,
                             MobileNetV3SmallShallowGridRanker)
from src.experiment import configure_training, write_json
from src.mobilenet_unet import MobileNetV3LargeUNet, MobileNetV3LargeRefineUNet
from src.tile_geometry import ORDER, TILES
from scripts.run_dynamic_roi_cascade import crop_tensor


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def predict_frame(path, ranker, segmenter, topk, batch_size, device, segmenter_dtype=torch.float32):
    with Image.open(path) as source:
        image = source.convert('RGB')
        if image.size != (2048, 1536):
            raise ValueError(f'Expected 2048x1536, got {image.size}')
        tiles = torch.stack([crop_tensor(image, tile) for tile in ORDER]).to(device)
    indices = list(range(12))
    if ranker is not None:
        indices = ranker(tiles.unsqueeze(0))[0].topk(topk).indices.tolist()
    prediction = np.zeros((1536, 2048), dtype=bool)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        masks = (segmenter(tiles[selected].to(dtype=segmenter_dtype))[:, 0].sigmoid() >= 0.5).cpu().numpy()
        for index, mask in zip(selected, masks):
            col, row = TILES[ORDER[index]]
            prediction[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = mask
    return prediction, [ORDER[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranker', type=Path)
    parser.add_argument('--ranker-arch', choices=['grid', 'shallow', 'evidence'], default='grid')
    parser.add_argument('--segmenter', type=Path, required=True)
    parser.add_argument('--segmenter-arch', choices=['mobilenetv3_large', 'mobilenetv3_large_refine'], default='mobilenetv3_large')
    parser.add_argument('--segmenter-format', choices=['eager', 'torchscript'], default='eager')
    parser.add_argument('--segmenter-precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--mode', choices=['ranked', 'full'], default='ranked')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.topk <= 12 or min(args.batch_size, args.warmup, args.repeats) < 1:
        parser.error('topk must be 1..12; batch-size, warmup and repeats must be positive')
    if args.mode == 'ranked' and args.ranker is None:
        parser.error('--ranker is required in ranked mode')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    configure_training(42)
    root = Path(__file__).resolve().parents[1] / 'data/processed/TACK_Tunnel_B/full_frames'
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest['records'] if r['split'] == args.split]
    if not records:
        raise ValueError('No frames in split')
    device = torch.device('cuda')
    ranker = None
    if args.mode == 'ranked':
        ranker_cls = {'grid': MobileNetV3SmallGridRanker,
                      'shallow': MobileNetV3SmallShallowGridRanker,
                      'evidence': MobileNetV3SmallEvidenceGridRanker}[args.ranker_arch]
        ranker = ranker_cls(False).to(device)
        ranker.load_state_dict(torch.load(args.ranker, map_location=device, weights_only=True)['model'])
        ranker.eval()
    segmenter_dtype = torch.float16 if args.segmenter_precision == 'fp16' else torch.float32
    if args.segmenter_format == 'torchscript':
        segmenter = torch.jit.load(str(args.segmenter), map_location=device)
    else:
        segmenter_cls = MobileNetV3LargeRefineUNet if args.segmenter_arch == 'mobilenetv3_large_refine' else MobileNetV3LargeUNet
        segmenter = segmenter_cls(False).to(device)
        segmenter.load_state_dict(torch.load(args.segmenter, map_location=device, weights_only=True)['model'])
    if args.segmenter_precision == 'fp16':
        segmenter = segmenter.half()
    segmenter.eval()
    for _ in range(args.warmup):
        predict_frame(root / records[0]['image'], ranker, segmenter, args.topk, args.batch_size, device, segmenter_dtype)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    totals = dict(tp=0, fp=0, fn=0)
    rows, timing_rows = [], []
    negative_frames = negative_fp = covered_tiles = positive_tiles = covered_pixels = total_pixels = 0
    for repeat in range(args.repeats):
        for record in records:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction, selected = predict_frame(root / record['image'], ranker, segmenter,
                                                  args.topk, args.batch_size, device, segmenter_dtype)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            timing_rows.append(dict(repeat=repeat + 1, group=record['group'], seconds=elapsed))
            if repeat:
                continue
            # Annotation I/O and quality scoring are outside inference timing.
            with Image.open(root / record['mask']) as mask:
                truth = np.asarray(mask) == 40
            tp = int(np.logical_and(prediction, truth).sum())
            fp = int(np.logical_and(prediction, ~truth).sum())
            fn = int(np.logical_and(~prediction, truth).sum())
            for key, value in [('tp', tp), ('fp', fp), ('fn', fn)]:
                totals[key] += value
            covered = sum(record['tiles'][tile]['crack_pixels'] for tile in selected)
            covered_pixels += covered
            total_pixels += int(truth.sum())
            covered_tiles += sum(record['tiles'][tile]['has_crack'] for tile in selected)
            positive_tiles += len(record['positive_tiles'])
            if not record['positive']:
                negative_frames += 1
                negative_fp += int(fp > 0)
            rows.append(dict(group=record['group'], selected_tiles=selected, tp=tp, fp=fp,
                             fn=fn, candidate_crack_pixels=covered, crack_pixels=int(truth.sum())))
    tp, fp, fn = (totals[key] for key in ('tp', 'fp', 'fn'))
    seconds = [row['seconds'] for row in timing_rows]
    report = dict(
        protocol='fixed-grid-warm-inference-v2', mode=args.mode, ranker_arch=args.ranker_arch,
        ranker=str(args.ranker) if ranker is not None else None,
        ranker_sha256=sha256(args.ranker) if ranker is not None else None,
        segmenter=str(args.segmenter), segmenter_arch=args.segmenter_arch,
        segmenter_format=args.segmenter_format, segmenter_precision=args.segmenter_precision,
        segmenter_sha256=sha256(args.segmenter),
        manifest_sha256=sha256(manifest_path), split=args.split, frame_size=[2048, 1536],
        tile_size=[512, 512], topk=args.topk if ranker is not None else 12,
        mean_selected_tiles=sum(len(r['selected_tiles']) for r in rows) / len(rows),
        batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats,
        timing_scope='decode, all-tile preparation/transfer, optional ranking, selected segmentation, CPU reconstruction; excludes annotation I/O and scoring',
        timing_limitations='warm models and mostly cached local files; offline sequential frames; no device energy measurement',
        mean_end_to_end_seconds=float(np.mean(seconds)),
        p95_end_to_end_seconds=float(np.percentile(seconds, 95)),
        per_repeat_mean_seconds=[float(np.mean([r['seconds'] for r in timing_rows if r['repeat'] == i + 1])) for i in range(args.repeats)],
        dice=2 * tp / max(2 * tp + fp + fn, 1), iou=tp / max(tp + fp + fn, 1),
        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), totals=totals,
        candidate_pixel_coverage=covered_pixels / max(total_pixels, 1),
        candidate_tile_recall=covered_tiles / max(positive_tiles, 1),
        negative_frames=negative_frames,
        negative_frame_false_positive_rate=negative_fp / negative_frames if negative_frames else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                         cpu_threads=torch.get_num_threads(), precision=f'{args.segmenter_precision.upper()} segmenter; FP32 ranker, TF32 allowed',
                         cudnn_benchmark=torch.backends.cudnn.benchmark),
        frames=rows, timing_samples=timing_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('frames', 'timing_samples')}, indent=2))


if __name__ == '__main__':
    main()
