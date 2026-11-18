import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coarse_gate import MobileNetV3SmallCoarseGate, coarse_gate_loss
from src.dataset import FullFrameCoarseDataset
from src.experiment import configure_training, save_checkpoint, write_json


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    pixel_tp = pixel_fp = pixel_fn = 0
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).sigmoid() >= 0.5
            truth = F.interpolate(y.to(device), size=pred.shape[-2:], mode='nearest').bool()
            image_pred = pred.flatten(1).any(1)
            image_truth = truth.flatten(1).any(1)
            tp += (image_pred & image_truth).sum().item()
            fp += (image_pred & ~image_truth).sum().item()
            fn += (~image_pred & image_truth).sum().item()
            pixel_tp += (pred & truth).sum().item()
            pixel_fp += (pred & ~truth).sum().item()
            pixel_fn += (~pred & truth).sum().item()
    return {
        'image_recall': tp / max(tp + fn, 1),
        'image_precision': tp / max(tp + fp, 1),
        'image_f1': 2 * tp / max(2 * tp + fp + fn, 1),
        'pixel_recall': pixel_tp / max(pixel_tp + pixel_fn, 1),
        'pixel_precision': pixel_tp / max(pixel_tp + pixel_fp, 1),
        'pixel_dice': 2 * pixel_tp / max(2 * pixel_tp + pixel_fp + pixel_fn, 1),
        'samples': len(loader.dataset),
    }


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--preview-width', type=int, default=512)
    parser.add_argument('--preview-height', type=int, default=384)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')
    frame_root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    if not (frame_root / 'manifest.json').exists():
        raise FileNotFoundError('Run scripts/stitch_tack.py first')
    configure_training(42)
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {s: FullFrameCoarseDataset(frame_root, s, (args.preview_width, args.preview_height))
                for s in ('train', 'val', 'test')}
    positive = datasets['train'].positive
    n_positive = sum(positive)
    weights = [1 / n_positive if flag else 1 / (len(positive) - n_positive) for flag in positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {
        'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler),
        'val': DataLoader(datasets['val'], args.batch_size),
        'test': DataLoader(datasets['test'], args.batch_size),
    }
    device = torch.device('cuda')
    model = MobileNetV3SmallCoarseGate(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1
    history = []
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        for step, (x, y, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = coarse_gate_loss(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite coarse gate loss')
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, loaders['val'], device)
        metrics['epoch'] = epoch
        history.append(metrics)
        write_json(args.output / 'history.json', history)
        if metrics['image_f1'] > best:
            best = metrics['image_f1']
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                            'val': metrics, 'preview_size': [args.preview_width, args.preview_height]})
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    report = {'model': 'MobileNetV3-Small coarse pixel gate',
              'preview_size': [args.preview_width, args.preview_height], 'smoke': args.smoke,
              'history': history, 'test': evaluate(model, loaders['test'], device)}
    write_json(args.output / 'metrics.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
