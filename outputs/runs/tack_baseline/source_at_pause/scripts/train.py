import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Support running this entry point directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unet import UNet
from src.dataset import CrackDataset
from src.training import EarlyStopping


def loss_fn(logits, target):
    prob = logits.sigmoid()
    dice = (2 * (prob * target).sum((1, 2, 3)) + 1) / (
        prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    positive = target.sum((1, 2, 3)) > 0
    # Empty masks learn background through BCE; Dice focuses on visible cracks.
    dice_loss = ((1 - dice) * positive).sum() / positive.sum().clamp_min(1)
    return nn.functional.binary_cross_entropy_with_logits(logits, target) + dice_loss


def evaluate(model, loader, device, limit=0):
    model.eval()
    tp = fp = fn = 0
    loss_sum = samples = 0
    negative_images = negative_fp_images = negative_fp_pixels = negative_pixels = 0
    with torch.no_grad():
        for step, (x, y, _) in enumerate(loader):
            if limit and step >= limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite evaluation loss')
            pred, truth = logits.sigmoid() >= 0.5, y.bool()
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()
            negatives = ~truth.flatten(1).any(1)
            negative_images += negatives.sum().item()
            negative_fp_images += pred[negatives].flatten(1).any(1).sum().item()
            negative_fp_pixels += pred[negatives].sum().item()
            negative_pixels += negatives.sum().item() * y.shape[-2] * y.shape[-1]
            loss_sum += loss.item() * len(x)
            samples += len(x)
    return {'loss': loss_sum / samples, 'dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'iou': tp / max(tp + fp + fn, 1), 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'samples': samples,
            'positive_images': samples - negative_images, 'negative_images': negative_images,
            'negative_false_positive_images': negative_fp_images,
            'negative_image_fp_rate': negative_fp_images / negative_images if negative_images else None,
            'negative_pixel_fp_rate': negative_fp_pixels / negative_pixels if negative_pixels else None}


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, default=project_root / 'outputs/runs/tack_unet')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda')
    root = project_root / 'data/TACK_Tunnel_Data'
    split_root = project_root / 'data/processed/TACK_Tunnel_B'
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {s: CrackDataset(root, split_root, s, augment=s == 'train') for s in ['train', 'val', 'test']}
    loaders = {}
    for split, ds in datasets.items():
        positive = [i for i, flag in enumerate(ds.positive) if flag]
        negative = [i for i, flag in enumerate(ds.positive) if not flag]
        if not positive or not negative:
            raise ValueError(f'{split} requires positive and negative samples')
        sampler = None
        selected = ds
        if args.smoke:
            limit = (3 if split == 'train' else 2) * args.batch_size
            indices = [i for pair in zip(positive, negative) for i in pair][:limit]
            selected = Subset(ds, indices)
        elif split == 'train':
            weights = [1 / len(positive) if flag else 1 / len(negative) for flag in ds.positive]
            sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True,
                                            generator=torch.Generator().manual_seed(42))
        loaders[split] = DataLoader(selected, batch_size=args.batch_size, sampler=sampler)
    config = {'model': 'UNet(base_channels=32)', 'pretrained': False, 'seed': 42,
              'source': str(root.relative_to(project_root)),
              'splits': str(split_root.relative_to(project_root)),
              'split_summary': json.loads((split_root / 'manifest.json').read_text())['summary'],
              'epochs': 1 if args.smoke else args.epochs, 'batch_size': args.batch_size,
              'learning_rate': 1e-3, 'optimizer': 'Adam',
              'scheduler': 'ReduceLROnPlateau', 'minimum_lr': 1e-5,
              'lr_patience': 5, 'lr_factor': 0.5,
              'early_stopping': {'min_epochs': 30, 'patience': 15, 'min_delta': 0.002},
              'loss': 'BCE(all images) + Dice(positive images)',
              'train_sampling': 'expected 1:1 positive/negative with replacement; len(train) draws/epoch',
              'augmentation': 'horizontal flip p=0.5 and uniform 0/90/180/270 degree rotation',
              'threshold': 0.5, 'smoke': args.smoke,
              'smoke_sampling': 'alternating positive/negative subset' if args.smoke else None,
              'torch': torch.__version__, 'cuda': torch.version.cuda}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.002,
        threshold_mode='abs', min_lr=1e-5)
    stopping = EarlyStopping()
    initial = {k: v.detach().clone() for k, v in model.named_parameters()} if args.smoke else None
    history = []
    best = -1
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(), 'smoke': args.smoke}), flush=True)
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        total = samples = steps = positive_seen = negative_seen = 0
        for step, (x, y, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            batch_positive = y.flatten(1).any(1).sum().item()
            positive_seen += batch_positive
            negative_seen += len(y) - batch_positive
            total += loss.item() * len(x)
            samples += len(x)
            steps += 1
            print(f'epoch={epoch} step={step + 1} loss={loss.item():.6f}', flush=True)
        metrics = evaluate(model, loaders['val'], device, 2 if args.smoke else 0)
        row = {'epoch': epoch, 'train_loss': total / samples, 'train_steps': steps,
               'train_samples': samples, 'train_positive': positive_seen, 'train_negative': negative_seen,
               'learning_rate': optimizer.param_groups[0]['lr'], 'val': metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics['dice'] > best:
            best = metrics['dice']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': metrics,
                        'smoke': args.smoke}, args.output / 'best.pt')
        scheduler.step(metrics['dice'])
        stop = stopping.update(epoch, metrics['dice'])
        row['next_learning_rate'] = optimizer.param_groups[0]['lr']
        row['epochs_without_improvement'] = stopping.stale_epochs
        (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if stop:
            print(f'Early stopping at epoch {epoch}: no effective Dice improvement for 15 epochs', flush=True)
            break
    changed = None
    if args.smoke:
        changed = any(not torch.equal(initial[k], v) for k, v in model.named_parameters())
        if not changed:
            raise RuntimeError('Optimizer did not update parameters')
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    test = evaluate(model, loaders['test'], device, 2 if args.smoke else 0)
    model.eval()
    prediction_ids = []
    # Save a crack example and a true negative, rather than an arbitrary first tile.
    for flag in [True, False]:
        index = datasets['test'].positive.index(flag)
        with torch.no_grad():
            x, y, key = datasets['test'][index]
            pred = (model(x[None].to(device)).sigmoid()[0, 0] >= 0.5).cpu().numpy()
        image = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        truth = Image.fromarray((y[0].numpy() * 255).astype(np.uint8)).convert('RGB')
        prediction = Image.fromarray((pred * 255).astype(np.uint8)).convert('RGB')
        canvas = Image.new('RGB', (image.width * 3, image.height))
        for i, im in enumerate([image, truth, prediction]):
            canvas.paste(im, (i * image.width, 0))
        canvas.save(args.output / f'prediction_{key}.png')
        prediction_ids.append(key)
    report = {'smoke': args.smoke, 'history': history, 'test': test,
              'stop_reason': 'early_stopping' if stop else 'epoch_limit',
              'parameters_updated': changed, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
              'prediction_ids': prediction_ids}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
