import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Support running this entry point directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.resnet_unet import ResNet34UNet
from src.mobilenet_unet import MobileNetV3LargeUNet, MobileNetV3LargeRefineUNet
from src.losses import loss_fn, loss_components, combine_loss
from src.dataset import CrackDataset
from src.training import EarlyStopping
from src.experiment import configure_training, save_checkpoint, write_json
from src.plotting import save_training_curves

LOGGER = logging.getLogger('training')


def evaluate(model, loader, device, limit=0):
    model.eval()
    tp = fp = fn = 0
    bce_sum = dice_sum = samples = 0
    negative_images = negative_fp_images = negative_fp_pixels = negative_pixels = 0
    with torch.no_grad():
        for step, (x, y, _) in enumerate(loader):
            if limit and step >= limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            bce, dice, positive = loss_components(logits, y)
            loss = combine_loss(bce, dice, positive)
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
            bce_sum += bce.sum().item()
            dice_sum += dice[positive].sum().item()
            samples += len(x)
    positive_images = samples - negative_images
    bce_loss = bce_sum / samples
    dice_loss = dice_sum / positive_images if positive_images else 0.0
    return {'loss': bce_loss + dice_loss, 'bce_loss': bce_loss, 'dice_loss': dice_loss, 'dice': 2 * tp / max(2 * tp + fp + fn, 1),
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
    parser.add_argument('--model', choices=['resnet34', 'mobilenetv3_large', 'mobilenetv3_large_refine'], default='resnet34')
    parser.add_argument('--output', type=Path, default=project_root / 'outputs/runs/tack_resnet34_unet')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')
    configure_training(42)
    device = torch.device('cuda')
    root = project_root / 'data/TACK_Tunnel_Data'
    split_root = project_root / 'data/processed/TACK_Tunnel_B'
    args.output.mkdir(parents=True, exist_ok=False)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.addHandler(logging.FileHandler(args.output / 'train.log', encoding='utf-8'))
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
    model_label = {'resnet34': 'ResNet34-U-Net',
                   'mobilenetv3_large': 'MobileNetV3-Large-U-Net',
                   'mobilenetv3_large_refine': 'MobileNetV3-Large-Refine-U-Net'}[args.model]
    encoder_weights = {
        'resnet34': 'ResNet34_Weights.IMAGENET1K_V1',
        'mobilenetv3_large': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2',
        'mobilenetv3_large_refine': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2',
    }[args.model]
    config = {'model': model_label, 'model_key': args.model, 'pretrained': True,
              'encoder_weights': encoder_weights,
              'encoder_warmup_epochs': 3, 'encoder_learning_rate': 1e-4,
              'normalization': 'ImageNet mean/std inside model, input remains RGB in [0,1]', 'seed': 42,
              'source': str(root.relative_to(project_root)),
              'splits': str(split_root.relative_to(project_root)),
              'split_summary': json.loads((split_root / 'manifest.json').read_text())['summary'],
              'epochs': 1 if args.smoke else args.epochs, 'batch_size': args.batch_size,
              'learning_rate': 1e-3, 'optimizer': 'Adam',
              'scheduler': 'ReduceLROnPlateau', 'minimum_lr': 1e-5,
              'lr_patience': 5, 'lr_factor': 0.5,
              'early_stopping': {'min_epochs': 30, 'patience': 15, 'min_delta': 0.002},
              'loss': 'BCE(all images) + Dice(positive images)',
              'loss_reporting': 'dataset-global mean BCE + mean Dice loss over positive images',
              'train_sampling': 'expected 1:1 positive/negative with replacement; len(train) draws/epoch',
              'augmentation': 'horizontal flip p=0.5 and uniform 0/90/180/270 degree rotation',
              'threshold': 0.5, 'smoke': args.smoke,
              'smoke_sampling': 'alternating positive/negative subset' if args.smoke else None,
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'deterministic_algorithms': False, 'tf32': True, 'cudnn_benchmark': True,
              'precision': 'FP32 with TF32 convolution/matmul; no AMP'}
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'process.json', {'pid': os.getpid(), 'status': 'running',
               'started_at': datetime.now(timezone.utc).isoformat()})
    write_json(args.output / 'experiment.json', {
        'status': 'experiment_record', 'deterministic_algorithms': False,
        'message': f'迁移学习：{encoder_weights} 编码器；保存逐轮原始指标和论文曲线，不要求逐位复现。'})
    model_class = {'resnet34': ResNet34UNet,
                   'mobilenetv3_large': MobileNetV3LargeUNet,
                   'mobilenetv3_large_refine': MobileNetV3LargeRefineUNet}[args.model]
    model = model_class(pretrained=True).to(device)
    optimizer = torch.optim.Adam([
        {'params': list(model.decoder.parameters()) + list(model.head.parameters()), 'lr': 1e-3},
        {'params': model.encoder.parameters(), 'lr': 1e-4}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.002,
        threshold_mode='abs', min_lr=[1e-5, 1e-6])
    stopping = EarlyStopping()
    initial = {k: v.detach().clone() for k, v in model.named_parameters()} if args.smoke else None
    history = []
    best = -1
    LOGGER.info(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(), 'smoke': args.smoke}))
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        epoch_started = time.monotonic()
        model.set_encoder_trainable(epoch > config['encoder_warmup_epochs'])
        model.train()
        train_bce_sum = train_dice_sum = samples = steps = positive_seen = negative_seen = 0
        for step, (x, y, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            bce, dice, positive = loss_components(model(x), y)
            loss = combine_loss(bce, dice, positive)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            gradient_checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if gradient_checks and not torch.stack(gradient_checks).all():
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            batch_positive = y.flatten(1).any(1).sum().item()
            positive_seen += batch_positive
            negative_seen += len(y) - batch_positive
            train_bce_sum += bce.detach().sum().item()
            train_dice_sum += dice.detach()[positive].sum().item()
            samples += len(x)
            steps += 1
            LOGGER.info(f'epoch={epoch} step={step + 1} loss={loss.item():.6f}')
        metrics = evaluate(model, loaders['val'], device, 2 if args.smoke else 0)
        row = {'epoch': epoch, 'train_loss': train_bce_sum / samples + train_dice_sum / max(positive_seen, 1),
               'train_bce_loss': train_bce_sum / samples, 'train_dice_loss': train_dice_sum / max(positive_seen, 1),
               'encoder_learning_rate': optimizer.param_groups[1]['lr'],
               'encoder_frozen': epoch <= config['encoder_warmup_epochs'], 'train_steps': steps,
               'train_samples': samples, 'train_positive': positive_seen, 'train_negative': negative_seen,
               'learning_rate': optimizer.param_groups[0]['lr'], 'val': metrics}
        history.append(row)
        LOGGER.info(json.dumps(row))
        if metrics['dice'] > best:
            best = metrics['dice']
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                            'val': metrics, 'smoke': args.smoke, 'model_name': model_label})
        scheduler.step(metrics['dice'])
        stop = stopping.update(epoch, metrics['dice'])
        row['next_learning_rate'] = optimizer.param_groups[0]['lr']
        row['next_encoder_learning_rate'] = optimizer.param_groups[1]['lr']
        row['epochs_without_improvement'] = stopping.stale_epochs
        row['duration_seconds'] = time.monotonic() - epoch_started
        write_json(args.output / 'history.json', history)
        save_training_curves(args.output)
        if stop:
            LOGGER.info(f'Early stopping at epoch {epoch}: no effective Dice improvement for 15 epochs')
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
    write_json(args.output / 'metrics.json', report)
    process = json.loads((args.output / 'process.json').read_text())
    process.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output / 'process.json', process)
    LOGGER.info(json.dumps(report))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        LOGGER.exception('Training failed')
        raise
