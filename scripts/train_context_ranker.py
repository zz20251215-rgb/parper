"""Train a two-path fixed-grid ranker with low-resolution frame context."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallContextRanker
from src.experiment import configure_training, save_checkpoint, write_json
from src.tile_geometry import ORDER, TILES, transform_frame

CONTEXT_SIZE = (256, 192)


class ContextFrameDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = root
        self.augment = augment
        manifest = json.loads((root / 'manifest.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.positive = [r['positive'] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image']) as image_file:
            image = image_file.convert('RGB')
        mirror = self.augment and np.random.rand() < 0.5
        image, source_tiles = transform_frame(image, mirror)
        tiles, labels = [], []
        for tile, source_tile in zip(ORDER, source_tiles):
            col, row = TILES[tile]
            crop = image.crop((col * 512, row * 512, (col + 1) * 512, (row + 1) * 512))
            tiles.append(torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float() / 255)
            labels.append(float(record['tiles'][source_tile]['has_crack']))
        context = image.resize(CONTEXT_SIZE, Image.Resampling.BILINEAR)
        context = torch.from_numpy(np.array(context, copy=True)).permute(2, 0, 1).float() / 255
        return torch.stack(tiles), context, torch.tensor(labels), record['group']


def evaluate(model, loader, device, topk):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for tiles, context, labels, _ in loader:
            scores = model(tiles.to(device), context.to(device))
            indices = scores.topk(min(topk, 12), dim=1).indices.cpu()
            truth = labels.bool()
            selected = torch.zeros_like(truth)
            selected.scatter_(1, indices, True)
            tp += (selected & truth).sum().item()
            fp += (selected & ~truth).sum().item()
            fn += (~selected & truth).sum().item()
    return {'topk': topk, 'tile_recall': tp / max(tp + fn, 1),
            'tile_precision': tp / max(tp + fp, 1), 'mean_selected_tiles': float(topk)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    configure_training(42)
    root = Path(__file__).resolve().parents[1] / 'data/processed/TACK_Tunnel_B/full_frames'
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {split: ContextFrameDataset(root, split, split == 'train') for split in ('train', 'val', 'test')}
    count = sum(datasets['train'].positive)
    weights = [1 / count if x else 1 / (len(datasets['train']) - count) for x in datasets['train'].positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler),
               'val': DataLoader(datasets['val'], args.batch_size),
               'test': DataLoader(datasets['test'], args.batch_size)}
    device = torch.device('cuda')
    model = MobileNetV3SmallContextRanker(True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best = [], -1
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        losses = []
        for step, (tiles, context, labels, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(tiles.to(device), context.to(device))
            positive = labels.sum().clamp_min(1)
            negative = labels.numel() - labels.sum()
            pos_weight = (negative / positive).clamp(1, 12)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.to(device), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = {'epoch': epoch, 'loss': float(np.mean(losses))}
        for topk in (1, 2, 3, 6):
            metrics[f'val_top{topk}'] = evaluate(model, loaders['val'], device, topk)
        history.append(metrics)
        write_json(args.output / 'history.json', history)
        print(json.dumps(metrics), flush=True)
        score = metrics['val_top6']['tile_recall']
        if score > best:
            best = score
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'val': metrics})
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    report = {'model': 'MobileNetV3-Small dual-path frame-context ranker',
              'context_size': CONTEXT_SIZE, 'history': history,
              'test': {f'top{topk}': evaluate(model, loaders['test'], device, topk)
                       for topk in (1, 2, 3, 6)}, 'smoke': args.smoke}
    write_json(args.output / 'metrics.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
