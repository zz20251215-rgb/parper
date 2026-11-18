"""Train a mass-aware ranker for the fixed 3x4 tile geometry."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallMassRanker
from src.experiment import configure_training, save_checkpoint, write_json
from src.tile_geometry import transform_frame

TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}
ORDER = list(TILES)
MAX_PIXELS = float(512 * 512)


class TileFrameDataset(Dataset):
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
        tiles, has_crack, masses, pixel_counts = [], [], [], []
        for tile, source_tile in zip(ORDER, source_tiles):
            col, row = TILES[tile]
            crop = image.crop((col * 512, row * 512, (col + 1) * 512, (row + 1) * 512))
            tiles.append(torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float() / 255)
            info = record['tiles'][source_tile]
            has_crack.append(float(info['has_crack']))
            count = float(info['crack_pixels'])
            masses.append(np.log1p(count) / np.log1p(MAX_PIXELS))
            pixel_counts.append(count)
        return (torch.stack(tiles), torch.tensor(has_crack), torch.tensor(masses, dtype=torch.float32),
                torch.tensor(pixel_counts), record['group'])


def scores_from_output(output, presence_weight):
    return output['mass'] + presence_weight * output['presence']


def evaluate(model, loader, device, topk, presence_weight):
    model.eval()
    tile_tp = tile_fp = tile_fn = 0
    covered_pixels = total_pixels = 0.0
    with torch.no_grad():
        for x, labels, masses, pixel_counts, _ in loader:
            output = model(x.to(device))
            scores = scores_from_output(output, presence_weight)
            indices = scores.topk(min(topk, 12), dim=1).indices.cpu()
            truth = labels.bool()
            selected = torch.zeros_like(truth)
            selected.scatter_(1, indices, True)
            tile_tp += (selected & truth).sum().item()
            tile_fp += (selected & ~truth).sum().item()
            tile_fn += (~selected & truth).sum().item()
            covered_pixels += (pixel_counts * selected).sum().item()
            total_pixels += pixel_counts.sum().item()
    return {
        'topk': topk,
        'presence_weight': presence_weight,
        'tile_recall': tile_tp / max(tile_tp + tile_fn, 1),
        'tile_precision': tile_tp / max(tile_tp + tile_fp, 1),
        'pixel_mass_coverage': covered_pixels / max(total_pixels, 1e-9),
        'mean_selected_tiles': float(topk),
    }


def loss_fn(output, labels, masses):
    positive = labels.sum().clamp_min(1)
    negative = labels.numel() - labels.sum()
    pos_weight = (negative / positive).clamp(1, 12)
    presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output['presence'], labels, pos_weight=pos_weight)
    positive_mask = labels.bool()
    if positive_mask.any():
        mass_loss = torch.nn.functional.smooth_l1_loss(output['mass'][positive_mask], masses[positive_mask])
    else:
        mass_loss = output['mass'].sum() * 0
    return presence_loss + mass_loss, presence_loss.detach(), mass_loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    configure_training(42)
    root = Path(__file__).resolve().parents[1] / 'data/processed/TACK_Tunnel_B/full_frames'
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {split: TileFrameDataset(root, split, split == 'train') for split in ('train', 'val', 'test')}
    positive = datasets['train'].positive
    count = sum(positive)
    weights = [1 / count if value else 1 / (len(positive) - count) for value in positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {
        'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler),
        'val': DataLoader(datasets['val'], args.batch_size),
        'test': DataLoader(datasets['test'], args.batch_size),
    }
    device = torch.device('cuda')
    model = MobileNetV3SmallMassRanker(True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best = [], -1
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        losses = []
        for step, (x, labels, masses, _, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, presence_loss, mass_loss = loss_fn(model(x.to(device)), labels.to(device), masses.to(device))
            loss.backward()
            optimizer.step()
            losses.append((loss.item(), presence_loss.item(), mass_loss.item()))
        metrics = {'epoch': epoch, 'loss': float(np.mean([v[0] for v in losses]))}
        metrics['val_top6'] = evaluate(model, loaders['val'], device, 6, 0.5)
        metrics['val_top3'] = evaluate(model, loaders['val'], device, 3, 0.5)
        history.append(metrics)
        write_json(args.output / 'history.json', history)
        print(json.dumps(metrics), flush=True)
        score = metrics['val_top6']['pixel_mass_coverage']
        if score > best:
            best = score
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'val': metrics})
    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    sweep = []
    for weight in (0.0, 0.25, 0.5, 1.0, 2.0):
        for topk in (2, 3, 4, 6):
            row = evaluate(model, loaders['val'], device, topk, weight)
            sweep.append(row)
    chosen = max(sweep, key=lambda row: (row['pixel_mass_coverage'], -row['topk'], -row['presence_weight']))
    report = {
        'model': 'MobileNetV3-Small mass-aware fixed-geometry ranker',
        'objective': 'presence BCE plus positive-tile log crack-pixel regression',
        'history': history,
        'validation_sweep': sweep,
        'chosen_validation_setting': chosen,
        'test': evaluate(model, loaders['test'], device, chosen['topk'], chosen['presence_weight']),
        'smoke': args.smoke,
    }
    write_json(args.output / 'metrics.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'history' and key != 'validation_sweep'}, indent=2))


if __name__ == '__main__':
    main()
