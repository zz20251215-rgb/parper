"""Train a ranker against frozen second-stage segmentation utility."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallUtilityRanker
from src.experiment import configure_training, save_checkpoint, write_json

TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}
ORDER = list(TILES)


class UtilityDataset(Dataset):
    def __init__(self, root, targets, split, augment=False):
        if augment:
            raise ValueError('Frozen segmenter targets require untransformed images; recompute targets before augmentation')
        self.root = root
        self.augment = augment
        manifest = json.loads((root / 'manifest.json').read_text())
        target_map = {row['group']: row for row in targets['records']}
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.target_map = target_map
        self.positive = [r['positive'] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image']) as image_file:
            image = image_file.convert('RGB')
        tiles, labels, utilities = [], [], []
        utility_row = self.target_map[record['group']]['tiles']
        for tile in ORDER:
            col, row = TILES[tile]
            crop = image.crop((col * 512, row * 512, (col + 1) * 512, (row + 1) * 512))
            tiles.append(torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float() / 255)
            labels.append(float(record['tiles'][tile]['has_crack']))
            utilities.append(float(utility_row[tile]['utility']))
        return torch.stack(tiles), torch.tensor(labels), torch.tensor(utilities), record['group']


def evaluate(model, loader, device, topk, presence_weight):
    model.eval()
    tp = fp = fn = 0
    utility_total = utility_selected = 0.0
    with torch.no_grad():
        for x, labels, utilities, _ in loader:
            output = model(x.to(device))
            scores = output['utility'] + presence_weight * output['presence']
            indices = scores.topk(min(topk, 12), dim=1).indices.cpu()
            truth = labels.bool()
            selected = torch.zeros_like(truth)
            selected.scatter_(1, indices, True)
            tp += (selected & truth).sum().item()
            fp += (selected & ~truth).sum().item()
            fn += (~selected & truth).sum().item()
            utility_selected += (utilities * selected).sum().item()
            utility_total += utilities.clamp_min(0).sum().item()
    return {
        'topk': topk, 'presence_weight': presence_weight,
        'tile_recall': tp / max(tp + fn, 1), 'tile_precision': tp / max(tp + fp, 1),
        'selected_utility': utility_selected / max(len(loader.dataset), 1),
        'positive_utility_coverage': utility_selected / max(utility_total, 1e-9),
        'mean_selected_tiles': float(topk),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    configure_training(42)
    root = Path(__file__).resolve().parents[1] / 'data/processed/TACK_Tunnel_B/full_frames'
    targets = json.loads(args.targets.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {split: UtilityDataset(root, targets, split) for split in ('train', 'val', 'test')}
    positive = datasets['train'].positive
    count = sum(positive)
    weights = [1 / count if value else 1 / (len(positive) - count) for value in positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler),
               'val': DataLoader(datasets['val'], args.batch_size),
               'test': DataLoader(datasets['test'], args.batch_size)}
    device = torch.device('cuda')
    model = MobileNetV3SmallUtilityRanker(True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best = [], -1
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        losses = []
        for step, (x, labels, utilities, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            optimizer.zero_grad(set_to_none=True)
            output = model(x.to(device))
            positive = labels.sum().clamp_min(1)
            negative = labels.numel() - labels.sum()
            pos_weight = (negative / positive).clamp(1, 12)
            presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output['presence'], labels.to(device), pos_weight=pos_weight)
            utility_loss = torch.nn.functional.smooth_l1_loss(output['utility'], utilities.to(device))
            loss = presence_loss + utility_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = {'epoch': epoch, 'loss': float(np.mean(losses)),
                   'val_top6': evaluate(model, loaders['val'], device, 6, 0.5),
                   'val_top3': evaluate(model, loaders['val'], device, 3, 0.5)}
        history.append(metrics)
        write_json(args.output / 'history.json', history)
        print(json.dumps(metrics), flush=True)
        score = metrics['val_top6']['positive_utility_coverage']
        if score > best:
            best = score
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'val': metrics})
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    sweep = [evaluate(model, loaders['val'], device, topk, weight)
             for weight in (0.0, 0.25, 0.5, 1.0, 2.0) for topk in (2, 3, 4, 6)]
    chosen = max(sweep, key=lambda row: (row['positive_utility_coverage'], -row['topk']))
    report = {'model': 'MobileNetV3-Small frozen-segmenter utility ranker',
              'objective': 'presence BCE plus per-tile frozen segmenter utility regression',
              'targets': str(args.targets), 'history': history,
              'validation_sweep': sweep, 'chosen_validation_setting': chosen,
              'test': evaluate(model, loaders['test'], device, chosen['topk'], chosen['presence_weight']),
              'smoke': args.smoke}
    write_json(args.output / 'metrics.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('history', 'validation_sweep')}, indent=2))


if __name__ == '__main__':
    main()
