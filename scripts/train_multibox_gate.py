"""Train the multi-scale, multi-candidate first-stage gate."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import configure_training, save_checkpoint, write_json
from src.multibox_gate import MobileNetV3SmallMultiBoxGate, decode_candidates, multibox_loss


def connected_boxes(mask, min_size=32):
    """Return normalized boxes for 8-connected crack components."""
    ys, xs = np.where(mask)
    points = set(zip(ys.tolist(), xs.tolist()))
    boxes = []
    while points:
        seed = points.pop()
        stack = [seed]
        min_y = max_y = seed[0]; min_x = max_x = seed[1]
        while stack:
            y, x = stack.pop()
            min_y = min(min_y, y); max_y = max(max_y, y)
            min_x = min(min_x, x); max_x = max(max_x, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dy and not dx:
                        continue
                    item = (y + dy, x + dx)
                    if item in points:
                        points.remove(item); stack.append(item)
        width = max(max_x - min_x + 1, min_size)
        height = max(max_y - min_y + 1, min_size)
        center_x = (min_x + max_x + 1) / 2
        center_y = (min_y + max_y + 1) / 2
        x0 = max(0, center_x - width / 2); x1 = min(2048, center_x + width / 2)
        y0 = max(0, center_y - height / 2); y1 = min(1536, center_y + height / 2)
        boxes.append(((x0 + x1) / 4096, (y0 + y1) / 3072,
                      (x1 - x0) / 2048, (y1 - y0) / 1536))
    return boxes


class MultiBoxDataset(Dataset):
    def __init__(self, frame_root, split, preview_size=(512, 384), augment=False):
        self.frame_root = frame_root; self.preview_size = preview_size; self.augment = augment
        manifest = json.loads((frame_root / 'manifest.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.positive = [bool(r['positive']) for r in self.records]
        self.boxes = []
        for record in self.records:
            with Image.open(frame_root / record['mask']) as mask_image:
                mask = np.asarray(mask_image) == 40
            self.boxes.append(connected_boxes(mask))

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.frame_root / record['image']) as image:
            image = image.convert('RGB').resize(self.preview_size, Image.Resampling.BILINEAR)
        boxes = [list(box) for box in self.boxes[index]]
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            for box in boxes: box[0] = 1.0 - box[0]
        x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        return x, boxes, torch.tensor(bool(boxes)), record['group']


def collate(batch):
    xs, boxes, present, groups = zip(*batch)
    return torch.stack(xs), list(boxes), torch.stack(present), list(groups)


def evaluate_presence(model, loader, device, threshold=0.2):
    model.eval(); tp = fp = fn = 0; candidate_count = 0
    with torch.no_grad():
        for x, _, present, _ in loader:
            output = model(x.to(device)); predictions = output['image_logit'].sigmoid() >= 0.5
            truth = present.to(device).bool()
            tp += (predictions & truth).sum().item(); fp += (predictions & ~truth).sum().item()
            fn += (~predictions & truth).sum().item()
            candidate_count += sum(len(v) for v in decode_candidates(output, threshold))
    return {'image_recall': tp / max(tp + fn, 1), 'image_precision': tp / max(tp + fp, 1),
            'image_f1': 2 * tp / max(2 * tp + fp + fn, 1), 'mean_candidates': candidate_count / max(len(loader.dataset), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40); parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--preview-width', type=int, default=512); parser.add_argument('--preview-height', type=int, default=384)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    configure_training(42); project = Path(__file__).resolve().parents[1]
    frame_root = project / 'data/processed/TACK_Tunnel_B/full_frames'; args.output.mkdir(parents=True, exist_ok=False)
    datasets = {s: MultiBoxDataset(frame_root, s, (args.preview_width, args.preview_height), augment=s == 'train') for s in ('train', 'val', 'test')}
    positive = datasets['train'].positive; n_positive = sum(positive)
    weights = [1 / n_positive if p else 1 / (len(positive) - n_positive) for p in positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(42))
    loaders = {'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler, collate_fn=collate),
               'val': DataLoader(datasets['val'], args.batch_size, collate_fn=collate),
               'test': DataLoader(datasets['test'], args.batch_size, collate_fn=collate)}
    device = torch.device('cuda'); model = MobileNetV3SmallMultiBoxGate(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); history = []; best = -1
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (x, boxes, present, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3: break
            optimizer.zero_grad(set_to_none=True); output = model(x.to(device))
            loss = multibox_loss(output, boxes, present.to(device));
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite multi-box loss')
            loss.backward(); optimizer.step()
        metrics = evaluate_presence(model, loaders['val'], device); metrics.update(epoch=epoch)
        history.append(metrics); write_json(args.output / 'history.json', history)
        if metrics['image_f1'] > best:
            best = metrics['image_f1']; save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'val': metrics})
        print(json.dumps(metrics), flush=True)
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    report = {'model': 'MobileNetV3-Small multi-scale multi-box gate', 'preview_size': [args.preview_width, args.preview_height], 'smoke': args.smoke, 'history': history, 'test': evaluate_presence(model, loaders['test'], device)}
    write_json(args.output / 'metrics.json', report); print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
