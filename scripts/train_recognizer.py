import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import FullFrameRecognizerDataset, RecognizerDataset
from src.experiment import configure_training, save_checkpoint, write_json
from src.recognizer import MobileNetV3SmallRecognizer, recognizer_loss


def box_iou(box_a, box_b):
    def corners(box):
        cx, cy, w, h = box.unbind(-1)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    a0, a1, a2, a3 = corners(box_a)
    b0, b1, b2, b3 = corners(box_b)
    inter = (torch.minimum(a2, b2) - torch.maximum(a0, b0)).clamp_min(0) * \
            (torch.minimum(a3, b3) - torch.maximum(a1, b1)).clamp_min(0)
    area_a = (a2 - a0).clamp_min(0) * (a3 - a1).clamp_min(0)
    area_b = (b2 - b0).clamp_min(0) * (b3 - b1).clamp_min(0)
    return inter / (area_a + area_b - inter).clamp_min(1e-8)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    ious = []
    with torch.no_grad():
        for x, present, target_box, _ in loader:
            out = model(x.to(device))
            pred = out['logit'].sigmoid() >= 0.5
            truth = present.to(device).bool()
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()
            if truth.any():
                ious.extend(box_iou(out['box'][truth], target_box.to(device)[truth]).tolist())
    return {'image_recall': tp / max(tp + fn, 1),
            'image_precision': tp / max(tp + fp, 1),
            'image_f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'box_iou_positive_mean': sum(ious) / len(ious) if ious else 0.0,
            'samples': len(loader.dataset)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--dataset', choices=['tiles', 'full_frames'], default='full_frames')
    parser.add_argument('--preview-width', type=int, default=512)
    parser.add_argument('--preview-height', type=int, default=384)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run')
    configure_training(42)
    project = Path(__file__).resolve().parents[1]
    root = project / 'data/TACK_Tunnel_Data'
    split_root = project / 'data/processed/TACK_Tunnel_B'
    frame_root = split_root / 'full_frames'
    args.output.mkdir(parents=True, exist_ok=False)
    if args.dataset == 'full_frames':
        if not (frame_root / 'manifest.json').exists():
            raise FileNotFoundError('Run scripts/stitch_tack.py before full-frame recognizer training')
        datasets = {s: FullFrameRecognizerDataset(
            frame_root, s, (args.preview_width, args.preview_height))
                    for s in ('train', 'val', 'test')}
    else:
        datasets = {s: RecognizerDataset(root, split_root, s, args.size)
                    for s in ('train', 'val', 'test')}
    positive = datasets['train'].positive
    weights = [1 / sum(positive) if p else 1 / (len(positive) - sum(positive)) for p in positive]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {'train': DataLoader(datasets['train'], args.batch_size, sampler=sampler),
               'val': DataLoader(datasets['val'], args.batch_size),
               'test': DataLoader(datasets['test'], args.batch_size)}
    device = torch.device('cuda')
    model = MobileNetV3SmallRecognizer(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (x, present, target_box, _) in enumerate(loaders['train']):
            if args.smoke and step >= 3:
                break
            optimizer.zero_grad(set_to_none=True)
            out = model(x.to(device))
            loss = recognizer_loss(out, present.to(device), target_box.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite recognizer loss')
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, loaders['val'], device)
        metrics['epoch'] = epoch
        history.append(metrics)
        write_json(args.output / 'history.json', history)
        if metrics['image_f1'] >= max(x['image_f1'] for x in history):
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'val': metrics})
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    report = {'model': 'MobileNetV3-Small recognizer', 'dataset': args.dataset,
              'preview_size': ([args.preview_width, args.preview_height]
                               if args.dataset == 'full_frames' else args.size),
              'smoke': args.smoke, 'history': history,
              'test': evaluate(model, loaders['test'], device)}
    write_json(args.output / 'metrics.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
