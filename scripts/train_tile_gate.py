import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CrackDataset
from src.experiment import configure_training, save_checkpoint, write_json
from src.tile_gate import MobileNetV3SmallTileGate


def evaluate(model, loader, device):
    model.eval(); tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y, _ in loader:
            truth = y.flatten(1).any(1).to(device)
            pred = model(x.to(device)).sigmoid() >= 0.5
            tp += (pred & truth).sum().item(); fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item(); tn += (~pred & ~truth).sum().item()
    return {'recall': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'false_positive_rate': fp / max(fp + tn, 1),
            'selected_rate': (tp + fp) / max(tp + fp + fn + tn, 1),
            'samples': tp + fp + fn + tn}


def main():
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    configure_training(42); a.output.mkdir(parents=True, exist_ok=False)
    root = project / 'data/TACK_Tunnel_Data'; split_root = project / 'data/processed/TACK_Tunnel_B'
    ds = {s: CrackDataset(root, split_root, s, augment=s == 'train') for s in ('train','val','test')}
    flags = ds['train'].positive; pos = sum(flags); neg = len(flags) - pos
    weights = [1 / pos if f else 1 / neg for f in flags]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(42))
    loaders = {'train': DataLoader(ds['train'], a.batch_size, sampler=sampler),
               'val': DataLoader(ds['val'], a.batch_size), 'test': DataLoader(ds['test'], a.batch_size)}
    device = torch.device('cuda'); model = MobileNetV3SmallTileGate(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = torch.nn.BCEWithLogitsLoss()
    history=[]; best=-1
    for epoch in range(1, (1 if a.smoke else a.epochs)+1):
        model.train()
        for step,(x,y,_) in enumerate(loaders['train']):
            if a.smoke and step >= 3: break
            target=y.flatten(1).any(1).float().to(device); optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x.to(device)),target); loss.backward(); optimizer.step()
        m=evaluate(model,loaders['val'],device); m['epoch']=epoch; history.append(m); write_json(a.output/'history.json',history)
        if m['f1']>best: best=m['f1']; save_checkpoint(a.output/'best.pt',{'model':model.state_dict(),'epoch':epoch,'val':m})
    model.load_state_dict(torch.load(a.output/'best.pt',map_location=device,weights_only=True)['model'])
    report={'model':'MobileNetV3-Small fixed-tile gate','smoke':a.smoke,'history':history,'test':evaluate(model,loaders['test'],device)}
    write_json(a.output/'metrics.json',report); print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
