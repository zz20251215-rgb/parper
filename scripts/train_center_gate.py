"""Train the center-heatmap candidate gate."""
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_multibox_gate import MultiBoxDataset, collate
from src.experiment import configure_training, save_checkpoint, write_json
from src.multibox_gate import MobileNetV3SmallCenterGate, center_gate_loss

def evaluate(model, loader, device):
    model.eval(); tp=fp=fn=0
    with torch.no_grad():
        for x, _, present, _ in loader:
            pred=model(x.to(device))['image_logit'].sigmoid()>=.5; truth=present.to(device).bool()
            tp+=(pred&truth).sum().item(); fp+=(pred&~truth).sum().item(); fn+=(~pred&truth).sum().item()
    return {'image_recall':tp/max(tp+fn,1),'image_precision':tp/max(tp+fp,1),'image_f1':2*tp/max(2*tp+fp+fn,1)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=40); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    configure_training(42); project=Path(__file__).resolve().parents[1]; root=project/'data/processed/TACK_Tunnel_B/full_frames'; a.output.mkdir(parents=True,exist_ok=False)
    ds={s:MultiBoxDataset(root,s,(512,384),augment=s=='train') for s in ('train','val','test')}; pos=ds['train'].positive; n=sum(pos); w=[1/n if x else 1/(len(pos)-n) for x in pos]; sampler=WeightedRandomSampler(w,len(w),replacement=True,generator=torch.Generator().manual_seed(42))
    loaders={'train':DataLoader(ds['train'],a.batch_size,sampler=sampler,collate_fn=collate),'val':DataLoader(ds['val'],a.batch_size,collate_fn=collate),'test':DataLoader(ds['test'],a.batch_size,collate_fn=collate)}
    device=torch.device('cuda'); model=MobileNetV3SmallCenterGate(True).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); hist=[]; best=-1
    for epoch in range(1,(1 if a.smoke else a.epochs)+1):
        model.train()
        for step,(x,b,present,_) in enumerate(loaders['train']):
            if a.smoke and step>=3: break
            opt.zero_grad(set_to_none=True); loss=center_gate_loss(model(x.to(device)),b,present.to(device)); loss.backward(); opt.step()
        m=evaluate(model,loaders['val'],device); m['epoch']=epoch; hist.append(m); write_json(a.output/'history.json',hist); print(json.dumps(m),flush=True)
        if m['image_f1']>best: best=m['image_f1']; save_checkpoint(a.output/'best.pt',{'model':model.state_dict(),'epoch':epoch,'val':m})
    model.load_state_dict(torch.load(a.output/'best.pt',map_location=device,weights_only=True)['model']); report={'model':'MobileNetV3-Small center heatmap gate','preview_size':[512,384],'smoke':a.smoke,'history':hist,'test':evaluate(model,loaders['test'],device)}; write_json(a.output/'metrics.json',report); print(json.dumps(report))
if __name__=='__main__': main()
