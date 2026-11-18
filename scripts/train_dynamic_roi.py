"""Train a fixed-geometry dynamic ROI ranker."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import (MobileNetV3SmallEvidenceGridRanker,
                             MobileNetV3SmallGridRanker,
                             MobileNetV3SmallShallowGridRanker, ranking_loss)
from src.experiment import configure_training, save_checkpoint, write_json
from src.tile_geometry import transform_frame

TILES={'A':(0,0),'B':(1,0),'C':(2,0),'D':(3,0),'E':(0,1),'F':(1,1),'G':(2,1),'H':(3,1),'I':(0,2),'L':(1,2),'M':(2,2),'N':(3,2)}
ORDER=list(TILES)

class TileFrameDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root=root; self.augment=augment; manifest=json.loads((root/'manifest.json').read_text()); self.records=[r for r in manifest['records'] if r['split']==split]; self.positive=[r['positive'] for r in self.records]
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(self.root/r['image']) as im: image=im.convert('RGB')
        xs=[]; labels=[]; mirror = self.augment and np.random.rand()<.5
        image, source_tiles = transform_frame(image, mirror)
        for tile, source_tile in zip(ORDER, source_tiles):
            col,row=TILES[tile]; crop=image.crop((col*512,row*512,(col+1)*512,(row+1)*512))
            xs.append(torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()/255); labels.append(float(r['tiles'][source_tile]['has_crack']))
        return torch.stack(xs), torch.tensor(labels), r['group']

def evaluate(model,loader,device,topk):
    model.eval(); tp=fp=fn=0; selected=0; groups=0
    with torch.no_grad():
        for x,y,_ in loader:
            scores=model(x.to(device)); idx=scores.topk(min(topk,12),dim=1).indices; truth=y.to(device).bool(); pred=torch.zeros_like(truth); pred.scatter_(1,idx,True); tp+=(pred&truth).sum().item(); fp+=(pred&~truth).sum().item(); fn+=(~pred&truth).sum().item(); selected+=pred.sum().item(); groups+=len(y)
    return {'topk':topk,'tile_recall':tp/max(tp+fn,1),'tile_precision':tp/max(tp+fp,1),'frame_tile_coverage':selected/max(groups*12,1),'mean_selected_tiles':selected/max(groups,1)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--pairwise-weight',type=float,default=0.0); p.add_argument('--arch',choices=['grid','shallow','evidence'],default='grid'); p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    configure_training(42); root=Path(__file__).resolve().parents[1]/'data/processed/TACK_Tunnel_B/full_frames'; a.output.mkdir(parents=True,exist_ok=False); ds={s:TileFrameDataset(root,s,s=='train') for s in ('train','val','test')}; pos=ds['train'].positive; n=sum(pos); w=[1/n if x else 1/(len(pos)-n) for x in pos]; sampler=WeightedRandomSampler(w,len(w),replacement=True,generator=torch.Generator().manual_seed(42)); loaders={'train':DataLoader(ds['train'],a.batch_size,sampler=sampler),'val':DataLoader(ds['val'],a.batch_size),'test':DataLoader(ds['test'],a.batch_size)}; device=torch.device('cuda'); model_cls = {'grid': MobileNetV3SmallGridRanker, 'shallow': MobileNetV3SmallShallowGridRanker, 'evidence': MobileNetV3SmallEvidenceGridRanker}[a.arch]; model=model_cls(True).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); hist=[]; best=-1
    for epoch in range(1,(1 if a.smoke else a.epochs)+1):
        model.train()
        for step,(x,y,_) in enumerate(loaders['train']):
            if a.smoke and step>=3: break
            opt.zero_grad(set_to_none=True); loss=ranking_loss(model(x.to(device)),y.to(device),a.pairwise_weight); loss.backward(); opt.step()
        metrics={'epoch':epoch}; metrics.update(evaluate(model,loaders['val'],device,1)); metrics['top2']=evaluate(model,loaders['val'],device,2); metrics['top3']=evaluate(model,loaders['val'],device,3); metrics['top6']=evaluate(model,loaders['val'],device,6); hist.append(metrics); write_json(a.output/'history.json',hist); print(json.dumps(metrics),flush=True)
        score=metrics['top6']['tile_recall'];
        if score>best: best=score; save_checkpoint(a.output/'best.pt',{'model':model.state_dict(),'epoch':epoch,'val':metrics})
    model.load_state_dict(torch.load(a.output/'best.pt',map_location=device,weights_only=True)['model']); report={'model':f'MobileNetV3-Small {a.arch} fixed-geometry dynamic ROI ranker','architecture':a.arch,'pairwise_weight':a.pairwise_weight,'history':hist,'test_top1':evaluate(model,loaders['test'],device,1),'test_top2':evaluate(model,loaders['test'],device,2),'test_top3':evaluate(model,loaders['test'],device,3),'test_top6':evaluate(model,loaders['test'],device,6),'smoke':a.smoke}; write_json(a.output/'metrics.json',report); print(json.dumps(report))
if __name__=='__main__': main()
