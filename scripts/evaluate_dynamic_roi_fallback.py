"""Calibrate a confidence-gap fallback for dynamic ROI ranking."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallGridRanker
from src.mobilenet_unet import MobileNetV3LargeUNet
TILES={'A':(0,0),'B':(1,0),'C':(2,0),'D':(3,0),'E':(0,1),'F':(1,1),'G':(2,1),'H':(3,1),'I':(0,2),'L':(1,2),'M':(2,2),'N':(3,2)}; ORDER=list(TILES)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ranker',type=Path,required=True); p.add_argument('--topk',type=int,default=6); p.add_argument('--gap',type=float,required=True); p.add_argument('--split',choices=['val','test'],default='test'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); project=Path(__file__).resolve().parents[1]
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
 dev=torch.device('cuda'); rank=MobileNetV3SmallGridRanker(False).to(dev); rank.load_state_dict(torch.load(a.ranker,map_location=dev,weights_only=True)['model']); rank.eval(); seg=MobileNetV3LargeUNet(False).to(dev); seg.load_state_dict(torch.load(project/'outputs/runs/tack_mobilenetv3_large/best.pt',map_location=dev,weights_only=True)['model']); seg.eval(); root=project/'data/processed/TACK_Tunnel_B/full_frames'; man=json.loads((root/'manifest.json').read_text()); totals={'tp':0,'fp':0,'fn':0}; selected=[]; fallbacks=0; times=[]
 for rec in [r for r in man['records'] if r['split']==a.split]:
  start=time.perf_counter()
  with Image.open(root/rec['image']) as im, Image.open(root/rec['mask']) as mm:
   image=im.convert('RGB'); truth=torch.from_numpy(np.asarray(mm)==40); crops=[]
   for t in ORDER:
    c,r=TILES[t]; arr=np.array(image.crop((c*512,r*512,(c+1)*512,r*512+512)),copy=True); crops.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
   with torch.no_grad(): scores=rank(torch.stack(crops).unsqueeze(0).to(dev))[0]; vals,idx=scores.topk(12); use=a.topk
   if float(vals[0]-vals[a.topk]) < a.gap: use=12; fallbacks+=1
   batch=torch.stack(crops[:use] if use==12 else [crops[i] for i in idx[:use]]).to(dev)
   with torch.no_grad(): logits=seg(batch)[:,0]
   torch.cuda.synchronize(dev); pred=torch.zeros((1536,2048),dtype=torch.bool); chosen=ORDER if use==12 else [ORDER[i] for i in idx[:use].tolist()]
   for j,t in enumerate(chosen): c,r=TILES[t]; pred[r*512:(r+1)*512,c*512:(c+1)*512]=logits[j].sigmoid().cpu()>=.5
   tp=(pred&truth).sum().item(); fp=(pred&~truth).sum().item(); fn=(~pred&truth).sum().item(); totals['tp']+=tp; totals['fp']+=fp; totals['fn']+=fn; selected.append(use); times.append(time.perf_counter()-start)
 tp,fp,fn=totals.values(); report={'split':a.split,'topk':a.topk,'gap':a.gap,'fallback_frames':fallbacks,'fallback_rate':fallbacks/len(selected),'mean_selected_tiles':sum(selected)/len(selected),'dice':2*tp/max(2*tp+fp+fn,1),'recall':tp/max(tp+fn,1),'negative_frames':sum(1 for r in man['records'] if r['split']==a.split and not r['positive']),'mean_end_to_end_seconds':sum(times)/len(times),'p95_end_to_end_seconds':float(np.percentile(times,95))}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
