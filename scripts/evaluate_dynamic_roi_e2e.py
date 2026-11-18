"""End-to-end timing for dynamic ROI ranking and segmentation reconstruction."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallGridRanker
from src.mobilenet_unet import MobileNetV3LargeUNet
TILES={'A':(0,0),'B':(1,0),'C':(2,0),'D':(3,0),'E':(0,1),'F':(1,1),'G':(2,1),'H':(3,1),'I':(0,2),'L':(1,2),'M':(2,2),'N':(3,2)}; ORDER=list(TILES)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ranker',type=Path,required=True); p.add_argument('--topk',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); project=Path(__file__).resolve().parents[1]
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
 dev=torch.device('cuda'); ranker=MobileNetV3SmallGridRanker(False).to(dev); ranker.load_state_dict(torch.load(a.ranker,map_location=dev,weights_only=True)['model']); ranker.eval(); seg=MobileNetV3LargeUNet(False).to(dev); seg.load_state_dict(torch.load(project/'outputs/runs/tack_mobilenetv3_large/best.pt',map_location=dev,weights_only=True)['model']); seg.eval(); root=project/'data/processed/TACK_Tunnel_B/full_frames'; man=json.loads((root/'manifest.json').read_text()); rows=[]
 for rec in [r for r in man['records'] if r['split']=='test']:
  started=time.perf_counter()
  with Image.open(root/rec['image']) as im: image=im.convert('RGB'); crops=[]
  for t in ORDER:
   c,r=TILES[t]; arr=np.array(image.crop((c*512,r*512,(c+1)*512,(r+1)*512)),copy=True); crops.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
  with torch.no_grad(): scores=ranker(torch.stack(crops).unsqueeze(0).to(dev))[0]; idx=scores.topk(min(a.topk,12)).indices.tolist(); batch=torch.stack([crops[i] for i in idx]).to(dev); _=seg(batch)
  torch.cuda.synchronize(dev); rows.append(time.perf_counter()-started)
 report={'topk':a.topk,'mean_end_to_end_seconds':sum(rows)/len(rows),'p95_end_to_end_seconds':float(np.percentile(rows,95)),'frames':len(rows)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
