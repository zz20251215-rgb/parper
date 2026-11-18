"""Evaluate dynamic ROI ranking with the frozen full-frame segmentation model."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamic_roi import MobileNetV3SmallGridRanker
from src.mobilenet_unet import MobileNetV3LargeUNet

TILES={'A':(0,0),'B':(1,0),'C':(2,0),'D':(3,0),'E':(0,1),'F':(1,1),'G':(2,1),'H':(3,1),'I':(0,2),'L':(1,2),'M':(2,2),'N':(3,2)}
ORDER=list(TILES)

def tensor_crop(image, tile):
    col,row=TILES[tile]; arr=np.array(image.crop((col*512,row*512,(col+1)*512,(row+1)*512)).convert('RGB'),copy=True)
    return torch.from_numpy(arr).permute(2,0,1).float()/255

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ranker',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--topk',type=int,required=True)
    a=p.parse_args(); project=Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    device=torch.device('cuda'); ranker=MobileNetV3SmallGridRanker(False).to(device); ranker.load_state_dict(torch.load(a.ranker,map_location=device,weights_only=True)['model']); ranker.eval()
    segmenter=MobileNetV3LargeUNet(False).to(device); segmenter.load_state_dict(torch.load(project/'outputs/runs/tack_mobilenetv3_large/best.pt',map_location=device,weights_only=True)['model']); segmenter.eval()
    root=project/'data/processed/TACK_Tunnel_B/full_frames'; manifest=json.loads((root/'manifest.json').read_text()); records=[r for r in manifest['records'] if r['split']=='test']; totals={'tp':0,'fp':0,'fn':0}; rows=[]
    for record in records:
        with Image.open(root/record['image']) as im, Image.open(root/record['mask']) as mm:
            image=im.convert('RGB'); truth=torch.from_numpy(np.asarray(mm)==40)
            x=torch.stack([tensor_crop(image,tile) for tile in ORDER]).unsqueeze(0).to(device)
            with torch.no_grad(): scores=ranker(x)[0]; indices=scores.topk(min(a.topk,12)).indices.tolist()
            selected=[ORDER[i] for i in indices]; batch=torch.stack([tensor_crop(image,t) for t in selected]).to(device)
            started=time.perf_counter()
            with torch.no_grad(): logits=segmenter(batch)[:,0]
            if device.type=='cuda': torch.cuda.synchronize(device)
            seconds=time.perf_counter()-started
            pred=torch.zeros((1536,2048),dtype=torch.bool)
            for j,t in enumerate(selected):
                col,row=TILES[t]; pred[row*512:(row+1)*512,col*512:(col+1)*512]=logits[j].sigmoid().cpu()>=.5
            tp=(pred&truth).sum().item(); fp=(pred&~truth).sum().item(); fn=(~pred&truth).sum().item(); totals['tp']+=tp; totals['fp']+=fp; totals['fn']+=fn; rows.append({'group':record['group'],'selected':selected,'seconds':seconds,'tp':tp,'fp':fp,'fn':fn})
    tp,fp,fn=totals.values(); report={'topk':a.topk,'mean_selected_tiles':a.topk,'dice':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'mean_segment_seconds':sum(r['seconds'] for r in rows)/max(len(rows),1),'totals':totals,'frames':rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
if __name__=='__main__': main()
