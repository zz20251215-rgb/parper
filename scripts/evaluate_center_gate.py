"""Evaluate center heatmap candidates against crack-pixel coverage."""
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_multibox_gate import MultiBoxDataset, collate
from src.multibox_gate import MobileNetV3SmallCenterGate, decode_centers

def evaluate(model, dataset, device, threshold, expansion):
    loader=DataLoader(dataset,8,collate_fn=collate); total=covered=0; hit=0; frames=[]
    with torch.no_grad():
        for x, truth, present, groups in loader:
            candidates=decode_centers(model(x.to(device)),threshold,64)
            for idx,group in enumerate(groups):
                boxes=candidates[idx]; covered_count=0
                for box in truth[idx]:
                    best=0
                    for _,c in boxes:
                        cx,cy,w,h=box; bx,by,bw,bh=c.tolist(); bw*=expansion; bh*=expansion
                        x0=max(cx-w/2,bx-bw/2); y0=max(cy-h/2,by-bh/2); x1=min(cx+w/2,bx+bw/2); y1=min(cy+h/2,by+bh/2)
                        inter=max(0,x1-x0)*max(0,y1-y0); union=w*h+bw*bh-inter; best=max(best,inter/max(union,1e-8))
                    covered_count += best >= .1
                total += len(truth[idx]); covered += covered_count; hit += bool(truth[idx]) and covered_count == len(truth[idx])
                frames.append({'group':group,'truth_boxes':len(truth[idx]),'candidates':len(boxes),'covered_boxes':covered_count})
    positives=sum(bool(x) for x in dataset.boxes)
    return {'threshold':threshold,'expansion':expansion,'mean_candidates':sum(x['candidates'] for x in frames)/max(len(frames),1),'box_coverage':covered/max(total,1),'all_box_frame_recall':hit/max(positives,1),'frames':frames}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--threshold',type=float,default=.3); p.add_argument('--expansion',type=float,default=1.5); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    project=Path(__file__).resolve().parents[1]; device=torch.device('cuda'); model=MobileNetV3SmallCenterGate(False).to(device); model.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True)['model']); model.eval(); d=MultiBoxDataset(project/'data/processed/TACK_Tunnel_B/full_frames',a.split,(512,384)); r=evaluate(model,d,device,a.threshold,a.expansion); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({k:v for k,v in r.items() if k!='frames'},indent=2))
if __name__=='__main__': main()
