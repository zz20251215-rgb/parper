import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.mobilenet_unet import MobileNetV3LargeRefineUNet
root=Path(__file__).resolve().parents[3]
torch.set_num_threads(4)
state=torch.load(root/'outputs/runs/tack_mobilenetv3_large_refine_v1/best.pt',map_location='cpu',weights_only=True)['model']
manifest=json.loads((root/'data/processed/TACK_Tunnel_B/full_frames/manifest.json').read_text())
record=next(r for r in manifest['records'] if r['split']=='val')
with Image.open(root/'data/processed/TACK_Tunnel_B/full_frames'/record['image']) as im:
 real=torch.from_numpy(np.array(im.convert('RGB').crop((0,0,512,512)),copy=True)).permute(2,0,1).float().unsqueeze(0)/255
inputs=[(f'random_{s}',torch.rand((1,3,512,512),generator=torch.Generator().manual_seed(s))) for s in [42,7,8]]+[('real_val_A',real)]
report={'purpose':'Read-only audit; independent eager and saved JIT models; no weight conversion or training','input_shape':[1,3,512,512],'torch':torch.__version__,'checks':[]}
for device in ['cpu','cuda']:
 e=MobileNetV3LargeRefineUNet(False).eval(); e.load_state_dict(state); e=e.to(device)
 t=torch.jit.load(str(root/'outputs/runs/tack_mobilenetv3_large_refine_v1/deploy/refine_fp32.ts'),map_location=device).eval()
 for name,x in inputs:
  x=x.to(device)
  with torch.inference_mode(): a=e(x); b=t(x)
  delta=(a-b).abs(); passed=True
  try: torch.testing.assert_close(b,a,rtol=1e-5,atol=1e-6)
  except AssertionError: passed=False
  row=dict(device=device,input=name,max_abs=delta.max().item(),mean_abs=delta.mean().item(),finite=bool(torch.isfinite(b).all()),mask_disagreement_fraction=((a>=0)!=(b>=0)).float().mean().item(),assert_close_passed=passed)
  report['checks'].append(row); print(json.dumps(row),flush=True)
 del e,t
report['limitation']='Batch=1, four sequential inputs, default backend flags; not target-device or full validation. Root cause not established.'
(Path(__file__).with_name('parity_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
