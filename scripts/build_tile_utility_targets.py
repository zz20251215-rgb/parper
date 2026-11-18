"""Build per-tile utility targets from the frozen second-stage segmenter."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mobilenet_unet import MobileNetV3LargeUNet

TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}
ORDER = list(TILES)


def tile_tensor(image, tile):
    col, row = TILES[tile]
    array = np.array(image.crop((col * 512, row * 512, (col + 1) * 512,
                                 (row + 1) * 512)).convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmenter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=24)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    project = Path(__file__).resolve().parents[1]
    root = project / 'data/processed/TACK_Tunnel_B/full_frames'
    manifest = json.loads((root / 'manifest.json').read_text())
    device = torch.device('cuda')
    model = MobileNetV3LargeUNet(False).to(device)
    model.load_state_dict(torch.load(args.segmenter, map_location=device, weights_only=True)['model'])
    model.eval()
    records = []
    for record in manifest['records']:
        with Image.open(root / record['image']) as image_file, Image.open(root / record['mask']) as mask_file:
            image = image_file.convert('RGB')
            mask = np.asarray(mask_file) == 40
            tiles = [tile_tensor(image, tile) for tile in ORDER]
            predictions = []
            with torch.no_grad():
                for start in range(0, len(tiles), args.batch_size):
                    batch = torch.stack(tiles[start:start + args.batch_size]).to(device)
                    predictions.extend((model(batch)[:, 0].sigmoid().cpu().numpy() >= 0.5))
            utility = {}
            for tile, prediction in zip(ORDER, predictions):
                col, row = TILES[tile]
                truth = mask[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512]
                tp = int(np.logical_and(prediction, truth).sum())
                fp = int(np.logical_and(prediction, ~truth).sum())
                fn = int(np.logical_and(~prediction, truth).sum())
                denom = 2 * tp + fp + fn
                dice = 2 * tp / denom if denom else 0.0
                false_area = fp / float(512 * 512)
                utility[tile] = {
                    'tp': tp, 'fp': fp, 'fn': fn, 'dice': dice,
                    'utility': dice if truth.any() else -false_area,
                    'has_crack': bool(truth.any()),
                }
            records.append({'group': record['group'], 'split': record['split'], 'tiles': utility})
    payload = {'segmenter': str(args.segmenter), 'definition': 'tile Dice for positives; negative false-area penalty', 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
