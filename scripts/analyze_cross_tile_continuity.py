"""Find annotation evidence that a crack crosses a fixed tile seam.

This is a geometric proxy for review, not a morphology label. A seam is
flagged only when crack pixels appear on both sides within a narrow band and
the two sides have a nearby row/column coordinate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
ORDER = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N')


def seam_candidates(binary, band=8, gap=3):
    rows = []
    # Vertical seams x=512,1024,1536; horizontal seams y=512,1024.
    for x in (512, 1024, 1536):
        left = binary[:, x - band:x]
        right = binary[:, x:x + band]
        left_y = np.flatnonzero(left.any(axis=1))
        right_y = np.flatnonzero(right.any(axis=1))
        if len(left_y) and len(right_y):
            distances = np.abs(left_y[:, None] - right_y[None, :])
            if int(distances.min()) <= gap:
                rows.append({'orientation': 'vertical', 'coordinate': x,
                             'left_pixels': int(left.sum()), 'right_pixels': int(right.sum()),
                             'nearest_axis_gap': int(distances.min())})
    for y in (512, 1024):
        top = binary[y - band:y, :]
        bottom = binary[y:y + band, :]
        top_x = np.flatnonzero(top.any(axis=0))
        bottom_x = np.flatnonzero(bottom.any(axis=0))
        if len(top_x) and len(bottom_x):
            distances = np.abs(top_x[:, None] - bottom_x[None, :])
            if int(distances.min()) <= gap:
                rows.append({'orientation': 'horizontal', 'coordinate': y,
                             'top_pixels': int(top.sum()), 'bottom_pixels': int(bottom.sum()),
                             'nearest_axis_gap': int(distances.min())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-root', type=Path, default=PROJECT / 'data/processed/TACK_Tunnel_B/full_frames')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/reports/cross_tile_continuity_v1.json')
    parser.add_argument('--band', type=int, default=8)
    parser.add_argument('--gap', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    manifest = json.loads((args.frames_root / 'manifest.json').read_text())
    records = []
    for record in manifest['records']:
        with Image.open(args.frames_root / record['mask']) as mask:
            binary = np.asarray(mask) == 40
        candidates = seam_candidates(binary, args.band, args.gap)
        if candidates:
            records.append({'group': record['group'], 'split': record['split'],
                            'crack_pixels': int(binary.sum()), 'seams': candidates,
                            'image': record['image'], 'mask': record['mask']})
    output = {'version': 1, 'band': args.band, 'gap': args.gap,
              'definition': 'crack pixels on both sides of the same seam within the band and axis gap; review proxy only',
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'frames': len(records), 'seams': sum(len(r['seams']) for r in records),
                      'splits': {s: sum(r['split'] == s for r in records) for s in ('train', 'val', 'test')}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
