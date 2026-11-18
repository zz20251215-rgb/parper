"""Build a reviewable difficulty-set manifest from fixed-grid cascade evidence.

This script is read-only with respect to the dataset. It creates a new report
folder containing JSON rows and contact sheets made from existing full frames.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
TILES = {
    'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
    'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
    'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2),
}


def classify(row, cross_tile_groups):
    labels = []
    if row['pixels'] <= 0:
        return labels
    if row['pixels'] <= 1000:
        labels.append('low_pixel_mass')
    if row['edge_touch']:
        labels.append('edge_touch_proxy')
    if row['group'] in cross_tile_groups:
        labels.append('cross_tile_proxy')
    if not row['selected_grid']:
        labels.append('ranker_missed')
    elif row['recall'] is not None and row['recall'] < 0.30:
        labels.append('segmenter_low_recall')
    if row['selected_grid'] != row['selected_context']:
        labels.append('ranker_disagreement')
    return labels


def make_sheet(rows, frames_root, path, title, limit):
    rows = rows[:limit]
    if not rows:
        return
    thumb = 192
    cell_w, cell_h = thumb * 2, 232
    cols = 3
    header = 32
    canvas = Image.new('RGB', (cols * cell_w, header + ((len(rows) + cols - 1) // cols) * cell_h), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
        small = ImageFont.truetype('DejaVuSans.ttf', 10)
    except OSError:
        font = small = None
    draw.text((8, 8), title, fill='black', font=font)
    for index, row in enumerate(rows):
        group = row['group']
        x, y = (index % cols) * cell_w, header + (index // cols) * cell_h
        frame_path = frames_root / 'images' / f'{group}.png'
        mask_path = frames_root / 'masks' / f'{group}.png'
        col, r = TILES[row['tile']]
        box = (col * 512, r * 512, (col + 1) * 512, (r + 1) * 512)
        with Image.open(frame_path) as image, Image.open(mask_path) as mask:
            image = image.convert('RGB').crop(box).resize((thumb, thumb), Image.Resampling.BILINEAR)
            binary = np.asarray(mask.crop(box)) == 40
        overlay = np.asarray(image).astype(np.float32)
        # The mask is resized with nearest-neighbour to preserve the review signal.
        mask_small = np.asarray(Image.fromarray((binary * 255).astype(np.uint8)).resize((thumb, thumb), Image.Resampling.NEAREST)) > 0
        overlay[mask_small] = overlay[mask_small] * 0.25 + np.array([255, 35, 35]) * 0.75
        canvas.paste(image, (x, y))
        canvas.paste(Image.fromarray(overlay.astype(np.uint8)), (x + thumb, y))
        labels = ','.join(row['labels'])
        draw.text((x + 3, y + thumb + 3), f"{group}_{row['tile']} | {row['split']} | px={row['pixels']}", fill='black', font=small)
        draw.text((x + 3, y + thumb + 18), f"sel={row['selected_grid']} rec={row['recall'] if row['recall'] is not None else 'NA'} | {labels}", fill='black', font=small)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, default=PROJECT / 'outputs/reports/tile_difficulty_analysis.json')
    parser.add_argument('--frames-root', type=Path, default=PROJECT / 'data/processed/TACK_Tunnel_B/full_frames')
    parser.add_argument('--continuity', type=Path, default=PROJECT / 'outputs/reports/cross_tile_continuity_v1.json')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/reports/difficulty_set_v1')
    parser.add_argument('--per-category', type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    rows = json.loads(args.analysis.read_text())
    manifest = json.loads((args.frames_root / 'manifest.json').read_text())
    continuity = json.loads(args.continuity.read_text())
    cross_tile_groups = {record['group'] for record in continuity['records']}
    frame_by_group = {record['group']: record for record in manifest['records']}
    categories = defaultdict(list)
    review_rows = []
    for source in rows:
        labels = classify(source, cross_tile_groups)
        if not labels:
            continue
        row = dict(source)
        row['labels'] = labels
        row['source_image'] = f"images/{source['group']}.png"
        row['source_mask'] = f"masks/{source['group']}.png"
        row['tile_xy'] = list(TILES[source['tile']])
        row['review_status'] = 'needs_manual_review'
        if source['group'] not in frame_by_group:
            raise ValueError(f"Analysis group missing from frame manifest: {source['group']}")
        review_rows.append(row)
        for label in labels:
            categories[label].append(row)
    # Keep the most actionable examples first and make ordering deterministic.
    for label, items in categories.items():
        items.sort(key=lambda r: (r['split'], r['pixels'], r['group'], r['tile']))
    review_rows.sort(key=lambda r: (r['split'], len(r['labels']) * -1, r['pixels'], r['group'], r['tile']))
    args.output.mkdir(parents=True)
    (args.output / 'contact_sheets').mkdir()
    (args.output / 'manifest.json').write_text(json.dumps({
        'version': 1,
        'source_analysis': str(args.analysis),
        'frames_manifest': str(args.frames_root / 'manifest.json'),
        'selection_policy': {
            'low_pixel_mass': 'positive tile with <=1000 crack pixels',
            'edge_touch_proxy': 'positive pixels touch an 8-pixel tile edge band',
            'ranker_missed': 'positive tile not selected by original grid top-6 ranker',
            'segmenter_low_recall': 'selected tile with effective recall <0.30',
            'ranker_disagreement': 'original and context rankers disagree on top-6 selection',
            'cross_tile_proxy': 'annotation pixels appear on both sides of a fixed seam within the review band',
        },
        'rows': review_rows,
        'category_counts': {key: len(value) for key, value in sorted(categories.items())},
        'manual_review_required': True,
    }, ensure_ascii=False, indent=2) + '\n')
    for label, items in sorted(categories.items()):
        make_sheet(items, args.frames_root, args.output / 'contact_sheets' / f'{label}.png', label, args.per_category)
    summary = {
        'rows': len(review_rows),
        'category_counts': {key: len(value) for key, value in sorted(categories.items())},
        'split_counts': dict(Counter(row['split'] for row in review_rows)),
        'groups': len({row['group'] for row in review_rows}),
        'contact_sheets': sorted(str(p.relative_to(args.output)) for p in (args.output / 'contact_sheets').glob('*.png')),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
