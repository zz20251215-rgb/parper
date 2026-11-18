import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def components(binary):
    visited = np.zeros_like(binary)
    areas = []
    height, width = binary.shape
    for row, col in np.argwhere(binary):
        if visited[row, col]:
            continue
        queue = deque([(int(row), int(col))])
        visited[row, col] = True
        area = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and binary[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        areas.append(area)
    return areas


def preview(source, record, size):
    with Image.open(source / '3_img' / f"{record['id']}.png") as image:
        rgb = np.array(image.convert('RGB'))
    with Image.open(source / '3_mask' / record['mask_file']) as mask:
        binary = np.array(mask) == 40
    overlay = rgb.astype(float)
    overlay[binary] = overlay[binary] * 0.25 + np.array([255, 35, 35]) * 0.75
    return [Image.fromarray(rgb).resize((size, size)),
            Image.fromarray(overlay.astype('uint8')).resize((size, size)),
            Image.fromarray((binary * 255).astype('uint8')).convert('RGB').resize((size, size), Image.Resampling.NEAREST)]


def sheet(source, records, path, large=False):
    size = 512 if large else 160
    columns = 1 if large else 3
    rows = (len(records) + columns - 1) // columns
    header = 50 if large else 42
    canvas = Image.new('RGB', (columns * size * 3, rows * (size + header)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 18 if large else 11)
    for index, record in enumerate(records):
        x = (index % columns) * size * 3
        y = (index // columns) * (size + header)
        label = f"{record['id']} | {record['split']} | crack {record['crack_pixels']} px"
        draw.text((x + 5, y + 3), label, font=font, fill='black')
        draw.text((x + 5, y + header // 2), 'Original / red overlay / crack mask (value 40)', font=font, fill='black')
        for j, image in enumerate(preview(source, record, size)):
            canvas.paste(image, (x + j * size, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/reports' / ('tack_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    if hasattr(os, 'nice'):
        os.nice(10)
    source = PROJECT / 'data/TACK_Tunnel_Data'
    prepared = PROJECT / 'data/processed/TACK_Tunnel_B'
    args.output.mkdir(parents=True, exist_ok=False)
    errors = []
    warnings = []
    manifest = json.loads((prepared / 'manifest.json').read_text())
    splits = {name: (prepared / 'splits' / f'{name}.txt').read_text().split() for name in ['train', 'val', 'test']}
    membership = defaultdict(list)
    for split, ids in splits.items():
        for key in ids:
            membership[key].append(split)
        if len(ids) != len(set(ids)):
            errors.append(f'{split} contains repeated IDs')
    records = []
    hashes = defaultdict(list)
    group_membership = defaultdict(set)
    mask_values = Counter()
    expected_masks = set()
    dhashes = {}
    for image_path in sorted((source / '3_img').glob('TB_*.png')):
        key = image_path.stem
        group, tile = key.rsplit('_', 1)
        mask_file = f'{group}_fuse_{tile}_1band.png'
        expected_masks.add(mask_file)
        try:
            with Image.open(image_path) as im:
                im.load()
                rgb = np.array(im.convert('RGB'))
                image_size, image_mode = im.size, im.mode
                small = np.array(im.convert('L').resize((9, 8)), dtype=np.int16)
            with Image.open(source / '3_mask' / mask_file) as im:
                im.load()
                mask, mode = np.array(im), im.mode
            if image_size != (512, 512) or mask.shape != (512, 512):
                raise ValueError(f'Unexpected dimensions {image_size} / {mask.shape}')
            values, counts = np.unique(mask, return_counts=True)
            if not set(values.tolist()) <= {0, 40, 160, 200}:
                raise ValueError(f'Unexpected mask values {values.tolist()}')
            mask_values.update({int(v): int(n) for v, n in zip(values, counts)})
            binary = mask == 40
            yy, xx = np.nonzero(binary)
            bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if len(xx) else None
            edge_only = bool(len(xx) and np.all((xx < 4) | (xx >= 508) | (yy < 4) | (yy >= 508)))
            gray = rgb.mean(axis=2)
            areas = components(binary) if binary.any() else []
            ring = np.array(Image.fromarray(binary.astype('uint8')).filter(ImageFilter.MaxFilter(9))) > 0
            ring &= ~binary
            contrast = float(gray[ring].mean() - gray[binary].mean()) if ring.any() else None
            split_list = membership.get(key, [])
            if len(split_list) != 1:
                errors.append(f'{key}: split membership {split_list}')
            split = split_list[0] if split_list else 'unassigned'
            group_membership[group].update(split_list)
            record = {'id': key, 'group': group, 'split': split, 'mask_file': mask_file,
                      'image_mode': image_mode, 'mask_mode': mode,
                      'has_crack': bool(binary.any()), 'crack_pixels': int(binary.sum()),
                      'crack_ratio': float(binary.mean()), 'water_pixels': int((mask == 160).sum()),
                      'crack_bbox_xyxy_exclusive': bbox, 'all_crack_in_edge_band_4px': edge_only,
                      'leaching_pixels': int((mask == 200).sum()),
                      'image_std': float(gray.std()), 'image_mean': float(gray.mean()),
                      'rgb_channels_identical': bool(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])),
                      'components_8_connected': len(areas), 'tiny_components_le_3_pixels': sum(a <= 3 for a in areas),
                      'largest_component_fraction': max(areas) / sum(areas) if areas else None,
                      'local_ring_minus_crack_intensity': contrast}
            expected = manifest['samples'].get(key)
            if expected is None or expected['has_crack'] != record['has_crack'] or expected['crack_pixels'] != record['crack_pixels'] or expected['group'] != group:
                errors.append(f'{key}: manifest disagrees with actual mask')
            hashes[hashlib.sha256(rgb.tobytes()).hexdigest()].append(key)
            bits = (small[:, 1:] > small[:, :-1]).ravel()
            dhashes[key] = int.from_bytes(np.packbits(bits).tobytes(), 'big')
            records.append(record)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{key}: {type(exc).__name__}: {exc}')
    actual_ids = {x['id'] for x in records}
    if actual_ids != set(membership):
        errors.append('Split IDs do not exactly cover decoded images')
    if actual_ids != set(manifest['samples']):
        errors.append('Manifest IDs do not exactly cover decoded images')
    orphan_masks = sorted(p.name for p in (source / '3_mask').glob('TB_*.png') if p.name not in expected_masks)
    if orphan_masks:
        errors.append(f'{len(orphan_masks)} masks have no matching image')
    leaking_groups = {g: sorted(s) for g, s in group_membership.items() if len(s) != 1}
    if leaking_groups:
        errors.append('Source image groups cross splits')
    group_tiles = defaultdict(set)
    for record in records:
        group_tiles[record['group']].add(record['id'].rsplit('_', 1)[1])
    incomplete = {g: sorted(v) for g, v in group_tiles.items() if v != set('ABCDEFGHILMN')}
    if incomplete:
        errors.append('Source image groups have incomplete tile suffixes')
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    conflicting_duplicates = []
    for ids in duplicate_groups:
        samples = [next(x for x in records if x['id'] == key) for key in ids]
        if len({x['split'] for x in samples}) > 1:
            errors.append(f'Exact duplicate images cross splits: {ids}')
        masks = set()
        for sample in samples:
            with Image.open(source / '3_mask' / sample['mask_file']) as im:
                masks.add(hashlib.sha256(im.tobytes()).hexdigest())
        if len(masks) > 1:
            conflicting_duplicates.append(ids)
            errors.append(f'Identical image has different labels: {ids}')
    stats = {}
    for split, ids in splits.items():
        selected = [x for x in records if x['id'] in set(ids)]
        positives = [x for x in selected if x['has_crack']]
        ratios = [x['crack_ratio'] for x in positives]
        stats[split] = {'images': len(selected), 'positive': len(positives), 'negative': len(selected) - len(positives),
                        'groups': len({x['group'] for x in selected}),
                        'positive_groups': len({x['group'] for x in positives}),
                        'crack_pixel_ratio': sum(x['crack_pixels'] for x in selected) / max(len(selected) * 512 * 512, 1),
                        'positive_image_mean_crack_ratio': float(np.mean(ratios)) if ratios else 0,
                        'positive_image_median_crack_pixels': float(np.median([x['crack_pixels'] for x in positives])) if positives else 0,
                        'cameras': dict(Counter(x['id'].split('_')[1] for x in selected))}
        for field in ['images', 'positive', 'negative', 'groups']:
            if stats[split][field] != manifest['summary'][split][field]:
                errors.append(f'{split}: manifest summary {field} mismatch')
        if not positives or len(positives) == len(selected):
            errors.append(f'{split}: lacks positive or negative samples')
    # Test the actual training loader on CPU, independently against decoded labels.
    import torch
    from unittest.mock import patch
    from src.dataset import CrackDataset
    torch.set_num_threads(2)
    loader_checked = 0
    for split in splits:
        ds = CrackDataset(source, prepared, split)
        for i in range(len(ds)):
            try:
                x, y, key = ds[i]
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f'{ds.ids[i]}: loader failed: {type(exc).__name__}: {exc}')
                continue
            group, tile = key.rsplit('_', 1)
            with Image.open(source / '3_mask' / f'{group}_fuse_{tile}_1band.png') as im:
                expected = np.array(im) == 40
            if x.shape != (3, 512, 512) or y.shape != (1, 512, 512) or not np.array_equal(y[0].numpy(), expected) or bool(y.any()) != ds.positive[i]:
                errors.append(f'{key}: actual loader tensor or label mismatch')
            if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                errors.append(f'{key}: invalid normalized input')
            loader_checked += 1
    plain = CrackDataset(source, prepared, 'train')
    augmented = CrackDataset(source, prepared, 'train', augment=True)
    for index in [plain.positive.index(True), plain.positive.index(False)]:
        x, y, key = plain[index]
        for rotation, turns in [(None, 0), (Image.Transpose.ROTATE_90, 1), (Image.Transpose.ROTATE_180, 2), (Image.Transpose.ROTATE_270, 3)]:
            with patch('src.dataset.random.random', return_value=0), patch('src.dataset.random.choice', return_value=rotation):
                xx, yy, _ = augmented[index]
            if not torch.equal(xx, torch.rot90(x.flip(-1), turns, (1, 2))) or not torch.equal(yy, torch.rot90(y.flip(-1), turns, (1, 2))):
                errors.append(f'{key}: image/mask geometric augmentation differs')
    positive_records = [x for x in records if x['has_crack']]
    tiny = sorted(positive_records, key=lambda x: x['crack_pixels'])[:12]
    contrast_candidates = sorted((x for x in positive_records if x['local_ring_minus_crack_intensity'] is not None), key=lambda x: x['local_ring_minus_crack_intensity'])[:12]
    fragmented = sorted(positive_records, key=lambda x: x['components_8_connected'], reverse=True)[:12]
    by_id = {x['id']: x for x in records}
    pairs = []
    keys = sorted(dhashes)
    for i, a in enumerate(keys):
        for b in keys[:i]:
            if by_id[a]['split'] == by_id[b]['split'] or by_id[a]['group'] == by_id[b]['group']:
                continue
            distance = (dhashes[a] ^ dhashes[b]).bit_count()
            if distance <= 6:
                pairs.append({'ids': [a, b], 'hamming_distance': distance})
    pairs.sort(key=lambda x: (x['hamming_distance'], x['ids']))
    if pairs:
        warnings.append('Cross-split perceptual-hash candidates require visual review; similarity alone does not prove leakage')
    warnings.append('Image-level 1:1 sampling does not balance pixels; see estimated balanced-sampling foreground fraction')
    report = {'status': 'FAIL' if errors else 'PASS_WITH_REVIEW_ITEMS', 'errors': errors, 'warnings': warnings,
              'source': str(source.relative_to(PROJECT)), 'samples': len(records), 'mask_pixel_values': dict(mask_values),
              'split_statistics': stats, 'source_group_leakage': leaking_groups, 'incomplete_tile_groups': incomplete,
              'orphan_masks': orphan_masks, 'exact_duplicate_groups': duplicate_groups,
              'conflicting_duplicate_labels': conflicting_duplicates, 'actual_loader_samples_checked': loader_checked,
              'augmentation_cases_checked': 8, 'rgb_grayscale_images': sum(x['rgb_channels_identical'] for x in records),
              'tiny_positive_counts': {str(n): sum(x['crack_pixels'] <= n for x in positive_records) for n in [3, 10, 50, 100, 200]},
              'expected_foreground_fraction_balanced_training': stats['train']['positive_image_mean_crack_ratio'] / 2,
              'crack_pixels_quantiles': {str(q): float(np.quantile([x['crack_pixels'] for x in positive_records], q)) for q in [0, .1, .5, .9, 1]},
              'tiny_crack_candidates': [x['id'] for x in tiny], 'low_contrast_candidates': [x['id'] for x in contrast_candidates],
              'fragmentation_candidates': [x['id'] for x in fragmented],
              'cross_split_dhash_candidate_count': len(pairs), 'cross_split_dhash_top_candidates': pairs[:24],
              'limitations': ['Only exact source-image groups and exact duplicates are proven; different cameras/frames may share a scene.',
                              'Intensity contrast, connected components and perceptual hashes are review cues, not proof of annotation error.',
                              'Not all images are manually reviewed; missing cracks require expert inspection.'],
              'samples_detail': records}
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    previews = args.output / 'previews'
    previews.mkdir()
    for i in range(0, len(positive_records), 12):
        sheet(source, positive_records[i:i + 12], previews / f'all_positive_{i // 12 + 1:02d}.jpg')
    for name, selected in [('tiny_cracks', tiny), ('low_contrast', contrast_candidates), ('fragmented', fragmented)]:
        sheet(source, selected, previews / f'{name}.jpg')
    rng = random.Random(42)
    negative = [x for x in records if not x['has_crack']]
    sheet(source, rng.sample(negative, 12), previews / 'negative_samples.jpg')
    examples = args.output / 'examples'
    examples.mkdir()
    selected = {x['id']: x for x in tiny[:2] + contrast_candidates[:2] + fragmented[:2]}
    for record in selected.values():
        sheet(source, [record], examples / f"{record['id']}.png", large=True)
    if pairs:
        canvas = Image.new('RGB', (800, 6 * 230), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, pair in enumerate(pairs[:6]):
            for j, key in enumerate(pair['ids']):
                with Image.open(source / '3_img' / f'{key}.png') as im:
                    canvas.paste(im.convert('RGB').resize((224, 224)), (j * 224, i * 230))
            draw.text((455, i * 230 + 10), '\n'.join(pair['ids']) + f"\ndHash distance {pair['hamming_distance']}", fill='black')
        canvas.save(previews / 'cross_split_similar.jpg')
    short = {k: v for k, v in report.items() if k != 'samples_detail'}
    print(json.dumps(short, indent=2))
    print('Report directory:', args.output)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
