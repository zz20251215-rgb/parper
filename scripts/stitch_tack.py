"""Build 2048x1536 TACK frames from the confirmed 3x4 tile layout."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


TILES = {
    'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
    'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
    'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2),
}
TILE_ORDER = tuple(TILES)
TILE_SIZE = 512
FRAME_SIZE = (2048, 1536)


def read_groups(source, split_root):
    manifest = json.loads((split_root / 'manifest.json').read_text())
    groups = {}
    for split in ('train', 'val', 'test'):
        for key in (split_root / 'splits' / f'{split}.txt').read_text().split():
            group, tile = key.rsplit('_', 1)
            if tile not in TILES:
                raise ValueError(f'Unknown tile suffix: {key}')
            if group in groups and groups[group]['split'] != split:
                raise ValueError(f'Group crosses splits: {group}')
            groups.setdefault(group, {'split': split, 'keys': {}})['keys'][tile] = key
    if len(groups) != 128:
        raise ValueError(f'Expected 128 groups, found {len(groups)}')
    for group, info in groups.items():
        if set(info['keys']) != set(TILES):
            raise ValueError(f'Group {group} does not contain exactly A B C D E F G H I L M N')
        for tile, key in info['keys'].items():
            sample = manifest['samples'][key]
            image = source / '3_img' / f'{key}.png'
            mask = source / '3_mask' / f'{group}_fuse_{tile}_1band.png'
            if not image.exists() or not mask.exists():
                raise FileNotFoundError(f'Missing pair for {key}')
            info.setdefault('samples', {})[tile] = {
                'key': key,
                'has_crack': bool(sample['has_crack']),
                'crack_pixels': int(sample['crack_pixels']),
                'xy': [TILES[tile][0] * TILE_SIZE, TILES[tile][1] * TILE_SIZE],
            }
    return groups


def build_frame(source, group, info):
    image_frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    mask_frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.uint8)
    for tile, key in info['keys'].items():
        x, y = TILES[tile][0] * TILE_SIZE, TILES[tile][1] * TILE_SIZE
        with Image.open(source / '3_img' / f'{key}.png') as image:
            image_array = np.asarray(image.convert('RGB'))
        with Image.open(source / '3_mask' / f'{group}_fuse_{tile}_1band.png') as mask:
            mask_array = np.asarray(mask)
        if image_array.shape != (TILE_SIZE, TILE_SIZE, 3):
            raise ValueError(f'Invalid image shape for {key}: {image_array.shape}')
        if mask_array.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f'Invalid mask shape for {key}: {mask_array.shape}')
        image_frame[y:y + TILE_SIZE, x:x + TILE_SIZE] = image_array
        mask_frame[y:y + TILE_SIZE, x:x + TILE_SIZE] = mask_array
    return image_frame, mask_frame


def write_if_new(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as existing:
            if not np.array_equal(np.asarray(existing), array):
                raise FileExistsError(f'Refusing to overwrite different generated file: {path}')
        return False
    Image.fromarray(array).save(path)
    return True


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=project / 'data/TACK_Tunnel_Data')
    parser.add_argument('--split-root', type=Path, default=project / 'data/processed/TACK_Tunnel_B')
    parser.add_argument('--output', type=Path,
                        default=project / 'data/processed/TACK_Tunnel_B/full_frames')
    args = parser.parse_args()
    groups = read_groups(args.source, args.split_root)
    records = []
    created = 0
    for group in sorted(groups):
        info = groups[group]
        image, mask = build_frame(args.source, group, info)
        image_path = args.output / 'images' / f'{group}.png'
        mask_path = args.output / 'masks' / f'{group}.png'
        created += write_if_new(image_path, image)
        created += write_if_new(mask_path, mask)
        crack_pixels = int((mask == 40).sum())
        records.append({
            'group': group,
            'split': info['split'],
            'image': str(image_path.relative_to(args.output)),
            'mask': str(mask_path.relative_to(args.output)),
            'width': FRAME_SIZE[0],
            'height': FRAME_SIZE[1],
            'tile_size': TILE_SIZE,
            'tile_order': TILE_ORDER,
            'positive': bool(crack_pixels),
            'crack_pixels': crack_pixels,
            'positive_tiles': [t for t in TILE_ORDER if info['samples'][t]['has_crack']],
            'tiles': info['samples'],
        })
    output_manifest = {
        'layout': ['ABCD', 'EFGH', 'ILMN'],
        'tile_size': TILE_SIZE,
        'frame_size': list(FRAME_SIZE),
        'source': str(args.source),
        'splits': {split: sum(r['split'] == split for r in records)
                   for split in ('train', 'val', 'test')},
        'positive_frames': sum(r['positive'] for r in records),
        'negative_frames': sum(not r['positive'] for r in records),
        'records': records,
    }
    manifest_path = args.output / 'manifest.json'
    text = json.dumps(output_manifest, ensure_ascii=False, indent=2) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != text:
        raise FileExistsError(f'Refusing to overwrite different generated file: {manifest_path}')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(text)
    print(json.dumps({
        'groups': len(records),
        'splits': output_manifest['splits'],
        'positive_frames': output_manifest['positive_frames'],
        'negative_frames': output_manifest['negative_frames'],
        'created_files': created,
        'manifest': str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
