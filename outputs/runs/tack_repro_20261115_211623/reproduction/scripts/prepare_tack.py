import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    project = Path(__file__).resolve().parents[1]
    source = project / 'data/TACK_Tunnel_Data'
    output = project / 'data/processed/TACK_Tunnel_B'
    groups = defaultdict(list)
    samples = {}
    hashes = {}
    for image_path in sorted((source / '3_img').glob('TB_*.png')):
        key = image_path.stem
        group, tile = key.rsplit('_', 1)
        mask_path = source / '3_mask' / f'{group}_fuse_{tile}_1band.png'
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.load()
            values = np.array(mask)
            if image.size != (512, 512) or mask.size != image.size or values.ndim != 2:
                raise ValueError(f'Invalid image/mask dimensions: {key}')
            if not set(np.unique(values)) <= {0, 40, 160, 200}:
                raise ValueError(f'Unexpected mask values: {key}')
            digest = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            if digest in hashes and hashes[digest] != group:
                raise ValueError(f'Identical pixels in different source groups: {key}')
            hashes[digest] = group
        groups[group].append(key)
        samples[key] = {'group': group, 'has_crack': bool((values == 40).any()),
                        'crack_pixels': int((values == 40).sum())}
    if len(groups) != 128 or len(samples) != 1536:
        raise ValueError('Expected 128 source images and 1536 tiles')
    expected_tiles = set('ABCDEFGHILMN')
    if any({k.rsplit('_', 1)[1] for k in ids} != expected_tiles for ids in groups.values()):
        raise ValueError('Incomplete source-image tile group')
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    split_groups = {'train': keys[:90], 'val': keys[90:115], 'test': keys[115:]}
    splits = {s: sorted(k for g in gs for k in groups[g]) for s, gs in split_groups.items()}
    manifest = {'source': 'data/TACK_Tunnel_Data', 'seed': 42, 'crack_value': 40,
                'grouping': 'Full filename prefix before the final tile suffix',
                'samples': samples, 'summary': {}}
    for split, ids in splits.items():
        positives = sum(samples[k]['has_crack'] for k in ids)
        pixels = sum(samples[k]['crack_pixels'] for k in ids)
        if not 0 < positives < len(ids):
            raise ValueError(f'Split must contain positive and negative images: {split}')
        manifest['summary'][split] = {'groups': len(split_groups[split]), 'images': len(ids),
                                      'positive': positives, 'negative': len(ids) - positives,
                                      'crack_pixel_ratio': pixels / (len(ids) * 512 * 512)}
    # An existing different split must be reviewed rather than silently replaced.
    paths = {output / 'splits' / f'{s}.txt': ''.join(k + '\n' for k in ids)
             for s, ids in splits.items()}
    paths[output / 'manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    for path, content in paths.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite different prepared data: {path}')
    for path, content in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps(manifest['summary'], indent=2))


if __name__ == '__main__':
    main()
