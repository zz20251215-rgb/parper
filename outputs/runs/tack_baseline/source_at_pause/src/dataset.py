import json
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, root, split_root, split, augment=False):
        self.root = root
        self.ids = (split_root / 'splits' / f'{split}.txt').read_text().split()
        samples = json.loads((split_root / 'manifest.json').read_text())['samples']
        self.positive = [samples[k]['has_crack'] for k in self.ids]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index]
        group, tile = key.rsplit('_', 1)
        with Image.open(self.root / '3_img' / f'{key}.png') as im:
            image = im.convert('RGB')
        with Image.open(self.root / '3_mask' / f'{group}_fuse_{tile}_1band.png') as im:
            mask = im.copy()
        if self.augment:
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            rotation = random.choice([None, Image.Transpose.ROTATE_90,
                                      Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
            if rotation is not None:
                image, mask = image.transpose(rotation), mask.transpose(rotation)
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        # Water and leaching are background for the crack-only task.
        y = torch.from_numpy((np.array(mask) == 40).astype(np.float32))[None]
        return x, y, key
