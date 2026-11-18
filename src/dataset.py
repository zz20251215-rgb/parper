import json
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.recognizer import mask_to_box


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


class RecognizerDataset(Dataset):
    """Low-resolution preview dataset for the first-stage cascade gate."""

    def __init__(self, root, split_root, split, size=256):
        self.root = root
        self.size = size
        self.ids = (split_root / 'splits' / f'{split}.txt').read_text().split()
        samples = json.loads((split_root / 'manifest.json').read_text())['samples']
        self.positive = [bool(samples[k]['has_crack']) for k in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index]
        group, tile = key.rsplit('_', 1)
        with Image.open(self.root / '3_img' / f'{key}.png') as im:
            image = im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(self.root / '3_mask' / f'{group}_fuse_{tile}_1band.png') as im:
            mask = im.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        y = torch.from_numpy((np.array(mask) == 40).astype(np.float32))[None]
        present = y.flatten().any()
        return x, present, mask_to_box(y), key


class FullFrameRecognizerDataset(Dataset):
    """Recognizer previews built from the confirmed 2048x1536 frames."""

    def __init__(self, frame_root, split, preview_size=(512, 384)):
        self.frame_root = frame_root
        self.preview_size = tuple(preview_size)
        manifest = json.loads((frame_root / 'manifest.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.positive = [bool(r['positive']) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.frame_root / record['image']) as im:
            image = im.convert('RGB').resize(self.preview_size, Image.Resampling.BILINEAR)
        with Image.open(self.frame_root / record['mask']) as im:
            mask = np.asarray(im)
        y = torch.from_numpy((mask == 40).astype(np.float32))[None]
        x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        present = bool(record['positive'])
        return x, torch.tensor(present), mask_to_box(y), record['group']


class FullFrameCoarseDataset(Dataset):
    """Full-frame previews and coarse masks for the first-stage pixel gate."""

    def __init__(self, frame_root, split, preview_size=(512, 384)):
        self.frame_root = frame_root
        self.preview_size = tuple(preview_size)
        manifest = json.loads((frame_root / 'manifest.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.positive = [bool(r['positive']) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.frame_root / record['image']) as im:
            image = im.convert('RGB').resize(self.preview_size, Image.Resampling.BILINEAR)
        with Image.open(self.frame_root / record['mask']) as im:
            mask = im.resize(self.preview_size, Image.Resampling.NEAREST)
        x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        y = torch.from_numpy((np.array(mask) == 40).astype(np.float32))[None]
        return x, y, record['group']
