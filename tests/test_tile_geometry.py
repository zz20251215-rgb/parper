import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.train_dynamic_roi import TileFrameDataset as PresenceDataset
from scripts.train_mass_ranker import TileFrameDataset as MassDataset
from scripts.train_utility_ranker import UtilityDataset
from src.tile_geometry import MIRRORED_ORDER, ORDER, TILES, transform_frame

ROOT = Path(__file__).resolve().parents[1] / 'data/processed/TACK_Tunnel_B/full_frames'


class TileGeometryTests(unittest.TestCase):
    def test_whole_frame_mirror_and_target_permutation(self):
        values = np.arange(2048 * 1536, dtype=np.uint32).reshape(1536, 2048)
        image = Image.fromarray(values.astype(np.int32))
        transformed, sources = transform_frame(image, True)
        np.testing.assert_array_equal(np.array(transformed), values[:, ::-1])
        self.assertEqual(sources, ('D', 'C', 'B', 'A', 'H', 'G', 'F', 'E', 'N', 'M', 'L', 'I'))
        reconstructed = Image.new(image.mode, image.size)
        for destination, source in zip(ORDER, sources):
            x, y = TILES[source]
            crop = ImageOps.mirror(image.crop((x * 512, y * 512, (x + 1) * 512, (y + 1) * 512)))
            col, row = TILES[destination]
            reconstructed.paste(crop, (col * 512, row * 512))
        np.testing.assert_array_equal(np.array(reconstructed), np.array(transformed))

    def test_real_datasets_match_whole_frame_and_original_mask(self):
        for dataset_cls in (PresenceDataset, MassDataset):
            dataset = dataset_cls(ROOT, 'val', augment=True)
            record = dataset.records[0]
            with patch('numpy.random.rand', return_value=0.0):
                sample = dataset[0]
            with Image.open(ROOT / record['image']) as image:
                expected = np.array(ImageOps.mirror(image.convert('RGB')))
            assembled = torch.cat([torch.cat(list(sample[0][row * 4:(row + 1) * 4]), dim=2)
                                   for row in range(3)], dim=1)
            np.testing.assert_array_equal((assembled.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), expected)
            with Image.open(ROOT / record['mask']) as mask:
                truth = np.array(ImageOps.mirror(mask)) == 40
            counts = [int(truth[y * 512:(y + 1) * 512, x * 512:(x + 1) * 512].sum())
                      for x, y in TILES.values()]
            self.assertEqual(sample[1].bool().tolist(), [count > 0 for count in counts])
            if dataset_cls is MassDataset:
                self.assertEqual(sample[3].tolist(), counts)
                self.assertEqual(sample[2].dtype, torch.float32)

    def test_invalid_frame_and_teacher_augmentation_rejected(self):
        with self.assertRaises(ValueError):
            transform_frame(Image.new('RGB', (512, 512)))
        with self.assertRaises(ValueError):
            UtilityDataset(ROOT, {}, 'train', augment=True)


if __name__ == '__main__':
    unittest.main()
