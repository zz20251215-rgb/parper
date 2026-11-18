import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.benchmark_roi_cascade import predict_frame
from src.tile_geometry import TILES


class RecordingSegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, batch):
        self.batch_sizes.append(len(batch))
        return (batch[:, :1] - 0.5) * 20


class FixedRanker(torch.nn.Module):
    def forward(self, batch):
        return torch.arange(12, device=batch.device).float().unsqueeze(0)


class RoiBenchmarkTests(unittest.TestCase):
    def test_inference_batching_and_background_without_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (2048, 1536), 'white').save(path)
            segmenter = RecordingSegmenter()
            full, selected = predict_frame(path, None, segmenter, 6, 6, torch.device('cpu'))
            self.assertEqual(segmenter.batch_sizes, [6, 6])
            self.assertTrue(full.all())
            self.assertEqual(len(selected), 12)
            segmenter.batch_sizes.clear()
            ranked, selected = predict_frame(path, FixedRanker(), segmenter, 3, 2, torch.device('cpu'))
            self.assertEqual(segmenter.batch_sizes, [2, 1])
            self.assertEqual(selected, ['N', 'M', 'L'])
            expected = np.zeros_like(ranked)
            for tile in selected:
                col, row = TILES[tile]
                expected[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = True
            np.testing.assert_array_equal(ranked, expected)


if __name__ == '__main__':
    unittest.main()

class TorchScriptSegmenterTests(unittest.TestCase):
    def test_torchscript_segmenter_matches_eager_dummy(self):
        eager = RecordingSegmenter().eval()
        scripted = torch.jit.trace(eager, torch.zeros(2, 3, 512, 512)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (2048, 1536), 'white').save(path)
            eager_pred, eager_selected = predict_frame(path, None, eager, 6, 6, torch.device('cpu'))
            scripted_pred, scripted_selected = predict_frame(path, None, scripted, 6, 6, torch.device('cpu'))
        np.testing.assert_array_equal(eager_pred, scripted_pred)
        self.assertEqual(eager_selected, scripted_selected)

    def test_benchmark_parser_exposes_torchscript_format(self):
        import subprocess
        result = subprocess.run(
            ['.venv/bin/python', 'scripts/benchmark_roi_cascade.py', '--help'],
            check=True, capture_output=True, text=True)
        self.assertIn('--segmenter-format', result.stdout)
        self.assertIn('torchscript', result.stdout)

    def test_benchmark_parser_exposes_segmenter_precision(self):
        import subprocess
        result = subprocess.run(
            ['.venv/bin/python', 'scripts/benchmark_roi_cascade.py', '--help'],
            check=True, capture_output=True, text=True)
        self.assertIn('--segmenter-precision', result.stdout)
        self.assertIn('fp16', result.stdout)
