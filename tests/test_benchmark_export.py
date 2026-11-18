import unittest

import torch

from scripts.benchmark_export import measure


class ExportBenchmarkTests(unittest.TestCase):
    def test_measure_returns_requested_sample_count(self):
        model = torch.nn.Identity()
        sample = torch.zeros(1, 3, 4, 4)
        result = measure(model, sample, warmup=1, repeats=3, device=torch.device('cpu'))
        self.assertEqual(result['samples'], 3)
        self.assertGreater(result['mean_ms'], 0.0)
        self.assertLessEqual(result['p50_ms'], result['max_ms'])
        self.assertLessEqual(result['p95_ms'], result['max_ms'])


if __name__ == '__main__':
    unittest.main()
