import tempfile
import unittest
from pathlib import Path

import torch

from scripts.export_torchscript import main


class TorchScriptExportTests(unittest.TestCase):
    def test_export_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model.ts'
            output.write_bytes(b'existing')
            checkpoint = Path('outputs/runs/tack_mobilenetv3_large_refine_v1/best.pt')
            old_argv = __import__('sys').argv
            try:
                __import__('sys').argv = ['export_torchscript', '--checkpoint', str(checkpoint),
                                          '--arch', 'mobilenetv3_large_refine', '--output', str(output)]
                with self.assertRaises(FileExistsError):
                    main()
            finally:
                __import__('sys').argv = old_argv

    def test_exported_model_has_expected_shape(self):
        model = torch.jit.load('outputs/runs/tack_mobilenetv3_large_refine_v1/deploy/refine_fp32.ts')
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 512, 512))
        self.assertEqual(tuple(output.shape), (1, 1, 512, 512))
        self.assertEqual(output.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
