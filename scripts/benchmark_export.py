"""Benchmark eager and TorchScript inference for a single 512x512 tile."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mobilenet_unet import MobileNetV3LargeRefineUNet


def measure(model, sample, warmup, repeats, device):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            values.append(time.perf_counter() - start)
    return {
        'mean_ms': 1000 * sum(values) / len(values),
        'p50_ms': 1000 * float(torch.tensor(values).quantile(0.5)),
        'p95_ms': 1000 * float(torch.tensor(values).quantile(0.95)),
        'min_ms': 1000 * min(values),
        'max_ms': 1000 * max(values),
        'samples': repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--torchscript', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if min(args.warmup, args.repeats) < 1:
        parser.error('warmup and repeats must be positive')
    project = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    eager_cpu = MobileNetV3LargeRefineUNet(False).eval()
    eager_cpu.load_state_dict(checkpoint['model'])
    traced_cpu = torch.jit.load(str(args.torchscript), map_location='cpu').eval()
    sample_cpu = torch.linspace(0.0, 1.0, 3 * 512 * 512, dtype=torch.float32).reshape(1, 3, 512, 512)
    with torch.inference_mode():
        eager_output = eager_cpu(sample_cpu)
        traced_output = traced_cpu(sample_cpu)
    torch.testing.assert_close(eager_output, traced_output, rtol=1e-5, atol=1e-6)
    report = {
        'protocol': 'torchscript-eager-benchmark-v1',
        'checkpoint': str(args.checkpoint),
        'torchscript': str(args.torchscript),
        'input_shape': list(sample_cpu.shape),
        'torch': torch.__version__,
        'python': platform.python_version(),
        'cpu': platform.processor(),
        'output_max_abs_diff': float((eager_output - traced_output).abs().max()),
        'output_mean_abs_diff': float((eager_output - traced_output).abs().mean()),
        'devices': {},
        'limitations': ['Host benchmark only; target device not selected.',
                        'Single-tile batch-1 timing; does not represent full cascade throughput.'],
    }
    report['devices']['cpu'] = {
        'eager': measure(eager_cpu, sample_cpu, args.warmup, args.repeats, torch.device('cpu')),
        'torchscript': measure(traced_cpu, sample_cpu, args.warmup, args.repeats, torch.device('cpu')),
    }
    if torch.cuda.is_available():
        device = torch.device('cuda')
        eager_gpu = MobileNetV3LargeRefineUNet(False).to(device).eval()
        eager_gpu.load_state_dict(checkpoint['model'])
        traced_gpu = torch.jit.load(str(args.torchscript), map_location=device).to(device).eval()
        sample_gpu = sample_cpu.to(device)
        with torch.inference_mode():
            torch.testing.assert_close(eager_gpu(sample_gpu).cpu(), traced_gpu(sample_gpu).cpu(), rtol=1e-5, atol=1e-6)
        torch.cuda.reset_peak_memory_stats(device)
        report['devices']['cuda'] = {
            'name': torch.cuda.get_device_name(device),
            'eager': measure(eager_gpu, sample_gpu, args.warmup, args.repeats, device),
            'torchscript': measure(traced_gpu, sample_gpu, args.warmup, args.repeats, device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
