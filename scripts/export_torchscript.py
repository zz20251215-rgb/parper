"""Export a trained segmentation model to TorchScript and verify eager parity."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mobilenet_unet import MobileNetV3LargeRefineUNet, MobileNetV3LargeUNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--arch', choices=['mobilenetv3_large', 'mobilenetv3_large_refine'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1:
        parser.error('height and width must be positive')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.precision == 'fp16' and not torch.cuda.is_available():
        raise RuntimeError('FP16 export requires CUDA; CPU FP16 convolution is not a valid verification path')

    model_cls = MobileNetV3LargeRefineUNet if args.arch == 'mobilenetv3_large_refine' else MobileNetV3LargeUNet
    model = model_cls(pretrained=False).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    export_device = torch.device('cuda' if args.precision == 'fp16' else 'cpu')
    if args.precision == 'fp16':
        # Trace in FP32 on CPU, then convert the complete traced module. Direct
        # CUDA FP16 tracing is not stable for this decoder: repeated calls can
        # diverge even though eager FP16 is deterministic. The converted
        # module is still checked against eager FP16 on CUDA before saving.
        sample32 = torch.rand(
            (1, 3, args.height, args.width),
            generator=torch.Generator().manual_seed(42),
            dtype=torch.float32,
        )
        # Remove in-place activations before tracing. They are safe during
        # eager inference but can make a converted half-precision graph
        # depend on storage reuse across calls.
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        with torch.inference_mode():
            traced = torch.jit.trace(model, sample32, strict=True, check_trace=True).half()
            eager_model = model.to(export_device).half()
            sample = sample32.to(export_device, dtype=torch.float16)
            eager = eager_model(sample)
            scripted = traced.to(export_device)(sample)
    else:
        model = model.to(export_device)
        sample = torch.linspace(
            0.0, 1.0, 3 * args.height * args.width, dtype=torch.float32, device=export_device
        ).reshape(1, 3, args.height, args.width)
        with torch.inference_mode():
            eager = model(sample)
            traced = torch.jit.trace(model, sample, strict=True, check_trace=True)
            scripted = traced(sample)
    torch.testing.assert_close(scripted, eager, rtol=1e-5, atol=1e-6)
    # A single trace input is insufficient for this decoder: CUDA execution
    # can diverge on later inputs even when the trace sample matches. Check
    # deterministic additional inputs before accepting the artifact.
    if args.precision == 'fp16':
        with torch.inference_mode():
            for seed in (7, 8):
                check = torch.rand(
                    (1, 3, args.height, args.width),
                    generator=torch.Generator(device='cuda').manual_seed(seed),
                    dtype=torch.float16,
                    device='cuda',
                )
                torch.testing.assert_close(
                    traced.to(export_device)(check), eager_model(check), rtol=1e-5, atol=1e-6
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Verify the serialized artifact itself. In particular, a CUDA half graph
    # can match before save and diverge after reload, so the in-memory check is
    # not sufficient for accepting an export.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f'.{args.output.stem}.', suffix='.tmp.ts', dir=args.output.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        traced.save(str(temp_path))
        loaded = torch.jit.load(str(temp_path), map_location=export_device).eval()
        if export_device.type == 'cuda':
            loaded = loaded.cuda()
        if args.precision == 'fp16':
            with torch.inference_mode():
                for seed in (42, 7, 8):
                    check = torch.rand(
                        (1, 3, args.height, args.width),
                        generator=torch.Generator(device='cuda').manual_seed(seed),
                        dtype=torch.float16,
                        device='cuda',
                    )
                    torch.testing.assert_close(
                        loaded(check), eager_model(check), rtol=1e-5, atol=1e-6
                    )
        else:
            with torch.inference_mode():
                torch.testing.assert_close(loaded(sample), eager, rtol=1e-5, atol=1e-6)
        temp_path.replace(args.output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    metadata = {
        'protocol': f'torchscript-{args.precision}-export-v1',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256(args.checkpoint),
        'arch': args.arch,
        'input_shape': list(sample.shape),
        'output_shape': list(eager.shape),
        'dtype': str(sample.dtype),
        'device': str(export_device),
        'torch': torch.__version__,
        'python': platform.python_version(),
        'eager_vs_torchscript_max_abs': float((eager - scripted).abs().max()),
        'eager_vs_torchscript_mean_abs': float((eager - scripted).abs().mean()),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'verification': 'torch.testing.assert_close passed on a deterministic synthetic 512x512 input',
        'limitations': ['No ONNX export: onnx package is unavailable in the current environment.',
                        'No INT8 quantization or target-device latency claim.'],
    }
    metadata_path = args.output.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
