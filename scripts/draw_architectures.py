from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.nice(10)
    import torch
    from src.unet import UNet
    from src.resnet_unet import ResNet34UNet
    from src.architecture.unet_diagram import make_tex as unet_tex
    from src.architecture.resnet_diagram import make_tex as resnet_tex

    torch.set_num_threads(2)
    tool = ROOT / '.tools/PlotNeuralNet'
    compiler = ROOT / '.tools/bin/tectonic'
    if not compiler.exists() or not tool.exists():
        raise RuntimeError('Run scripts/setup_architecture_tools.py first')
    output = ROOT / 'outputs/reports/model_architecture'
    build = ROOT / 'tmp/pdfs/model_architecture'
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    audit = {}
    specs = [
        ('unet', UNet(), unet_tex, 'down', 'decode', 7760097,
         [[32,512,512],[64,256,256],[128,128,128],[256,64,64],[512,32,32]],
         [[256,64,64],[128,128,128],[64,256,256],[32,512,512]], [512,256,128,64]),
        ('resnet34_unet', ResNet34UNet(pretrained=False), resnet_tex, 'encoder', 'decoder', 24436241,
         [[64,256,256],[64,128,128],[128,64,64],[256,32,32],[512,16,16]],
         [[256,32,32],[128,64,64],[64,128,128],[32,256,256],[16,512,512]], [768,384,192,128,32]),
    ]
    for name, model, generate, enc, dec, count, expected_enc, expected_dec, expected_inputs in specs:
        observed = {'encoder': [], 'decoder': [], 'decoder_input_channels': []}
        hooks = []
        for layer in getattr(model, enc):
            hooks.append(layer.register_forward_hook(lambda m, a, b: observed['encoder'].append(list(b.shape[1:]))))
        for layer in getattr(model, dec):
            def record(m, args, result):
                observed['decoder'].append(list(result.shape[1:]))
                observed['decoder_input_channels'].append(args[0].shape[1])
            hooks.append(layer.register_forward_hook(record))
        with torch.inference_mode():
            result = model.eval()(torch.zeros(1, 3, 512, 512))
        for hook in hooks:
            hook.remove()
        observed['parameters'] = sum(p.numel() for p in model.parameters())
        observed['output'] = list(result.shape[1:])
        assert observed['parameters'] == count, observed
        assert observed['encoder'] == expected_enc, observed
        assert observed['decoder'] == expected_dec, observed
        assert observed['decoder_input_channels'] == expected_inputs, observed
        assert observed['output'] == [1,512,512], observed
        if name == 'resnet34_unet':
            observed['encoder_parameters'] = sum(p.numel() for p in model.encoder.parameters())
            assert observed['encoder_parameters'] == 21284672
        audit[name] = observed
        source = output / f'{name}.tex'
        source.write_text(generate(tool))
        subprocess.run([str(compiler), '--keep-logs', '--outdir', str(build), str(source)], check=True)
        pdf = output / f'{name}.pdf'
        shutil.copy2(build / pdf.name, pdf)
        subprocess.run(['pdftocairo', '-svg', str(pdf), str(output / f'{name}.svg')], check=True)
        subprocess.run(['pdftoppm', '-singlefile', '-png', '-r', '160', str(pdf), str(output / name)], check=True)
        print(f'Created {pdf}', flush=True)
    (output / 'shape_audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
