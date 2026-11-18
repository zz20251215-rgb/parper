import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def configure(seed=42):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'}:
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before importing torch')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng(sampler):
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'algorithm': state[0], 'keys': state[1].tolist(), 'position': state[2],
                      'has_gauss': state[3], 'cached_gaussian': state[4]},
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
            'sampler': sampler.generator.get_state() if hasattr(sampler, 'generator') else None}


def restore_rng(state, sampler):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['algorithm'], np.array(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    if state['sampler'] is not None:
        sampler.generator.set_state(state['sampler'])


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def record_run(project, output, source, split_root):
    snapshot = output / 'reproduction'
    snapshot.mkdir()
    source_hashes = {}
    for folder in ['src', 'scripts']:
        for path in sorted((project / folder).rglob('*')):
            if path.suffix not in {'.py', '.html', '.js', '.css'} or not path.is_file():
                continue
            relative = path.relative_to(project)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ['requirements.lock.txt', 'README.md']:
        shutil.copy2(project / name, snapshot / name)
    shutil.copytree(split_root, snapshot / 'dataset_manifest')
    manifest = json.loads((split_root / 'manifest.json').read_text())
    data_hashes = {}
    digest = hashlib.sha256()
    for key in sorted(manifest['samples']):
        group, tile = key.rsplit('_', 1)
        paths = [source / '3_img' / f'{key}.png',
                 source / '3_mask' / f'{group}_fuse_{tile}_1band.png']
        for path in paths:
            relative = str(path.relative_to(project))
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            data_hashes[relative] = value
            digest.update(relative.encode() + b'\0' + bytes.fromhex(value))
    write_json(snapshot / 'data_sha256.json', data_hashes)
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.strip()
    record = {'status': 'deterministic_enabled', 'created_at': datetime.now(timezone.utc).isoformat(),
              'command': sys.argv, 'python_executable': sys.executable,
              'seed': 42, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                              'cudnn': torch.backends.cudnn.version(), 'driver': driver,
                              'gpu': torch.cuda.get_device_name(), 'packages': versions,
                              'cpu_threads': torch.get_num_threads(),
                              'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                              'cudnn_benchmark': torch.backends.cudnn.benchmark,
                              'cudnn_deterministic': torch.backends.cudnn.deterministic,
                              'tf32': False, 'dataloader_workers': 0},
              'data': {'fingerprint': digest.hexdigest(), 'files': len(data_hashes),
                       'splits_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted((split_root / 'splits').glob('*.txt'))}},
              'source_sha256': source_hashes,
              'checkpoint': 'last.pt contains model, optimizer, scheduler, early stopping, history and RNG states',
              'limitations': '确定性验证限于相同软硬件环境；不同设备或 PyTorch/CUDA 版本不保证逐位一致。'}
    write_json(output / 'reproducibility.json', record)
    return record
