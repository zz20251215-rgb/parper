import hashlib
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COMMIT = 'e96bc852189c2089dd500527a0a01a5a36e8977e'
URL = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
SHA256 = '8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'
BINARY_SHA256 = 'a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da'


def main():
    tools = PROJECT / '.tools'
    tools.mkdir(exist_ok=True)
    repository = tools / 'PlotNeuralNet'
    if not repository.exists():
        subprocess.run(['git', 'init', str(repository)], check=True)
        subprocess.run(['git', '-C', str(repository), 'remote', 'add', 'origin',
                        'https://github.com/HarisIqbal88/PlotNeuralNet.git'], check=True)
        subprocess.run(['git', '-C', str(repository), 'fetch', '--depth', '1', 'origin', COMMIT], check=True)
        subprocess.run(['git', '-C', str(repository), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    commit = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True, check=True).stdout.strip()
    if commit != COMMIT:
        raise RuntimeError('Existing PlotNeuralNet checkout differs from the pinned version')
    binary = tools / 'bin/tectonic'
    if not binary.exists():
        temporary = PROJECT / 'tmp/architecture_setup'
        temporary.mkdir(parents=True, exist_ok=False)
        archive_path = temporary / 'tectonic.tar.gz'
        urllib.request.urlretrieve(URL, archive_path)
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Tectonic archive checksum mismatch')
        with tarfile.open(archive_path) as archive:
            members = [m for m in archive.getmembers() if Path(m.name).name == 'tectonic' and m.isfile()]
            if len(members) != 1:
                raise RuntimeError('Unexpected Tectonic archive layout')
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(archive.extractfile(members[0]).read())
            binary.chmod(0o755)
        archive_path.unlink()
        temporary.rmdir()
    if hashlib.sha256(binary.read_bytes()).hexdigest() != BINARY_SHA256:
        raise RuntimeError('Existing Tectonic binary checksum mismatch')
    info = {'PlotNeuralNet': {'url': 'https://github.com/HarisIqbal88/PlotNeuralNet', 'commit': commit},
            'tectonic': {'version': '0.17.0', 'url': URL, 'archive_sha256': SHA256,
                         'binary_sha256': BINARY_SHA256}}
    (tools / 'architecture_tools.json').write_text(json.dumps(info, indent=2) + '\n')
    subprocess.run([str(binary), '--version'], check=True)
    print('PlotNeuralNet and portable LaTeX configured in .tools/')


if __name__ == '__main__':
    main()
