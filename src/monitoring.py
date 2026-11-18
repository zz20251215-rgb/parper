import csv
import io
import json
import math
import re
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path


class TrainingMonitor:
    def __init__(self, run, interval=5):
        self.run = Path(run).resolve()
        self.interval = interval
        self.history = deque(maxlen=720)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cpu_previous = None
        self.json_cache = {}
        self.log_path = self.run / 'resources.jsonl'
        self.load_errors = []
        if self.log_path.exists():
            with self.log_path.open() as stream:
                for line in stream:
                    try:
                        self.history.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.load_errors = ['resources.jsonl contains an incomplete or invalid sample']

    def read_json(self, name, default, errors):
        path = self.run / name
        if not path.exists():
            return default
        try:
            result = json.loads(path.read_text())
            self.json_cache[name] = result
            return result
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f'{name}: {type(exc).__name__}; using last valid value')
            return self.json_cache.get(name, default)

    def process_info(self, errors):
        metadata = self.read_json('process.json', {}, errors)
        pid = metadata.get('pid')
        result = {'pid': pid, 'state': None, 'rss_mb': None}
        if not isinstance(pid, int) or pid < 1:
            return result
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if not any(part.endswith(b'/train.py') or part == b'train.py' for part in cmd):
                errors.append('Recorded PID no longer belongs to the training entry point')
                return result
            output = None
            for index, argument in enumerate(cmd):
                if argument == b'--output' and index + 1 < len(cmd):
                    output = cmd[index + 1]
                elif argument.startswith(b'--output='):
                    output = argument.split(b'=', 1)[1]
            if not output:
                errors.append('Recorded PID has no verifiable --output training directory')
                return result
            output_path = Path(output.decode(errors='surrogateescape'))
            if not output_path.is_absolute():
                output_path = Path(f'/proc/{pid}/cwd').resolve(strict=True) / output_path
            if output_path.resolve() != self.run:
                errors.append('Recorded PID belongs to a different training run')
                return result
            values = {}
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    values[key] = value.strip()
            result['state'] = values.get('State', '').split()[0] or None
            if 'VmRSS' in values:
                result['rss_mb'] = int(values['VmRSS'].split()[0]) / 1024
        except (OSError, ValueError) as exc:
            errors.append(f'Training process unavailable: {type(exc).__name__}')
        return result

    def gpu_info(self, errors):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
            rows = list(csv.reader(io.StringIO(result.stdout)))
            if not rows or len(rows[0]) != 6:
                raise ValueError('Unexpected GPU telemetry response')
            name, used, total, util, temperature, power = [v.strip() for v in rows[0]]
            values = {}
            for key, value in [('used_mb', used), ('total_mb', total), ('utilization_pct', util),
                               ('temperature_c', temperature), ('power_w', power)]:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = None
                    errors.append(f'GPU {key} sensor unavailable')
            return {'name': name, **values}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f'GPU telemetry unavailable: {type(exc).__name__}')
            return None

    def ram_info(self, errors):
        try:
            values = {line.split(':')[0]: int(line.split()[1])
                      for line in Path('/proc/meminfo').read_text().splitlines()}
            total = values['MemTotal'] / 1024
            used = (values['MemTotal'] - values['MemAvailable']) / 1024
            return {'used_mb': used, 'total_mb': total, 'percent': used / total * 100}
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f'System RAM telemetry unavailable: {type(exc).__name__}')
            return {'used_mb': None, 'total_mb': None, 'percent': None}

    def cpu_info(self, errors):
        utilization = None
        temperature = None
        try:
            fields = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
            # guest times are already included in user/nice counters.
            total, idle = sum(fields[:8]), fields[3] + fields[4]
            if self.cpu_previous:
                previous_total, previous_idle = self.cpu_previous
                elapsed = total - previous_total
                if elapsed > 0:
                    utilization = 100 * (1 - (idle - previous_idle) / elapsed)
            self.cpu_previous = (total, idle)
        except (OSError, ValueError, IndexError) as exc:
            errors.append(f'CPU utilization unavailable: {type(exc).__name__}')
        candidates = []
        for directory in Path('/sys/class/hwmon').glob('hwmon*'):
            try:
                name = (directory / 'name').read_text().strip()
                if name not in {'k10temp', 'coretemp'}:
                    continue
                for path in directory.glob('temp*_input'):
                    label_path = path.with_name(path.name.replace('_input', '_label'))
                    label = label_path.read_text().strip() if label_path.exists() else ''
                    value = float(path.read_text()) / 1000
                    priority = 0 if label == 'Tctl' or label.startswith('Package') else 1
                    candidates.append((priority, value))
            except (OSError, ValueError) as exc:
                errors.append(f'CPU temperature sensor unavailable: {type(exc).__name__}')
        if candidates:
            priority = min(p for p, _ in candidates)
            temperature = max(v for p, v in candidates if p == priority)
        else:
            errors.append('CPU temperature unavailable: no readable k10temp/coretemp sensor')
        return {'utilization_pct': utilization, 'temperature_c': temperature}

    def sample(self):
        errors = list(self.load_errors)
        process = self.process_info(errors)
        sample = {'timestamp': datetime.now().astimezone().isoformat(),
                  'gpu': self.gpu_info(errors), 'ram': self.ram_info(errors),
                  'cpu': self.cpu_info(errors), 'process': {'rss_mb': process['rss_mb']},
                  'errors': errors}
        with self.lock:
            try:
                with self.log_path.open('a') as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
            except OSError as exc:
                errors.append(f'Resource history could not be saved: {type(exc).__name__}')
            self.history.append(sample)
        return sample

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def _collect(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def close(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=4)

    def state(self):
        errors = []
        config = self.read_json('config.json', {}, errors)
        history = self.read_json('history.json', [], errors)
        metrics = self.read_json('metrics.json', None, errors)
        process = self.process_info(errors)
        lines = []
        try:
            with (self.run / 'train.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f'Training log unavailable: {type(exc).__name__}')
        latest_epoch = history[-1].get('epoch', 0) if history else 0
        latest_step = history[-1].get('train_steps', 0) if history else 0
        for line in reversed(lines):
            match = re.search(r'epoch=(\d+) step=(\d+)', line)
            if match:
                latest_epoch, latest_step = map(int, match.groups())
                break
        if metrics is not None:
            status = 'completed'
        elif process['state'] in {'T', 't'}:
            status = 'paused'
        elif process['state'] in {'R', 'S', 'D', 'I'}:
            status = 'running'
        elif any('Traceback (most recent call last)' in line for line in lines):
            status = 'failed'
        elif process['pid']:
            status = 'stopped'
        else:
            status = 'unknown'
        repro = self.read_json('reproducibility.json', None, errors)
        if repro is None:
            repro = self.read_json('experiment.json', None, errors)
        if repro is None:
            repro = self.read_json('reproducibility_note.json', None, errors)
        total_images = config.get('split_summary', {}).get('train', {}).get('images', 0)
        batch = config.get('batch_size', 1)
        with self.lock:
            samples = list(self.history)
        resources = samples[-1] if samples else None
        return {'run': self.run.name, 'status': status,
                'process': {'pid': process['pid'], 'state': process['state']},
                'progress': {'epoch': latest_epoch, 'step': latest_step,
                             'total_steps': math.ceil(total_images / max(batch, 1)),
                             'max_epochs': config.get('epochs')},
                'config': config, 'history': history, 'metrics': metrics,
                'resources': resources, 'resource_history': samples,
                'latest_log': lines[-20:], 'reproducibility': repro,
                'errors': errors}
