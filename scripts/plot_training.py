import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.plotting import save_training_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = (PROJECT / args.run).resolve()
    if not run.is_relative_to(PROJECT / 'outputs/runs') or not run.is_dir():
        parser.error('--run must be an existing directory inside outputs/runs')
    for path in save_training_curves(run).values():
        print(path)


if __name__ == '__main__':
    main()
