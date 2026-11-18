import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FIELDS = ['epoch', 'train_loss', 'val_loss', 'dice', 'iou', 'precision', 'recall',
          'train_bce_loss', 'train_dice_loss', 'val_bce_loss', 'val_dice_loss',
          'learning_rate', 'next_learning_rate', 'encoder_learning_rate', 'next_encoder_learning_rate', 'encoder_frozen', 'negative_image_fp_rate',
          'negative_pixel_fp_rate', 'duration_seconds', 'train_positive', 'train_negative']
COLORS = {'blue': '#0072B2', 'orange': '#D55E00', 'green': '#009E73', 'purple': '#CC79A7'}


def save_training_curves(run_dir):
    run_dir = Path(run_dir)
    history = json.loads((run_dir / 'history.json').read_text())
    if not history:
        raise ValueError('No completed epochs in history.json')
    rows = []
    for entry in history:
        validation = entry.get('val', {})
        row = {key: entry.get(key) for key in FIELDS}
        row['val_loss'] = validation.get('loss')
        row['val_bce_loss'] = validation.get('bce_loss')
        row['val_dice_loss'] = validation.get('dice_loss')
        for key in ['dice', 'iou', 'precision', 'recall',
                    'negative_image_fp_rate', 'negative_pixel_fp_rate']:
            row[key] = validation.get(key)
        rows.append(row)
    csv_path = run_dir / 'history.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    epochs = [row['epoch'] for row in rows]

    def values(key, percent=False):
        return [float('nan') if row[key] is None else row[key] * (100 if percent else 1)
                for row in rows]

    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
             'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
             'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
             'axes.edgecolor': '#64748B', 'axes.linewidth': 0.7,
             'grid.color': '#E2E8F0', 'grid.linewidth': 0.6,
             'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        figure, axes = plt.subplots(3, 2, figsize=(10.5, 9), layout='constrained')
        try:
            panels = [
                ('(a) Training and validation loss', 'Loss',
                 [('train_loss', 'Training', 'blue', '-', 'o'),
                  ('val_loss', 'Validation', 'orange', '--', 's')], False),
                ('(b) Validation overlap', 'Score',
                 [('dice', 'Dice / F1', 'blue', '-', 'o'),
                  ('iou', 'IoU', 'orange', '--', 's')], False),
                ('(c) Validation precision and recall', 'Score',
                 [('precision', 'Precision', 'green', '-', '^'),
                  ('recall', 'Recall', 'purple', '-.', 'D')], False),
                ('(d) Learning rate', 'Learning rate',
                 [('learning_rate', 'Learning rate', 'blue', '-', 's')], False),
                ('(e) Negative-image false-positive rate', 'False-positive images (%)',
                 [('negative_image_fp_rate', 'Image false-positive rate', 'orange', '--', 's')], True),
                ('(f) Negative-pixel false-positive rate', 'False-positive pixels (%)',
                 [('negative_pixel_fp_rate', 'Pixel false-positive rate', 'purple', '-.', 'D')], True),
            ]
            for index, (axis, (title, ylabel, series, percent)) in enumerate(zip(axes.flat, panels)):
                if index == 3 and any(row['encoder_learning_rate'] is not None for row in rows):
                    series = [('learning_rate', 'Decoder', 'blue', '-', 's'),
                              ('encoder_learning_rate', 'Encoder', 'orange', '--', 'o')]
                for key, label, color, linestyle, marker in series:
                    axis.plot(epochs, values(key, percent), label=label, color=COLORS[color],
                              linestyle=linestyle, marker=marker, markersize=3,
                              markerfacecolor='white', markeredgewidth=0.8, linewidth=1.5,
                              markevery=max(1, len(rows) // 20),
                              drawstyle='steps-post' if key in ['learning_rate', 'encoder_learning_rate'] else 'default')
                axis.set(title=title, xlabel='Epoch', ylabel=ylabel)
                if len(epochs) == 1:
                    axis.set_xticks(epochs)
                else:
                    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                if index in [1, 2]:
                    axis.set_ylim(0, 1)
                elif index == 3:
                    axis.set_yscale('log')
                elif index == 4:
                    axis.set_ylim(0, 100)
                else:
                    peak = max((row[key] * (100 if percent else 1)
                                for row in rows for key, *_ in series if row[key] is not None), default=0)
                    axis.set_ylim(0, peak * 1.1 if peak > 0 else 1)
                axis.grid(axis='y', linestyle='--')
                axis.set_axisbelow(True)
                axis.legend(loc='best', frameon=False)
            directory = run_dir / 'figures'
            directory.mkdir(exist_ok=True)
            paths = {'csv': csv_path}
            for extension in ['pdf', 'svg', 'png']:
                path = directory / f'training_curves.{extension}'
                temporary = path.with_suffix(f'.{extension}.tmp')
                figure.savefig(temporary, format=extension, dpi=300, facecolor='white')
                temporary.replace(path)
                paths[extension] = path
            return paths
        finally:
            plt.close(figure)
