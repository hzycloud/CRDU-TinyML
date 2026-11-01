"""Render training and feasibility curves from archived numerical records."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round2/figures_v3'
COLORS = ['#235789', '#AA4A28', '#18754C', '#775799', '#383838']
MARKERS = ['o', 's', '^', 'D', 'v']
BUDGETS = [1.0, .95, .9, .85, .8]


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(OUT / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11.5, 'axes.titlesize': 12,
        'axes.labelsize': 11.5, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(2254 / 300, 1534 / 300))
    for row, task in enumerate(('cifar', 'speech')):
        folder = ROOT / ('results/review_revision/cifar_static' if task == 'cifar'
                         else 'results/round2/speech')
        paths = sorted(folder.glob('seed*/history.json'))
        if len(paths) != 5:
            raise ValueError(f'{task}: expected five archived training seeds')
        for i, path in enumerate(paths):
            history = json.loads(path.read_text())
            summary = json.loads((path.parent / 'training_summary.json').read_text())
            epochs = [h['epoch'] for h in history]
            train = [h['train_loss'] for h in history]
            validation = [h.get('validation_nll', h.get('validation_metrics', {}).get('nll')) for h in history]
            accuracy = [h.get('validation_accuracy', h.get('validation_metrics', {}).get('accuracy')) for h in history]
            axes[row, 0].plot(epochs, validation, color=COLORS[i], marker=MARKERS[i],
                              markevery=6, ms=3.5, lw=1.3, label=f"Seed {summary['seed']}")
            axes[row, 0].plot(epochs, train, color=COLORS[i], ls='--', alpha=.55, lw=1)
            selected = summary['best_epoch']
            axes[row, 0].plot(selected, validation[selected - 1], marker='*', color=COLORS[i],
                              markeredgecolor='black', markeredgewidth=.5, ms=9, zorder=5)
            axes[row, 1].plot(epochs, accuracy, color=COLORS[i], marker=MARKERS[i],
                              markevery=6, ms=3.5, lw=1.3)
        label = 'CIFAR-10' if row == 0 else 'Speech'
        axes[row, 0].set(title=label + ' loss', xlabel='Epoch', ylabel='Loss')
        axes[row, 0].set_ylim(bottom=0)
        axes[row, 1].set(title=label + ' validation accuracy', xlabel='Epoch', ylabel='Accuracy')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(.5, .995), handlelength=1.4, columnspacing=1.1)
    fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.85, wspace=.28, hspace=.55)
    save(fig, 'fig2_training')

    fig, axes = plt.subplots(2, 2, figsize=(2254 / 300, 1294 / 300))
    styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))]
    for row, task in enumerate(('cifar', 'speech')):
        path = ROOT / f'results/round2/analysis_{task}/coverage.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            coverage = list(csv.DictReader(stream))
        main_rows = [r for r in coverage if abs(float(r['tolerance']) - .01) < 1e-9]
        seeds = sorted({int(r['seed']) for r in main_rows})
        values = np.array([[int(next(r['feasible_count'] for r in main_rows
                            if int(r['seed']) == seed and float(r['budget']) == budget))
                           for budget in BUDGETS] for seed in seeds])
        if values.shape != (5, 5):
            raise ValueError(f'{task}: incomplete feasibility grid')
        axes[row, 0].imshow(values, cmap='Blues', aspect='auto', vmin=0)
        for i in range(5):
            for j in range(5):
                axes[row, 0].text(j, i, str(values[i, j]), ha='center', va='center',
                                  color='white' if values[i, j] >= .55 * values.max() else 'black')
        axes[row, 0].grid(False)
        label = 'CIFAR-10' if row == 0 else 'Speech'
        axes[row, 0].set(xticks=range(5), xticklabels=[f'{b:.2f}' for b in BUDGETS],
                         yticks=range(5), yticklabels=seeds, xlabel='Weight payload budget',
                         ylabel='Training seed', title=label + ' feasible candidates')
        tolerances = sorted({float(r['tolerance']) for r in coverage})
        for i, budget in enumerate(BUDGETS):
            means = [np.mean([int(r['feasible_count']) for r in coverage
                              if float(r['budget']) == budget and float(r['tolerance']) == tol])
                     for tol in tolerances]
            axes[row, 1].plot(np.array(tolerances) * 100, means, color=COLORS[i],
                              marker=MARKERS[i], ls=styles[i], ms=4, lw=1.4, label=f'B={budget:.2f}')
        axes[row, 1].set(xlabel='Accuracy tolerance (pp)', ylabel='Mean candidates',
                         title=label + ' constraint sensitivity')
        axes[row, 1].set_ylim(bottom=0)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(.5, .995), handlelength=1.8, columnspacing=1.1)
    fig.subplots_adjust(left=.09, right=.99, bottom=.13, top=.83, wspace=.31, hspace=.72)
    save(fig, 'fig3_feasibility')


if __name__ == '__main__':
    main()
