"""Render unsmoothed charts from exported W&B metric CSVs; no credentials needed."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 160})


def read(run):
    with (ROOT / 'data' / f'{run}.csv').open() as f:
        return list(csv.DictReader(f))


def series(ax, rows, key, label, percent=False, **kwargs):
    points = [(float(r['step']), float(r[key])) for r in rows if r.get(key)]
    ax.plot([x for x, _ in points], [y * (100 if percent else 1) for _, y in points],
            label=label, **kwargs)
    ax.set_xlabel('Logged rollout step'); ax.grid(alpha=.2); ax.legend(fontsize=8)


rows = read('p2wswixp')
fig, axs = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
fig.suptitle('RFT sanity run • OLMo-2-0425-1B • seed 0 • 2× H100', fontsize=15)
for ax, kind in zip(axs.flat, ['reward', 'format_reward', 'loss', 'grad_norm', 'token_entropy', 'response_length']):
    percent = kind in ('reward', 'format_reward')
    series(ax, rows, f'train/{kind}', 'Training batch', percent, color='#2563eb')
    if kind in ('reward', 'format_reward', 'response_length'):
        series(ax, rows, f'val/{kind}', 'Validation (1,024 questions)', percent,
               marker='o', color='#d97706')
    titles = {'reward': 'Correct answers (%)', 'format_reward': 'Accepted format (%)',
              'loss': 'RFT loss (constant normalization)', 'grad_norm': 'Gradient norm',
              'token_entropy': 'Entropy (accepted training responses only)',
              'response_length': 'Response length (tokens)'}
    ax.set_title(titles[kind])
    if percent: ax.set_ylim(0, 100)
fig.savefig(ROOT / 'rft-sanity.png'); plt.close(fig)

rows = read('w9vc0c3v')
fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
fig.suptitle('Earlier nanochat-style RL • OLMo base • 1× H100 • stopped', fontsize=14)
series(axs[0], rows, 'reward', 'Training correctness', True, color='#2563eb')
series(axs[0], rows, 'answer_format_rate', 'Training format', True, color='#d97706')
axs[0].set_title('Training rewards (%)'); axs[0].set_ylim(0, 100)
axs[0].axvspan(66, 130, color='#dc2626', alpha=.08, label='Final 65 zero-reward batches')
axs[0].legend(fontsize=8)
series(axs[1], rows, 'pass@1', 'Validation pass@1', True, marker='o', color='#2563eb')
series(axs[1], rows, 'pass@8', 'Validation pass@8', True, marker='o', color='#d97706')
axs[1].set_title('Validation: 400 questions (%)'); axs[1].set_ylim(bottom=0)
fig.savefig(ROOT / 'earlier-rl.png'); plt.close(fig)

bench = [read('fgkoyxj4'), read('hnlrh0lb')]
total = [float(r[-1]['timing/total_workload_seconds']) for r in bench]
generation = [sum(float(x['timing/generation_and_grading_seconds']) for x in r[:-1]) / 3 for r in bench]
cost = [total[0] * 2 * .001097, total[1] * 2 * .001736]
fig, axs = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
fig.suptitle('Three-step RFT benchmarks • same image • two GPUs per run', fontsize=14)
for ax, values, title in zip(axs, [total, generation, cost],
                             ['Total workload (seconds)', 'Mean generation + grading (seconds)', 'Estimated GPU-only workload cost ($)']):
    bars = ax.bar(['H100', 'B200'], values, color=['#2563eb', '#d97706'])
    ax.bar_label(bars, fmt='%.3f', padding=4); ax.set_ylim(0, max(values)*1.2)
    ax.set_title(title, fontsize=10); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.savefig(ROOT / 'gpu-benchmarks.png'); plt.close(fig)
