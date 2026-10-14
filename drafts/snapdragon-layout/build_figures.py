"""Article figures from audited measurements; no device jobs are submitted.

Run with the accelerator-bench notebook environment:
    python build_figures.py /path/to/accelerator-bench
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('research_repo', type=Path)
args = parser.parse_args()
repo = args.research_repo.resolve()
sys.path.insert(0, str(repo / 'src'))
from accelerator_bench.layout_research import (
    COLORS, VISION_CASES, hello_layout, convolution_evidence,
    preparation_evidence, vision_validation,
)

destination = Path(__file__).resolve().parent / 'figures'
destination.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 11,
    'figure.facecolor': 'white', 'svg.hashsalt': 'snapdragon-layout-draft',
})
toy = hello_layout()
conv = convolution_evidence(repo)
prep = {r['variant']: r for r in preparation_evidence(repo)}
vision = {c['case_id']: c for c in vision_validation(repo)['cases']}


def save(fig, name, narrow):
    suffix = '-mobile' if narrow else ''
    fig.savefig(destination / f'{name}{suffix}.svg', metadata={'Date': None})
    fig.savefig(destination / f'{name}{suffix}.png', dpi=160)
    plt.close(fig)


def pair(narrow):
    return plt.subplots(2, 1, figsize=(3.8, 6.8)) if narrow else plt.subplots(1, 2, figsize=(7.0, 3.6))


def finish_pair(fig, axes, name, narrow, legend=False):
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.16)
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, .07 if legend else 0, 1, 1], h_pad=2.0, w_pad=2.0)
    save(fig, name, narrow)


for narrow in (False, True):
    fig, ax = plt.subplots(figsize=(3.8, 2.5) if narrow else (7, 2.5))
    ax.set_xlim(-.15, 12.05)
    ax.set_ylim(-.45, 3.6)
    ax.axis('off')
    for i, (label, values) in enumerate((('NCHW', toy['nchw_storage']), ('NHWC', toy['nhwc_storage']))):
        y = 2.2 - i * 1.6
        ax.text(0, y + .9, label, weight='bold', fontsize=10)
        for j, value in enumerate(values):
            ax.add_patch(Rectangle((j, y), .94, .7, facecolor=COLORS[int(value >= 10)]))
            ax.text(j + .47, y + .35, str(int(value)), ha='center', va='center', color='white', fontsize=9 if narrow else 11)
    ax.text(0, -.25, 'Channel 0', color=COLORS[0], fontsize=9)
    ax.text(4, -.25, 'Channel 1', color=COLORS[1], fontsize=9)
    ax.text(12, -.25, 'Memory order →', ha='right', fontsize=9)
    fig.tight_layout(pad=.7)
    save(fig, 'storage_order', narrow)

    fig, axes = pair(narrow)
    for j, label in enumerate(('NCHW', 'NHWC')):
        rows = sorted((r for r in conv if r['layout'] == label), key=lambda r: r['size'])
        x = np.arange(4) + (j - .5) * .25
        med = np.array([r['median_us'] for r in rows])
        error = [med - [r['min_us'] for r in rows], np.array([r['max_us'] for r in rows]) - med]
        axes[0].errorbar(x, med, yerr=error, color=COLORS[j], marker='_', markersize=12, capsize=3, linestyle='none', label=label)
        for pos, row in zip(x, rows):
            axes[0].scatter(pos + np.linspace(-.035, .035, row['n']), [job['median_us'] for job in row['jobs']], color=COLORS[j], s=19, zorder=3)
        axes[1].bar(x, [r['transpose_instances'] for r in rows], width=.22, color=COLORS[j])
    for ax in axes:
        ax.set_xticks(range(4), ['16', '32', '64', '128'])
        ax.set_xlabel('H = W')
        ax.set_ylim(bottom=0)
    axes[0].set(title='QNN call', ylabel='Latency (µs)')
    axes[1].set(title='HTP profile', ylabel='Transpose instances', yticks=[0, 4, 18])
    finish_pair(fig, axes, 'convolution_layout', narrow, legend=True)

    fig, axes = pair(narrow)
    labels = ['O3', 'O1', 'O3 / no fold']
    for i, variant in enumerate(('optimized', 'prepare_fast', 'no_conv_relu_fold')):
        row = prep[variant]
        color = [COLORS[0], '#667085', COLORS[1]][i]
        axes[0].barh(i, row['median_us'], color=color)
        axes[0].errorbar(row['median_us'], i, xerr=[[row['median_us']-row['min_us']], [row['max_us']-row['median_us']]], color='black', capsize=3)
        axes[1].barh(i, row['htp_nodes'], color=color)
    for ax in axes:
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_xlabel('QNN latency (µs)')
    axes[1].set_xlabel('HTP nodes')
    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    save(fig, 'preparation', narrow)

    fig, axes = pair(narrow)
    labels = ['NCHW', 'NHWC', 'Permute\nfirst', 'Normalize\nfirst']
    for ax, field, title, ylabel in zip(axes, ('max_tolerance_ratio', 'logits_outside_tolerance'), ('Error vs. allowance', 'Logits outside tolerance'), ('Max error / allowance', 'Count / 1,000')):
        for i, case in enumerate(VISION_CASES):
            row = vision[case]
            values = np.array([r[field] for r in row['jobs']])
            med = float(np.median(values))
            ax.errorbar(i-.10, med, yerr=[[med-values.min()], [values.max()-med]], fmt='o', capsize=3, color=COLORS[1], label='NPU' if i == 0 else None)
            ax.scatter(i+.10, row['cpu_fp16'][field], color='#667085', marker='D', s=28, label='CPU FP16' if i == 0 else None)
        ax.set(title=title, ylabel=ylabel)
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.axvline(1.5, color='#98a2b3', linewidth=.8, linestyle=':')
    axes[0].axhline(1, color='#201d18', linewidth=1, linestyle='--')
    axes[0].text(2.97, 1.025, 'Limit', fontsize=9, ha='right')
    finish_pair(fig, axes, 'vision_validation', narrow, legend=True)

revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
provenance = {
    'source_repository': 'https://github.com/Nzouh/accelerator-bench',
    'source_commit': revision,
    'measurements': 'Saved October 13–14, 2026 AI Hub jobs; no new device runs.',
    'validation': 'Research helpers verify saved HTP/output hashes and recompute logit error counts.',
    'input_image_credit': 'NASA / scikit-image: Eileen Collins, astronaut image; 224×224 crop.',
    'image_source': 'https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut',
    'assets_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file()},
}
(destination.parent / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
print(f'Built 8 figure variants from {revision[:7]}; saved-source checks passed.')
