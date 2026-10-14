"""Build the article's three benchmark figures from saved, validated results.

Usage: python build_benchmark_figures.py /path/to/accelerator-bench
Requires the accelerator-bench notebook environment. Submits no device jobs.
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
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('research_repo', type=Path)
args = parser.parse_args()
repo = args.research_repo.resolve()
site = Path(__file__).resolve().parents[2]
output = site / 'frontend/articles/snapdragon-layout/figures'
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo / 'src'))
from accelerator_bench.comparison import aggregate_sweep, gpu_matches

gpu_path = repo / 'results/rtx3060-ac-2026-10-13.json'
gpu = json.loads(gpu_path.read_text())
assert gpu['status'] == 'complete' and len(gpu['records']) == 56
assert all(r['status'] == 'ok' and r['correctness']['passed'] and r['graph_correctness']['passed'] for r in gpu['records'])
rows = gpu['records']
remote = aggregate_sweep(repo / 'results/hub-s24-sweep-2026-10-13')
local = gpu_matches(gpu, remote['rows'])
npu = [r for r in remote['rows'] if r['variant'] == 'optimized']
assert len(npu) == 12 and all(r['complete'] and r['n'] == 3 for r in npu)
assert not remote['failures']

TEAL, ORANGE, GRAY, PURPLE = '#087e8b', '#bd4b22', '#667085', '#7965a3'
OPS = [('matmul', 'Matrix multiplication', 'Input rows'), ('conv_relu', 'Convolution + ReLU', 'Image width = height'), ('rmsnorm', 'RMSNorm', 'Input rows')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.titlesize': 11, 'figure.facecolor': 'white', 'svg.hashsalt': 'accelerator-benchmark-article'})


def size(row):
    return row['case']['shape'][2 if row['case']['op'] == 'conv_relu' else 0]


def group(op, dtype='float16', variant='compiled'):
    return sorted((r for r in rows if r['case']['op'] == op and r['dtype'] == dtype and r['variant'] == variant), key=size)


def panels(narrow, height=3.6):
    return plt.subplots(3, 1, figsize=(3.8, 8.8)) if narrow else plt.subplots(1, 3, figsize=(7.5, height))


def finish(fig, axes, name, narrow):
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        ax.set_axisbelow(True)
    fig.legend(handles, labels, loc='lower center', ncol=2 if narrow and len(labels)>2 else len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, .075 if narrow else .12, 1, 1], h_pad=2.3, w_pad=1.0)
    suffix = '-mobile' if narrow else ''
    stem = output / f'{name}{suffix}'
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(stem.with_suffix('.png'), dpi=160)
    plt.close(fig)


for narrow in (False, True):
    fig, axes = panels(narrow)
    for ax, (op, title, xlabel) in zip(axes, OPS):
        selected = sorted((r for r in npu if r['case']['op']==op), key=size)
        x = np.arange(4)
        for metric, label, color, marker, offset in [('graph_cuda_ms','GPU replay',TEAL,'o',-.08), ('wall_ms','GPU via Python',GRAY,'s',0)]:
            measurements = [local[r['case']['id']][metric] for r in selected]
            med = np.array([r['median'] * 1000 for r in measurements])
            error = [med - [r['p10']*1000 for r in measurements], np.array([r['p90']*1000 for r in measurements])-med]
            ax.errorbar(x+offset, med, yerr=error, color=color, marker=marker, ms=4, capsize=2, lw=1.2, label=label)
        med = np.array([r['median_us'] for r in selected])
        error = [med - [r['min_us'] for r in selected], np.array([r['max_us'] for r in selected])-med]
        ax.errorbar(x+.08, med, yerr=error, color=ORANGE, marker='^', ms=4, capsize=2, lw=1.2, label='NPU via QNN')
        ax.set(title=title, xlabel=xlabel, yscale='log', ylim=(.7, 1400))
        ax.set_xticks(x, [str(size(r)) for r in selected])
        ax.set_yticks([1,10,100,1000], ['1','10','100','1,000'])
        ax.grid(axis='y', alpha=.18)
        if narrow or ax is axes[0]: ax.set_ylabel('Time per operation (µs)')
    finish(fig, axes, 'benchmark_latency', narrow)

    fig, axes = panels(narrow, height=3.9)
    for ax, (op, title, _) in zip(axes, OPS):
        for dtype, label, color, offset in [('float16','FP16',TEAL,-.15), ('float32','FP32',PURPLE,.15)]:
            eager, compiled = group(op,dtype,'eager'), group(op,dtype,'compiled')
            assert [r['case'] for r in eager] == [r['case'] for r in compiled]
            ratios = [a['graph_cuda_ms']['median']/b['graph_cuda_ms']['median'] for a,b in zip(eager,compiled)]
            y = np.arange(4)+offset
            ax.hlines(y, np.minimum(1,ratios), np.maximum(1,ratios), color=color, lw=2)
            ax.scatter(ratios,y,color=color,s=25,label=label,zorder=3)
            for ratio, pos in zip(ratios,y):
                ax.annotate(f'{ratio:.2f}×',(ratio,pos),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        ax.axvline(1,color='#98a2b3',lw=1,ls='--',zorder=0)
        ax.set_xscale('log',base=2)
        ax.set_xlim(.5,24)
        ax.set_xticks([.5,1,2,4,8,16],['0.5','1','2','4','8','16'])
        ax.tick_params(axis='x',labelsize=8)
        ax.set_yticks(range(4),[str(size(r)) for r in eager])
        ax.set_ylim(3.6,-.6)
        ax.set(title=title,xlabel='Before / after time (×)')
        ax.set_ylabel('Image width = height' if op=='conv_relu' else 'Input rows')
        ax.grid(axis='x',alpha=.12)
    finish(fig, axes, 'compiler_speedup', narrow)

    fig, axes = panels(narrow)
    for ax, (op,title,xlabel) in zip(axes,OPS):
        for variant,label,color,marker in [('eager','Eager',GRAY,'s'),('compiled','Compiled',TEAL,'o'),('fused','Hand-fused',ORANGE,'^')]:
            selected=group(op,variant=variant)
            if not selected: continue
            assert all(r['energy']['status']=='measured' for r in selected)
            energy=[r['energy']['gpu_energy_mj_per_execution'] for r in selected]
            ax.plot(range(4),energy,marker=marker,color=color,ms=4,lw=1.3,label=label)
        ax.set(title=title,xlabel=xlabel,yscale='log',ylim=(.03,20))
        ax.set_yticks([.1,1,10],['0.1','1','10'])
        ax.set_xticks(range(4),[str(size(r)) for r in group(op)])
        ax.grid(axis='y',alpha=.18)
        if narrow or ax is axes[0]: ax.set_ylabel('Energy per operation (mJ)')
    finish(fig, axes, 'gpu_energy', narrow)

profile_path=repo/'results/rtx3060-ac-rmsnorm-profile.json'
profile=json.loads(profile_path.read_text())
assert profile['variants']['eager']['cuda_activities_per_call']==9
assert profile['variants']['compiled']['cuda_activities_per_call']==1
spotlight={variant:next(r for r in group('rmsnorm',variant=variant) if size(r)==32) for variant in ('eager','compiled','fused')}
summary={v:{'graph_us':r['graph_cuda_ms']['median']*1000,'python_us':r['wall_ms']['median']*1000,
    'power_w':r['energy']['mean_gpu_power_w'],'energy_mj':r['energy']['gpu_energy_mj_per_execution']} for v,r in spotlight.items()}
provenance={'source_repository':'https://github.com/Nzouh/accelerator-bench',
    'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (gpu_path,profile_path,repo/'results/hub-s24-sweep-2026-10-13/results.json')},
    'figure_basis':'All 12 shapes; compiled/eager medians; FP16 energy sweep. No new runs or NPU energy estimates.',
    'rmsnorm_fp16_32x1024':summary,
    'presentation_references':['https://pytorch.org/blog/accelerated-pytorch-inference/','https://www.collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt/']}
(Path(__file__).parent/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Built 3 figures in desktop/mobile variants; 56 GPU configurations and 12 matched NPU cases checked.')
print(json.dumps(summary,indent=2))
