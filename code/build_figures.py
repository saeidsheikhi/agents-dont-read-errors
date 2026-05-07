"""Reproduce the paper's figures and LaTeX tables from analyze.py output.

Run analyze.py first to produce results_summary.json and merged_records.pkl;
then run this script to produce figures/*.pdf and tables/*.tex.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
FIG_DIR = REPO_DIR / 'figures'
TAB_DIR = REPO_DIR / 'paper' / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(CODE_DIR / 'merged_records.pkl')
with open(CODE_DIR / 'results_summary.json') as f:
    summary = json.load(f)

MODELS = ['mistral_7b', 'llama3_1_8b', 'qwen2_5_14b', 'qwen3_14b']
MODEL_LABELS = {
    'mistral_7b':  'Mistral-7B',
    'llama3_1_8b': 'Llama-3.1-8B',
    'qwen2_5_14b': 'Qwen2.5-14B',
    'qwen3_14b':   'Qwen3-14B',
}
CONDITIONS = ['real', 'scrambled_plausible', 'misleading', 'empty']
COND_LABELS = ['Real', 'Scrambled', 'Misleading', 'Empty']
COND_COLORS = ['#2E86AB', '#E89619', '#A23B72', '#6C757D']


# ---- Figure 1: per-condition ERR bar chart with 5pp band ---------------
fig, ax = plt.subplots(figsize=(7.5, 3.5))
x = np.arange(len(MODELS))
width = 0.20
for i, (cond, label, color) in enumerate(zip(CONDITIONS, COND_LABELS, COND_COLORS)):
    means, sems = [], []
    for m in MODELS:
        vals = df[(df.model == m) & (df.condition == cond) & (df.n_errors > 0)]['err'].values.astype(float)
        means.append(vals.mean() if len(vals) else 0.0)
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, label=label,
           color=color, edgecolor='black', linewidth=0.5,
           error_kw={'linewidth': 0.7, 'capsize': 2})
for j, m in enumerate(MODELS):
    real_mean = df[(df.model == m) & (df.condition == 'real') & (df.n_errors > 0)]['err'].mean()
    ax.axhspan(max(0, real_mean - 0.05), real_mean + 0.05,
               xmin=(j - 0.4) / len(MODELS), xmax=(j + 0.4) / len(MODELS),
               color='lightblue', alpha=0.15, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
ax.set_ylabel('Error Recovery Rate (ERR)')
ax.set_ylim(0, 0.85)
ax.legend(loc='upper left', frameon=False, ncol=4, columnspacing=1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', alpha=0.4)
for cond in ['real', 'scrambled_plausible']:
    val = df[(df.model == 'qwen3_14b') & (df.condition == cond) & (df.n_errors > 0)]['err'].mean()
    i = CONDITIONS.index(cond)
    ax.text(3 + (i - 1.5) * width, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_main_bar.pdf')
plt.savefig(FIG_DIR / 'fig1_main_bar.png')
plt.close()
print('Wrote fig1_main_bar')


# ---- Figure 2: engagement-vs-reading trade-off ----------------------
fig, ax = plt.subplots(figsize=(5.5, 3.8))
markers = ['o', 'o', 'o', 's']
colors = ['#777', '#777', '#777', '#C42021']
annot_offset = {
    'Mistral-7B':    (-3, -8, 'right'),
    'Llama-3.1-8B':  (3, 5, 'left'),
    'Qwen2.5-14B':   (-3, 5, 'right'),
    'Qwen3-14B':     (3, 0, 'left'),
}
for m, mk, c in zip(MODELS, markers, colors):
    e = summary['engagement'][m]['real'] * 100
    co = summary['contrasts'][m]
    d = co['delta_pp']
    lo = co['ci_lo_pp']
    hi = co['ci_hi_pp']
    ax.errorbar(e, d, yerr=[[d - lo], [hi - d]], fmt=mk, color=c, ecolor=c,
                markersize=10, capsize=3, elinewidth=1, zorder=3)
    lbl = MODEL_LABELS[m]
    dx, dy, ha = annot_offset[lbl]
    ax.annotate(lbl, (e, d), xytext=(e + dx, d + dy),
                ha=ha, fontsize=9, color=c, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axhspan(-5, 5, color='lightblue', alpha=0.2, zorder=0,
           label='$\\pm$5pp equivalence band')
ax.set_xlabel(r'Engagement rate on real condition (% of trials with $\geq$1 tool call)')
ax.set_ylabel(r'$\Delta$ERR (real $-$ scrambled), pp')
ax.set_xlim(35, 110)
ax.set_ylim(-15, 35)
ax.legend(loc='lower left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_title('Engagement vs error-reading trade-off', fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_engagement_tradeoff.pdf')
plt.savefig(FIG_DIR / 'fig2_engagement_tradeoff.png')
plt.close()
print('Wrote fig2_engagement_tradeoff')


# ---- Figure 3: forest plot -------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.0))
y_pos = np.arange(len(MODELS))[::-1]
labels_rev = [MODEL_LABELS[m] for m in MODELS][::-1]
deltas = [summary['contrasts'][m]['delta_pp'] for m in MODELS][::-1]
los = [summary['contrasts'][m]['ci_lo_pp'] for m in MODELS][::-1]
his = [summary['contrasts'][m]['ci_hi_pp'] for m in MODELS][::-1]
ps_holm = [summary['contrasts'][m]['wilcoxon_p_holm'] for m in MODELS][::-1]
ns = [summary['contrasts'][m]['n_pairs'] for m in MODELS][::-1]
for y, d, l, h, p, n in zip(y_pos, deltas, los, his, ps_holm, ns):
    color = '#C42021' if p < 0.05 else '#5A5A5A'
    ax.errorbar(d, y, xerr=[[d - l], [h - d]], fmt='o', color=color, ecolor=color,
                markersize=8, capsize=4, elinewidth=1.2)
    sig = '*' if p < 0.05 else ''
    ax.text(h + 1.5, y, f'$p={p:.3f}$  $n={n}${sig}',
            va='center', fontsize=8.5, color=color)
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvspan(-5, 5, color='lightblue', alpha=0.15, zorder=0, label='±5pp band')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_rev)
ax.set_xlabel(r'$\Delta$ERR (real $-$ scrambled), percentage points (95% BCa CI)')
ax.set_xlim(-15, 45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_forest.pdf')
plt.savefig(FIG_DIR / 'fig3_forest.png')
plt.close()
print('Wrote fig3_forest')


# ---- Figure 4: condition gradient ------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.5))
for m, marker, color in zip(MODELS, ['o', 's', '^', 'D'],
                              ['#888888', '#1F77B4', '#2CA02C', '#C42021']):
    means = []
    for cond in CONDITIONS:
        v = df[(df.model == m) & (df.condition == cond) & (df.n_errors > 0)]['err'].mean()
        means.append(v)
    ax.plot(range(len(CONDITIONS)), means, marker=marker, linestyle='-',
            label=MODEL_LABELS[m], color=color, markersize=7, linewidth=1.2)
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(COND_LABELS)
ax.set_xlabel('Error-message condition (information content decreasing $\\rightarrow$)')
ax.set_ylabel('Error Recovery Rate (ERR)')
ax.legend(loc='upper right', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_ylim(0, 0.75)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_condition_gradient.pdf')
plt.savefig(FIG_DIR / 'fig4_condition_gradient.png')
plt.close()
print('Wrote fig4_condition_gradient')


# ---- Tables ----------------------------------------------------------
def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)): return '--'
    return f'{x:.{d}f}'

with open(TAB_DIR / 'tab_main.tex', 'w') as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\caption{Per-condition error recovery rate (ERR) and engagement rate
(fraction of 100 MBPP trials in which the agent issued $\geq 1$ tool call).
ERR is computed only over trials with at least one tool error.}
\label{tab:main}
\begin{tabular}{l rrrr r}
\toprule
& \multicolumn{4}{c}{ERR by condition} & Engagement \\
\cmidrule(lr){2-5}
Model & Real & Scrambled & Misleading & Empty & (real) \\
\midrule
""")
    for m in MODELS:
        cells = [fmt(df[(df.model == m) & (df.condition == cond) & (df.n_errors > 0)]['err'].mean(), 3)
                 for cond in CONDITIONS]
        f.write(f"{MODEL_LABELS[m]} & {' & '.join(cells)} & {summary['engagement'][m]['real']:.2f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print('Wrote tab_main.tex')

with open(TAB_DIR / 'tab_contrast.tex', 'w') as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\caption{Primary contrast: paired difference in error recovery rate between
the \textsc{Real} and \textsc{Scrambled} conditions, on tasks where both
trials encountered at least one tool error. CI: 95\% BCa bootstrap interval
on the paired mean difference (2{,}000 resamples). $p$-values are from the
Wilcoxon signed-rank test (Pratt's method for ties), Holm-corrected across
the four model tests. Significance after correction marked with $\ast$.}
\label{tab:contrast}
\begin{tabular}{l r rr rr l}
\toprule
Model & $n$ & ERR$_{\text{real}}$ & ERR$_{\text{scr}}$ & $\Delta$pp & 95\% CI & $p_{\text{Holm}}$ \\
\midrule
""")
    for m in MODELS:
        c = summary['contrasts'][m]
        sig = '\\ast' if c['wilcoxon_p_holm'] is not None and c['wilcoxon_p_holm'] < 0.05 else ''
        f.write(f"{MODEL_LABELS[m]} & {c['n_pairs']} & {fmt(c['err_real'])} & {fmt(c['err_scr'])} & "
                f"${c['delta_pp']:+.2f}$ & $[{c['ci_lo_pp']:+.2f}, {c['ci_hi_pp']:+.2f}]$ & "
                f"${c['wilcoxon_p_holm']:.3f}^{{{sig}}}$ \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print('Wrote tab_contrast.tex')

with open(TAB_DIR / 'tab_csi.tex', 'w') as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\caption{Content Sensitivity Index (CSI) per model.}
\label{tab:csi}
\begin{tabular}{l rrrr r}
\toprule
Model & ERR$_{\text{real}}$ & ERR$_{\text{scr}}$ & ERR$_{\text{mis}}$ & ERR$_{\text{emp}}$ & CSI \\
\midrule
""")
    for m in MODELS:
        c = summary['csi'][m]
        cs_str = '--' if c['csi'] is None else f"{c['csi']:.2f}"
        f.write(f"{MODEL_LABELS[m]} & {fmt(c['err_real'])} & {fmt(c['err_scr'])} & "
                f"{fmt(c['err_mis'])} & {fmt(c['err_emp'])} & {cs_str} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print('Wrote tab_csi.tex')

print('\nAll figures and tables written.')
