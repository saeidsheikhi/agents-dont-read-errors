"""Reproduce per-cell statistics from the released trajectory JSONLs.

Loads all three result files, filters to MBPP-only as in the paper,
computes per-cell ERR / ECR / CSI, BCa bootstrap CIs on the
real-vs-scrambled paired difference, Wilcoxon signed-rank tests
(Pratt's method for ties), and Holm-corrected p-values.

Inputs:  ../data/*_results.jsonl
Outputs: results_summary.json
         merged_records.pkl
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

SOURCES = [
    ('original_1800', DATA_DIR / '20260428T081951Z-06f3bf77_results.jsonl'),
    ('qwen3_8192',    DATA_DIR / '20260428T174000Z-6e3d244c_results.jsonl'),
    ('llama_400',     DATA_DIR / '20260428T214430Z-759dec7b_results.jsonl'),
]

MODELS = ['mistral_7b', 'llama3_1_8b', 'qwen2_5_14b', 'qwen3_14b']
MODEL_LABELS = {
    'mistral_7b':  'Mistral-7B',
    'llama3_1_8b': 'Llama-3.1-8B',
    'qwen2_5_14b': 'Qwen2.5-14B',
    'qwen3_14b':   'Qwen3-14B',
}
CONDITIONS = ['real', 'scrambled_plausible', 'misleading', 'empty']


def load_records() -> list[dict]:
    out = []
    for source, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(
                f'Missing data file: {path}. '
                f'See ../data/README.md for download instructions.'
            )
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                ctx = rec['context']
                model = ctx['model']
                # Use qwen3_8192 for qwen3, original for mistral & qwen2.5,
                # llama_400 for llama. This matches the paper's primary analysis.
                keep = (
                    (source == 'original_1800' and model in ('mistral_7b', 'qwen2_5_14b')) or
                    (source == 'qwen3_8192'    and model == 'qwen3_14b') or
                    (source == 'llama_400'     and model == 'llama3_1_8b')
                )
                if keep and ctx['dataset'] == 'mbpp':
                    out.append(rec)
    return out


def bca_bootstrap_paired_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                               alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """BCa bootstrap CI for mean of paired differences (x - y)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(x) - np.asarray(y)
    n = len(diff)
    if n == 0:
        return (float('nan'), float('nan'))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = diff[idx].mean()
    point = diff.mean()
    z0 = stats.norm.ppf((boot_means < point).mean())
    if not np.isfinite(z0):
        z0 = 0.0
    jk = np.array([np.delete(diff, i).mean() for i in range(n)])
    jk_mean = jk.mean()
    num = ((jk_mean - jk) ** 3).sum()
    den = 6.0 * (((jk_mean - jk) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    return (float(np.quantile(boot_means, p_lo)),
            float(np.quantile(boot_means, p_hi)))


def primary_contrast(df: pd.DataFrame, model: str) -> dict[str, Any]:
    """Real vs scrambled, paired by task_id, on trials where both had >0 errors."""
    sub = df[df.model == model]
    real = sub[(sub.condition == 'real') & (sub.n_errors > 0)].set_index('task_id')['err']
    scr  = sub[(sub.condition == 'scrambled_plausible') & (sub.n_errors > 0)].set_index('task_id')['err']
    common = real.index.intersection(scr.index)
    rv = real.loc[common].values.astype(float)
    sv = scr.loc[common].values.astype(float)
    n = len(common)
    if n < 5:
        return dict(n_pairs=n, err_real=float('nan'), err_scr=float('nan'),
                    delta_pp=float('nan'), ci_lo_pp=float('nan'),
                    ci_hi_pp=float('nan'), wilcoxon_p=float('nan'),
                    sign_p=float('nan'), n_pos=0, n_neg=0, n_zero_diff=0)
    mean_real = float(rv.mean())
    mean_scr = float(sv.mean())
    delta_pp = 100 * (mean_real - mean_scr)
    lo, hi = bca_bootstrap_paired_diff(rv, sv)
    diff = rv - sv
    if (diff != 0).sum() == 0:
        wp = 1.0
    else:
        try:
            wp = float(stats.wilcoxon(rv, sv, zero_method='pratt',
                                       alternative='two-sided').pvalue)
        except Exception:
            wp = float('nan')
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    sign_p = float(2 * stats.binom.cdf(min(n_pos, n_neg), n_pos+n_neg, 0.5)
                    if (n_pos + n_neg) > 0 else 1.0)
    return dict(n_pairs=n, err_real=mean_real, err_scr=mean_scr,
                delta_pp=delta_pp, ci_lo_pp=100*lo, ci_hi_pp=100*hi,
                wilcoxon_p=wp, sign_p=sign_p, n_pos=n_pos, n_neg=n_neg,
                n_zero_diff=int((diff == 0).sum()))


def csi_for(df: pd.DataFrame, model: str) -> dict[str, Any]:
    """CSI = (err_real - err_scrambled) / (err_real - err_empty), clipped to [0, 1]."""
    sub = df[(df.model == model) & (df.n_errors > 0)]
    means = sub.groupby('condition')['err'].mean()
    err_real = float(means.get('real', float('nan')))
    err_scr  = float(means.get('scrambled_plausible', float('nan')))
    err_mis  = float(means.get('misleading', float('nan')))
    err_emp  = float(means.get('empty', float('nan')))
    denom = err_real - err_emp
    if denom <= 0 or not math.isfinite(denom):
        v = float('nan')
    else:
        v = (err_real - err_scr) / denom
        v = max(0.0, min(1.0, v))
    return {'err_real': err_real, 'err_scr': err_scr, 'err_mis': err_mis,
            'err_emp': err_emp, 'csi': v}


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction across a list of p-values."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adj = [None] * len(p_values)
    running_max = 0.0
    k = len(p_values)
    for rank, i in enumerate(order):
        raw = p_values[i]
        factor = k - rank
        val = min(1.0, raw * factor)
        val = max(val, running_max)
        running_max = val
        adj[i] = val
    return adj


def main():
    out_dir = Path(__file__).resolve().parent
    records = load_records()
    print(f'Loaded {len(records)} MBPP records across {len(MODELS)} models × 4 conditions × 100 tasks')
    df = pd.DataFrame([{
        'model': r['context']['model'],
        'condition': r['context']['condition'],
        'task_id': r['context']['task_id'],
        'scaffold': r['context']['scaffold'],
        'task_success': r['metrics']['task_success'],
        'err': r['metrics']['err'],
        'ecr': r['metrics']['ecr'],
        'n_errors': r['metrics']['n_errors'],
        'n_recoveries': r['metrics']['n_recoveries'],
        'wall_time_seconds': r['metrics']['wall_time_seconds'],
        'has_tool_call': any(t.get('kind')=='tool_call' for t in r['trajectory']['turns']),
    } for r in records])
    df.to_pickle(out_dir / 'merged_records.pkl')
    print(f'Wrote {out_dir}/merged_records.pkl')

    contrasts = {m: primary_contrast(df, m) for m in MODELS}
    p_holm = holm_correct([contrasts[m]['wilcoxon_p'] for m in MODELS])
    for m, ph in zip(MODELS, p_holm):
        contrasts[m]['wilcoxon_p_holm'] = ph

    print('\n=== Primary contrasts ===')
    for m in MODELS:
        c = contrasts[m]
        print(f'  {MODEL_LABELS[m]:<15}  err_real={c["err_real"]:.4f}  '
              f'err_scr={c["err_scr"]:.4f}  Δ={c["delta_pp"]:+5.2f}pp  '
              f'CI=[{c["ci_lo_pp"]:+.2f},{c["ci_hi_pp"]:+.2f}]  '
              f'p_Holm={c["wilcoxon_p_holm"]:.4f}  n={c["n_pairs"]}')

    csis = {m: csi_for(df, m) for m in MODELS}
    print('\n=== CSI per cell ===')
    for m in MODELS:
        c = csis[m]
        cs_str = '--' if (isinstance(c['csi'], float) and math.isnan(c['csi'])) else f'{c["csi"]:.3f}'
        print(f'  {MODEL_LABELS[m]:<15}  CSI={cs_str}')

    engagement = {}
    for m in MODELS:
        sub = df[df.model == m]
        engagement[m] = sub.groupby('condition')['has_tool_call'].mean().to_dict()
    print('\n=== Engagement per (model, condition) ===')
    for m in MODELS:
        e = engagement[m]
        print(f'  {MODEL_LABELS[m]:<15}  '
              f'real={e.get("real",0):.2f}  '
              f'scr={e.get("scrambled_plausible",0):.2f}  '
              f'mis={e.get("misleading",0):.2f}  '
              f'emp={e.get("empty",0):.2f}')

    summary = {
        'n_records': len(records),
        'models': MODELS,
        'contrasts': {m: {k: (None if isinstance(v, float) and math.isnan(v) else v)
                          for k, v in contrasts[m].items()}
                      for m in MODELS},
        'csi': {m: {k: (None if isinstance(v, float) and math.isnan(v) else v)
                     for k, v in csis[m].items()} for m in MODELS},
        'engagement': engagement,
    }
    with open(out_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_dir}/results_summary.json')


if __name__ == '__main__':
    main()
