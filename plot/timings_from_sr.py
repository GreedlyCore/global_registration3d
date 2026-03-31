#!/usr/bin/env python3
"""Generate timing summary plots from a general overall CSV.

Supported inputs:
1) overall_detail.csv-like files with per-scene rows:
   dataset,scene,method,feat,...,time_s,csv_path
2) aggregated overall_*.csv files:
   method,feat,voxel_size,alpha,beta,...,time_s_mean,...

If `csv_path` is available, the script reads referenced run CSV files and reproduces
the same timing functionality that was previously embedded in
`plot_rte_rre_from_sr.py`:
- stacked horizontal bars by stage (downsample/feature/corr/registration)
- registration-only horizontal bars

# When stage-level timings are not available, the script still exports a total-time
# horizontal bar for all methods found in the input CSV.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from i18n import default_i18n_path, load_i18n_labels, tr

from utils import (
    METHOD_ORDER,
    build_run_cache,
    canonical_method_name,
    default_output_dir,
    method_display_name,
    output_path,
    read_csv_df,
)


TIMING_STAGE_COLS = ['ds_time_s', 'feat_time_s', 'corr_time_s', 'reg_time_s', 'total_time_s']


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'method' not in out.columns:
        raise RuntimeError('Input CSV must contain a method column')

    out['method'] = out['method'].map(canonical_method_name)

    if 'time_s' in out.columns:
        out['time_total'] = pd.to_numeric(out['time_s'], errors='coerce')
    elif 'time_s_mean' in out.columns:
        out['time_total'] = pd.to_numeric(out['time_s_mean'], errors='coerce')
    else:
        out['time_total'] = np.nan

    for col in TIMING_STAGE_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    return out


def _ordered_methods(methods: List[str]) -> List[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    tail = sorted([m for m in methods if m not in ordered])
    return ordered + tail


def _collect_stage_timing_from_runs(df: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    timing: Dict[str, Dict[str, List[float]]] = {}
    if 'csv_path' not in df.columns:
        return timing

    run_paths = df['csv_path'].dropna().astype(str).tolist()
    cache = build_run_cache(run_paths)

    for _, row in df.iterrows():
        method = str(row['method']).strip()
        run_csv = str(row.get('csv_path', '')).strip()
        if not method or not run_csv:
            continue

        if method not in timing:
            timing[method] = {k: [] for k in TIMING_STAGE_COLS}

        stats = cache.get(run_csv)
        if not stats:
            continue

        # build_run_cache returns aggregate stats; stage means require direct read fallback.
        # If per-stage isn't available through cache stats, we parse directly from the run csv.
        try:
            run_df = pd.read_csv(run_csv)
        except Exception:
            continue

        if 'pair_id' in run_df.columns:
            run_df = run_df[run_df['pair_id'].astype(str) != 'SUMMARY']

        for col in TIMING_STAGE_COLS:
            if col not in run_df.columns:
                continue
            vals = pd.to_numeric(run_df[col], errors='coerce').dropna().tolist()
            timing[method][col].extend([float(v) for v in vals])

    return timing


def _collect_stage_timing_direct(df: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    timing: Dict[str, Dict[str, List[float]]] = {}
    has_any_stage = any(col in df.columns for col in TIMING_STAGE_COLS)
    if not has_any_stage:
        return timing

    for method, g in df.groupby('method'):
        timing[str(method)] = {k: [] for k in TIMING_STAGE_COLS}
        for col in TIMING_STAGE_COLS:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors='coerce').dropna().tolist()
                timing[str(method)][col].extend([float(v) for v in vals])
    return timing


def _collect_total_timing(df: pd.DataFrame, stage_timing: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    total: Dict[str, float] = {}

    for method, g in df.groupby('method'):
        vals = pd.to_numeric(g['time_total'], errors='coerce').dropna().tolist()
        if vals:
            total[str(method)] = float(np.mean(vals))

    # Fill from stage totals if not present.
    for method, m_timing in stage_timing.items():
        if method in total:
            continue
        vals = m_timing.get('total_time_s', [])
        if vals:
            total[method] = float(np.mean(vals))
            continue
        stack_vals = [
            float(np.mean(m_timing[c]))
            for c in ('ds_time_s', 'feat_time_s', 'corr_time_s', 'reg_time_s')
            if m_timing.get(c)
        ]
        if stack_vals:
            total[method] = float(np.sum(stack_vals))

    return total


def _plot_stage_and_reg(
    stage_timing: Dict[str, Dict[str, List[float]]],
    stem: str,
    labels: Dict[str, str],
) -> List[str]:
    methods_present = [m for m in stage_timing.keys() if stage_timing[m]]
    methods = _ordered_methods(methods_present)
    if not methods:
        return []

    stage_map = [
        ('ds_time_s', tr(labels, 'legend_downsampling'), '#4C78A8'),
        ('feat_time_s', tr(labels, 'legend_features_creation'), '#F58518'),
        ('corr_time_s', tr(labels, 'legend_corr_creation'), '#54A24B'),
        ('reg_time_s', tr(labels, 'legend_registration'), '#E45756'),
    ]

    stage_means: Dict[str, List[float]] = {k: [] for k, _, _ in stage_map}
    reg_means: List[float] = []
    for method in methods:
        for key, _, _ in stage_map:
            vals = stage_timing[method].get(key, [])
            stage_means[key].append(float(np.mean(vals)) if vals else 0.0)
        reg_vals = stage_timing[method].get('reg_time_s', [])
        reg_means.append(float(np.mean(reg_vals)) if reg_vals else 0.0)

    y = np.arange(len(methods), dtype=float)

    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(methods), dtype=float)
    for key, label, color in stage_map:
        vals = np.array(stage_means[key], dtype=float)
        ax1.barh(y, vals, left=left, color=color, label=label, height=0.5)
        left += vals
    ax1.set_yticks(y)
    ax1.set_yticklabels([method_display_name(m) for m in methods])
    ax1.set_xlabel(tr(labels, 'xlabel_time_s'))
    ax1.set_title(tr(labels, 'title_pipeline_time'))
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.legend(loc='lower right', frameon=False)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8.5, 3.8))
    ax2.barh(y, reg_means, color='#E45756', height=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([method_display_name(m) for m in methods])
    ax2.set_xlabel(tr(labels, 'xlabel_reg_time_s'))
    ax2.set_title(tr(labels, 'title_registration_time'))
    ax2.grid(True, axis='x', alpha=0.3)
    fig2.tight_layout()

    out_stacked = f'{stem}_timing_stacked.png'
    out_reg = f'{stem}_timing_registration.png'
    fig1.savefig(out_stacked, dpi=180)
    fig2.savefig(out_reg, dpi=180)
    plt.close(fig1)
    plt.close(fig2)
    return [out_stacked, out_reg]


def _plot_total(total_by_method: Dict[str, float], stem: str, labels: Dict[str, str]) -> str:
    methods = _ordered_methods(list(total_by_method.keys()))
    if not methods:
        raise RuntimeError('No timing values found in CSV')

    y = np.arange(len(methods), dtype=float)
    vals = [total_by_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(y, vals, color='#4C78A8', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([method_display_name(m) for m in methods])
    ax.set_xlabel(tr(labels, 'xlabel_time_s'))
    ax.set_title(tr(labels, 'title_pipeline_time'))
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()

    out_path = f'{stem}_timing_total.png'
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_png', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--lang', default='EN', choices=['EN', 'RU'])
    args = parser.parse_args()

    labels = load_i18n_labels(default_i18n_path(os.path.dirname(__file__)), args.lang)

    df = read_csv_df(args.input_csv)
    if df.empty:
        raise RuntimeError('Input CSV has no rows')
    df = _normalize_df(df)

    if args.output_png.strip():
        stem = os.path.splitext(args.output_png.strip())[0]
        out_dir = os.path.dirname(stem)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = args.output_dir.strip() if args.output_dir.strip() else default_output_dir(args.input_csv)
        os.makedirs(out_dir, exist_ok=True)
        stem = output_path(out_dir, 'timing_summary', '').rstrip('.')

    stage_timing = _collect_stage_timing_from_runs(df)
    if not stage_timing:
        stage_timing = _collect_stage_timing_direct(df)

    exported: List[str] = []
    if stage_timing:
        exported.extend(_plot_stage_and_reg(stage_timing, stem, labels))

    total_by_method = _collect_total_timing(df, stage_timing)
    exported.append(_plot_total(total_by_method, stem, labels))

    for path in exported:
        print(f'saved timing plot: {path}')


if __name__ == '__main__':
    main()