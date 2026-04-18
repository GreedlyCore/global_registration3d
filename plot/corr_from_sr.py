#!/usr/bin/env python3
"""Generate correspondence summary plots from a general overall CSV.

Metrics produced:
- average inlier correspondences
- average outlier correspondences
- inlier ratio (%) = n_inliers / (n_inliers + n_outliers) * 100

Input can be:
1) overall_detail.csv-like file with `csv_path` to run-level CSV files
2) a general overall file that already contains correspondence columns

Output:
- one PNG with 3 horizontal bar subplots (inliers/outliers/ratio)
- one CSV summary table with rows grouped by dataset and method
- printed summary in terminal averaged across scenes per dataset
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


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'method' not in out.columns:
        raise RuntimeError('Input CSV must contain method column')

    out['method'] = out['method'].map(canonical_method_name)

    if 'dataset' not in out.columns:
        out['dataset'] = 'ALL'
    out['dataset'] = out['dataset'].astype(str).str.strip().replace({'': 'ALL'})

    if 'scene' not in out.columns:
        out['scene'] = ''
    out['scene'] = out['scene'].astype(str).str.strip()
    return out


def _attach_corr_columns(df: pd.DataFrame, input_base_dir: str = '') -> pd.DataFrame:
    out = df.copy()

    has_direct = all(col in out.columns for col in ('n_inliers', 'n_outliers'))
    if has_direct:
        out['n_inliers_val'] = pd.to_numeric(out['n_inliers'], errors='coerce')
        out['n_outliers_val'] = pd.to_numeric(out['n_outliers'], errors='coerce')
        if 'n_corr_init' in out.columns:
            out['n_corr_init_val'] = pd.to_numeric(out['n_corr_init'], errors='coerce')
        else:
            out['n_corr_init_val'] = np.nan
        return out

    if 'csv_path' not in out.columns:
        raise RuntimeError(
            'Input CSV has no n_inliers/n_outliers columns and no csv_path for run-level extraction'
        )

    cache = build_run_cache(out['csv_path'].dropna().astype(str).tolist(), base_dir=input_base_dir)
    inliers: List[float] = []
    outliers: List[float] = []
    corr_init: List[float] = []

    for path in out['csv_path'].astype(str).tolist():
        stats = cache.get(path.strip(), {})
        inliers.append(float(stats.get('n_inliers', np.nan)))
        outliers.append(float(stats.get('n_outliers', np.nan)))
        corr_init.append(float(stats.get('n_corr_init', np.nan)))

    out['n_inliers_val'] = inliers
    out['n_outliers_val'] = outliers
    out['n_corr_init_val'] = corr_init
    return out


def _aggregate_dataset_method(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work['ratio_percent'] = np.where(
        (work['n_inliers_val'] + work['n_outliers_val']) > 0,
        100.0 * work['n_outliers_val'] / (work['n_inliers_val'] + work['n_outliers_val']),
        np.nan,
    )

    grouped = (
        work.groupby(['dataset', 'method'], as_index=False)[['n_inliers_val', 'n_outliers_val', 'ratio_percent']]
        .mean()
    )
    grouped['method_label'] = grouped['method'].map(method_display_name)
    return grouped


def _ordered_methods(methods: List[str]) -> List[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    tail = sorted([m for m in methods if m not in ordered])
    return ordered + tail


def _plot_metric(ax, summary: pd.DataFrame, metric: str, title: str, xlabel: str) -> None:
    datasets = sorted(summary['dataset'].unique().tolist())
    methods = _ordered_methods(summary['method'].unique().tolist())
    y = np.arange(len(methods), dtype=float)
    width = 0.8 / max(1, len(datasets))

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2']
    for d_idx, dataset in enumerate(datasets):
        vals = []
        for method in methods:
            match = summary[(summary['dataset'] == dataset) & (summary['method'] == method)]
            if match.empty:
                vals.append(np.nan)
            else:
                vals.append(float(match[metric].iloc[0]))
        offset = (d_idx - (len(datasets) - 1) / 2.0) * width
        ax.barh(y + offset, vals, height=width * 0.9, color=colors[d_idx % len(colors)], label=dataset)

    ax.set_yticks(y)
    ax.set_yticklabels([method_display_name(m) for m in methods])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis='x', alpha=0.3)


def _print_summary(summary: pd.DataFrame, labels: Dict[str, str]) -> None:
    print(f"{tr(labels, 'title_average_inliers')} / {tr(labels, 'title_average_outliers')} / {tr(labels, 'title_outlier_ratio')}")
    printable = summary.copy()
    printable = printable[['dataset', 'method_label', 'n_inliers_val', 'n_outliers_val', 'ratio_percent']]
    printable = printable.rename(
        columns={
            'dataset': 'dataset',
            'method_label': 'method',
            'n_inliers_val': 'avg_inliers',
            'n_outliers_val': 'avg_outliers',
            'ratio_percent': 'outlier_ratio_percent',
        }
    )
    print(printable.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_png', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--lang', default='EN', choices=['EN', 'RU'])
    args = parser.parse_args()

    labels = load_i18n_labels(default_i18n_path(os.path.dirname(__file__)), args.lang)

    raw = read_csv_df(args.input_csv)
    if raw.empty:
        raise RuntimeError('Input CSV has no rows')

    df = _normalize_input(raw)
    df = _attach_corr_columns(df, input_base_dir=os.path.dirname(os.path.abspath(args.input_csv)))
    summary = _aggregate_dataset_method(df)
    if summary.empty:
        raise RuntimeError('No correspondence metrics could be aggregated')

    if args.output_png.strip():
        out_png = args.output_png.strip()
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = args.output_dir.strip() if args.output_dir.strip() else default_output_dir(args.input_csv)
        out_png = output_path(out_dir, 'corr_summary', '.png')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    _plot_metric(axes[0], summary, 'n_inliers_val', tr(labels, 'title_average_inliers'), tr(labels, 'ylabel_average_inliers'))
    _plot_metric(axes[1], summary, 'n_outliers_val', tr(labels, 'title_average_outliers'), tr(labels, 'ylabel_average_outliers'))
    _plot_metric(axes[2], summary, 'ratio_percent', tr(labels, 'title_outlier_ratio'), tr(labels, 'ylabel_outlier_ratio'))

    handles, legend_labels = axes[2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc='upper center', ncol=max(1, len(legend_labels)), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_csv = output_path(out_dir, 'corr_summary', '.csv')
    summary.to_csv(out_csv, index=False)

    _print_summary(summary, labels)
    print(f'saved correspondence plot: {out_png}')
    print(f'saved correspondence summary: {out_csv}')


if __name__ == '__main__':
    main()
