#!/usr/bin/env python3
"""
Build LaTeX-ready tables from a general overall CSV for SR/RRE/RTE/time reporting.

Supported input (current pipeline):
1) overall_detail.csv-like schema:
   dataset,scene,method,feat,dist_min,dist_max,dist_tag,...,sr_percent,time_s,csv_path
2) sr_gt_summary.csv-like schema:
   dataset,scene,method,dist_min,dist_max,dist_tag,sr_percent,csv_path
3) aggregated overall_<method>_<feat>.csv:
   method,feat,voxel_size,alpha,beta,...,sr_percent_mean,time_s_mean,...
   (table-1 cannot be generated from this schema because scene/dist columns are absent)

Outputs:
- sr_table_scene_dist.txt    : LaTeX table body with scene x distance-bin SR (%)
- metrics_table_summary.txt  : LaTeX table body with SR/RRE/RTE(cm)/Time summary
- sr_table_scene_dist.csv    : pivot table data (for debugging/manual edits)
- metrics_table_summary.csv  : summary table data

Example:
  python3 plot/rte_rre_from_sr.py \
    --input_csv /home/sonieth3/Documents/analyze_logs/19-08-33.943/overall_detail.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from i18n import default_i18n_path, load_i18n_labels, tr

from utils import (
    build_run_cache,
    canonical_method_name,
    default_output_dir,
    dist_label,
    dist_sort_key,
    dist_tag_from_row,
    method_feat_label,
    output_path,
    read_csv_df,
    safe_float,
    scene_sort_key,
)


def _normalize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'method' not in out.columns:
        out['method'] = ''
    if 'feat' not in out.columns:
        out['feat'] = ''

    out['method'] = out['method'].map(canonical_method_name)
    out['feat'] = out['feat'].astype(str).str.strip()
    out['method_feat'] = [method_feat_label(m, f) for m, f in zip(out['method'], out['feat'])]

    if 'scene' not in out.columns:
        out['scene'] = ''
    out['scene'] = out['scene'].astype(str).str.strip()

    out['dist_tag_norm'] = out.apply(dist_tag_from_row, axis=1)
    out['dist_label'] = out['dist_tag_norm'].map(dist_label)

    # Normalize SR + timing columns with current-pipeline priority.
    if 'sr_percent' in out.columns:
        out['sr_percent_val'] = pd.to_numeric(out['sr_percent'], errors='coerce')
    elif 'sr_percent_mean' in out.columns:
        out['sr_percent_val'] = pd.to_numeric(out['sr_percent_mean'], errors='coerce')
    else:
        out['sr_percent_val'] = np.nan

    if 'time_s' in out.columns:
        out['time_s_val'] = pd.to_numeric(out['time_s'], errors='coerce')
    elif 'time_s_mean' in out.columns:
        out['time_s_val'] = pd.to_numeric(out['time_s_mean'], errors='coerce')
    else:
        out['time_s_val'] = np.nan

    return out


def _attach_run_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'csv_path' not in out.columns:
        out['rre_deg'] = np.nan
        out['rte_m'] = np.nan
        out['run_time_s'] = np.nan
        return out

    cache = build_run_cache(out['csv_path'].dropna().astype(str).tolist())
    rre_vals: List[float] = []
    rte_vals: List[float] = []
    time_vals: List[float] = []

    for path in out['csv_path'].astype(str).tolist():
        stats = cache.get(path.strip(), {})
        rre_vals.append(safe_float(stats.get('rre_deg', np.nan)))
        rte_vals.append(safe_float(stats.get('rte_m', np.nan)))
        time_vals.append(safe_float(stats.get('time_s', np.nan)))

    out['rre_deg'] = rre_vals
    out['rte_m'] = rte_vals
    out['run_time_s'] = time_vals
    return out


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(arr.mean())


def build_sr_scene_dist_table(df: pd.DataFrame) -> pd.DataFrame:
    needed = {'scene', 'dist_tag_norm', 'sr_percent_val'}
    if not needed.issubset(df.columns):
        raise RuntimeError('Input CSV does not contain scene/dist/SR columns needed for table-1')

    valid = df[
        df['scene'].astype(str).str.len().gt(0)
        & df['dist_tag_norm'].astype(str).str.len().gt(0)
        & pd.to_numeric(df['sr_percent_val'], errors='coerce').notna()
    ].copy()

    if valid.empty:
        raise RuntimeError('No valid scene+distance rows found for SR table')

    grouped = (
        valid.groupby(['method_feat', 'scene', 'dist_tag_norm'], as_index=False)['sr_percent_val']
        .mean()
    )

    pivot = grouped.pivot_table(
        index='method_feat',
        columns=['scene', 'dist_tag_norm'],
        values='sr_percent_val',
        aggfunc='mean',
    )

    # Consistent ordering for scenes and bins.
    scenes = sorted({c[0] for c in pivot.columns}, key=scene_sort_key)
    bins = sorted({c[1] for c in pivot.columns}, key=dist_sort_key)
    ordered_cols: List[Tuple[str, str]] = []
    for scene in scenes:
        for dist in bins:
            if (scene, dist) in pivot.columns:
                ordered_cols.append((scene, dist))

    pivot = pivot[ordered_cols]
    pivot = pivot.sort_index()

    # Pretty labels for latex header (scene remains same, distance becomes 2~6m style).
    pivot.columns = pd.MultiIndex.from_tuples(
        [(scene, dist_label(dist)) for scene, dist in pivot.columns],
        names=['Sequence', 'Distance'],
    )
    return pivot


def build_metrics_summary_table(df: pd.DataFrame, target_dist_tag: str) -> pd.DataFrame:
    work = df.copy()

    if target_dist_tag:
        norm = target_dist_tag.strip().replace('-', '_').replace('~', '_')
        work = work[work['dist_tag_norm'] == norm]

    if work.empty:
        raise RuntimeError('No rows available after applying target distance-bin filter')

    rows = []
    for method_feat, g in work.groupby('method_feat'):
        sr = _mean_or_nan(g['sr_percent_val'].tolist())

        rre_candidates = pd.to_numeric(g.get('rre_deg', np.nan), errors='coerce')
        rte_candidates = pd.to_numeric(g.get('rte_m', np.nan), errors='coerce')
        time_candidates = pd.to_numeric(g.get('time_s_val', np.nan), errors='coerce')

        # Fallback to parsed per-run timing when main aggregate time is absent.
        if np.isnan(_mean_or_nan(time_candidates.tolist())) and 'run_time_s' in g.columns:
            time_candidates = pd.to_numeric(g['run_time_s'], errors='coerce')

        rows.append(
            {
                'Method': method_feat,
                'SR(%)': sr,
                'RRE(deg)': _mean_or_nan(rre_candidates.tolist()),
                'RTE(cm)': _mean_or_nan((rte_candidates * 100.0).tolist()),
                'Time(s)': _mean_or_nan(time_candidates.tolist()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError('Could not build metrics summary table from input rows')

    return out.sort_values('Method').reset_index(drop=True)


def _save_latex_and_csv(df: pd.DataFrame, txt_path: str, csv_path: str, float_fmt: str) -> None:
    latex = df.to_latex(
        index=True if isinstance(df.index, pd.MultiIndex) or df.index.name is not None else False,
        float_format=lambda x: format(x, float_fmt),
        na_rep='-',
        escape=False,
        multicolumn=True,
        multicolumn_format='c',
    )

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    df.to_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--target_dist_tag', default='10_12', help='Used for summary metrics table (e.g., 10_12)')
    parser.add_argument('--lang', default='EN', choices=['EN', 'RU'])
    args = parser.parse_args()

    labels = load_i18n_labels(default_i18n_path(os.path.dirname(__file__)), args.lang)

    output_dir = args.output_dir.strip() if args.output_dir else default_output_dir(args.input_csv)

    raw_df = read_csv_df(args.input_csv)
    if raw_df.empty:
        raise RuntimeError('Input CSV has no rows')

    df = _normalize_core_columns(raw_df)
    df = _attach_run_metrics(df)

    # Table-1: scene x dist SR table.
    if {'scene', 'dist_tag_norm'}.issubset(df.columns) and df['scene'].str.len().gt(0).any():
        sr_table = build_sr_scene_dist_table(df)
        sr_txt = output_path(output_dir, 'sr_table_scene_dist', '.txt')
        sr_csv = output_path(output_dir, 'sr_table_scene_dist', '.csv')
        _save_latex_and_csv(sr_table, sr_txt, sr_csv, '.1f')
        print(f"{tr(labels, 'title_success_rate')}: {sr_txt}")
        print(f'saved table data: {sr_csv}')
    else:
        print(f"[warn] {tr(labels, 'title_success_rate')}: scene/dist columns not available, skipped export")

    # Table-2: SR/RRE/RTE/time table.
    metrics_df = build_metrics_summary_table(df, args.target_dist_tag)
    metrics_txt = output_path(output_dir, 'metrics_table_summary', '.txt')
    metrics_csv = output_path(output_dir, 'metrics_table_summary', '.csv')
    _save_latex_and_csv(metrics_df.set_index('Method'), metrics_txt, metrics_csv, '.2f')
    print(f"{tr(labels, 'title_average_rre')} / {tr(labels, 'title_average_rte')}: {metrics_txt}")
    print(f'saved table data: {metrics_csv}')


if __name__ == '__main__':
    main()
