#!/usr/bin/env python3
"""Build per-grid-cell tuning tables from overall_detail.csv.

This script reads an overall_detail-style CSV (one row per scene/run), then
averages metrics across all datasets/scenes for each parameter cell.

Supported use cases:
1) QUATRO parameter tuning
   Output columns:
     noise_bound, noise_bound_coeff, SR, RRE, RTE, OR, Time, n_samples

2) MACPP parameter tuning
   Output columns:
     K1, K2, alpha_dis, SR, RRE, RTE, OR, Time, n_samples

Inputs:
- overall_detail.csv-like file with at least: method, sr_percent/time_s (or csv_path)
- csv_path is used to extract RE/TE and correspondence counts when available

Examples:
  python3 plot/tuning_grid_table.py \
      --input_csv results/feat_research/quatro-solo-test/overall_detail.csv

  python3 plot/tuning_grid_table.py \
      --input_csv results/feat_research/macpp-sweep/overall_detail.csv \
      --method macpp
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from utils import build_run_cache, canonical_method_name


def _safe_series_to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors='coerce')


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ''


def _required_columns_for_method(method: str) -> List[str]:
    if method == 'quatro':
        return ['noise_bound', 'noise_bound_coeff']
    if method == 'macpp':
        return ['K1', 'K2', 'alpha_dis']
    raise RuntimeError(f'Unsupported method: {method}')


def _materialize_quatro_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'noise_bound_coeff' not in out.columns:
        if 'alpha' in out.columns:
            out['noise_bound_coeff'] = out['alpha']
        elif 'quatro_noise_bound_coeff' in out.columns:
            out['noise_bound_coeff'] = out['quatro_noise_bound_coeff']
    if 'noise_bound' not in out.columns:
        if 'beta' in out.columns:
            out['noise_bound'] = out['beta']
        elif 'quatro_noise_bound' in out.columns:
            out['noise_bound'] = out['quatro_noise_bound']
    return out


def _attach_run_metrics(df: pd.DataFrame, input_base_dir: str) -> pd.DataFrame:
    out = df.copy()

    if 'csv_path' not in out.columns:
        out['rre_deg_val'] = np.nan
        out['rte_m_val'] = np.nan
        out['n_inliers_val'] = np.nan
        out['n_outliers_val'] = np.nan
        return out

    csv_paths = out['csv_path'].fillna('').astype(str).tolist()
    cache = build_run_cache(csv_paths, base_dir=input_base_dir)

    rre_vals: List[float] = []
    rte_vals: List[float] = []
    inlier_vals: List[float] = []
    outlier_vals: List[float] = []

    for p in csv_paths:
        stats = cache.get(p.strip(), {})
        rre_vals.append(float(stats.get('rre_deg', np.nan)))
        rte_vals.append(float(stats.get('rte_m', np.nan)))
        inlier_vals.append(float(stats.get('n_inliers', np.nan)))
        outlier_vals.append(float(stats.get('n_outliers', np.nan)))

    out['rre_deg_val'] = rre_vals
    out['rte_m_val'] = rte_vals
    out['n_inliers_val'] = inlier_vals
    out['n_outliers_val'] = outlier_vals
    return out


def _select_method(df: pd.DataFrame, method_arg: str) -> str:
    if method_arg:
        return canonical_method_name(method_arg)

    if 'method' not in df.columns:
        raise RuntimeError('Input CSV has no method column; pass --method explicitly')

    methods = sorted(
        {
            canonical_method_name(m)
            for m in df['method'].fillna('').astype(str).tolist()
            if str(m).strip()
        }
    )
    if not methods:
        raise RuntimeError('Could not infer method from input CSV; pass --method explicitly')
    if len(methods) > 1:
        raise RuntimeError(
            f'Input CSV contains multiple methods {methods}; pass --method to select one'
        )
    return methods[0]


def _build_group_table(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == 'quatro':
        df = _materialize_quatro_columns(df)

    req_cols = _required_columns_for_method(method)
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing parameter columns for {method}: {missing}')

    work = df.copy()
    for c in req_cols:
        work[c] = _safe_series_to_numeric(work, c)

    work = work.dropna(subset=req_cols)
    if work.empty:
        raise RuntimeError('No valid parameter rows found after parsing tuning columns')

    work['SR'] = _safe_series_to_numeric(work, 'sr_percent')
    work['Time'] = _safe_series_to_numeric(work, 'time_s')
    work['RRE'] = _safe_series_to_numeric(work, 'rre_deg_val')
    work['RTE'] = _safe_series_to_numeric(work, 'rte_m_val')

    denom = work['n_inliers_val'] + work['n_outliers_val']
    work['OR'] = np.where(
        denom > 0,
        100.0 * work['n_outliers_val'] / denom,
        np.nan,
    )

    agg_cols = req_cols + ['SR', 'RRE', 'RTE', 'OR', 'Time']
    summary = work.groupby(req_cols, as_index=False)[['SR', 'RRE', 'RTE', 'OR', 'Time']].mean()
    summary['n_samples'] = work.groupby(req_cols).size().values

    summary = summary[req_cols + ['SR', 'RRE', 'RTE', 'OR', 'Time', 'n_samples']]

    for c in agg_cols:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors='coerce')

    summary = summary.sort_values(req_cols, kind='mergesort').reset_index(drop=True)
    return summary


def _default_output_path(input_csv: str, method: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(input_csv))
    return os.path.join(base_dir, f'grid_summary_{method}.csv')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True, help='Path to overall_detail.csv-like file')
    parser.add_argument(
        '--method',
        default='',
        help='Method name (quatro/macpp). If omitted, infer from the input CSV',
    )
    parser.add_argument(
        '--output_csv',
        default='',
        help='Output path. Default: <input_dir>/grid_summary_<method>.csv',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise RuntimeError('Input CSV has no rows')

    method = _select_method(df, args.method.strip())
    if method not in ('quatro', 'macpp'):
        raise RuntimeError(f'Unsupported method: {method}. Supported: quatro, macpp')

    if 'method' in df.columns:
        canon = df['method'].fillna('').astype(str).map(canonical_method_name)
        df = df[canon == method].copy()
        if df.empty:
            raise RuntimeError(f'No rows for method={method} in input CSV')

    df = _attach_run_metrics(df, input_base_dir=os.path.dirname(os.path.abspath(args.input_csv)))
    summary = _build_group_table(df, method)

    out_csv = args.output_csv.strip() if args.output_csv.strip() else _default_output_path(args.input_csv, method)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(out_csv, index=False)
    print(f'saved tuning summary: {out_csv}')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


if __name__ == '__main__':
    main()