#!/usr/bin/env python3
"""
Create heatmaps from an overall_<method>_<feat>.csv file.

Input CSV expected columns:
  method, feat, voxel_size, alpha, beta, rnormal, rFPFH,
  sr_percent_mean, time_s_mean, n_runs, ...

Example:
  python3 plot/plot_feat_research.py \
      --input_csv results/feat_research/12-00-00/overall/overall_kiss_FasterPFH.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from heat_overview_plot import parameter_grid, draw_heatmap  # noqa: E402


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _build_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[float, float, float], Dict[str, float]]:
    lookup: Dict[Tuple[float, float, float], Dict[str, float]] = {}
    for r in rows:
        key = (
            _safe_float(r.get('voxel_size', 'nan')),
            _safe_float(r.get('alpha', 'nan')),
            _safe_float(r.get('beta', 'nan')),
        )
        lookup[key] = {
            'sr': _safe_float(r.get('sr_percent_mean', 'nan')),
            'tm': _safe_float(r.get('time_s_mean', 'nan')),
        }
    return lookup


def _default_output_path(input_csv: str) -> str:
    base = os.path.splitext(os.path.basename(input_csv))[0]
    folder = os.path.join(os.path.dirname(input_csv), 'plots')
    return os.path.join(folder, f'{base}_fig7.png')


def _build_grids(rows: List[Dict[str, str]]):
    voxels, alphas, betas = parameter_grid()
    lookup = _build_lookup(rows)

    sr_data = []
    tm_data = []
    for v in voxels:
        sr = np.full((len(alphas), len(betas)), np.nan, dtype=float)
        tm = np.full((len(alphas), len(betas)), np.nan, dtype=float)
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                metrics = lookup.get((float(v), float(a), float(b)))
                if metrics is None:
                    continue
                sr[i, j] = metrics['sr']
                tm[i, j] = metrics['tm']
        sr_data.append(sr)
        tm_data.append(tm)

    return voxels, alphas, betas, sr_data, tm_data


def _infer_label(rows: List[Dict[str, str]]) -> str:
    methods = sorted({r.get('method', '').strip() for r in rows if r.get('method', '').strip()})
    feats = sorted({r.get('feat', '').strip() for r in rows if r.get('feat', '').strip()})

    method_txt = methods[0] if len(methods) == 1 else '+'.join(methods)
    feat_txt = feats[0] if len(feats) == 1 else '+'.join(feats)
    return f'{method_txt} | {feat_txt}'


def _finite_range(data: np.ndarray, default: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return default

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def _range_ticks(vmin: float, vmax: float) -> List[float]:
    return np.linspace(vmin, vmax, 6).tolist()


def plot_feat_research(input_csv: str, output_png: str, title: str = '') -> str:
    rows = _read_rows(input_csv)
    if not rows:
        raise RuntimeError('Input CSV has no rows')

    voxels, alphas, betas, sr_data, tm_data = _build_grids(rows)

    fig, axes = plt.subplots(2, len(voxels), figsize=(4.8 * len(voxels), 8))

    is_rightmost_col = len(voxels) - 1
    for col, nu in enumerate(voxels):
        has_finite_sr = bool(np.isfinite(sr_data[col]).any())
        draw_heatmap(
            ax=axes[0, col],
            data=sr_data[col],
            title=f'nu = {nu:.1f} m',
            vmin=0,
            vmax=100,
            cbar_label='Success rate [%]',
            fmt='.1f',
            alpha_vals=alphas,
            beta_vals=betas,
            highlight_best=has_finite_sr,
            add_cbar=(col == is_rightmost_col),
        )

    for col, nu in enumerate(voxels):
        tm_min, tm_max = _finite_range(tm_data[col])
        draw_heatmap(
            ax=axes[1, col],
            data=tm_data[col],
            title=f'nu = {nu:.1f} m',
            vmin=tm_min,
            vmax=tm_max,
            cbar_label='Time [s]',
            fmt='.3f',
            alpha_vals=alphas,
            beta_vals=betas,
            highlight_best=False,
            cbar_ticks=_range_ticks(tm_min, tm_max),
            cbar_tick_fmt='%.2f',
            add_cbar=(col == is_rightmost_col),
        )

    fig_title = title.strip() if title.strip() else _infer_label(rows)
    fig.suptitle(
        f'{fig_title} | GT bin 10~12 m | KITTI(01,04)+MulRan(DCC02,RIVERSIDE02,KAIST02)',
        fontsize=12,
    )
    # Manual spacing without tight_layout to avoid conflicts
    # TODO: fix, wrong wspace param working
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, wspace=0.05, hspace=0.2)

    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_png', default='')
    parser.add_argument('--title', default='')
    args = parser.parse_args()

    output_png = args.output_png.strip() if args.output_png else ''
    if not output_png:
        output_png = _default_output_path(args.input_csv)

    out = plot_feat_research(args.input_csv, output_png, args.title)
    print(f'saved plot: {out}')


if __name__ == '__main__':
    main()
