#!/usr/bin/env python3
"""
Plot GT-binned benchmark summaries and correspondence statistics.

Supported input CSV schemas:
    1) Legacy analyze output (sr_gt_by_setup.csv):
         columns include: seq, reg, feat, gt_bin, sr_percent
    2) Sweep summary output (sr_gt_summary.csv):
         columns include: dataset, scene, method, dist_tag, sr_percent, csv_path

For sweep summaries, this script also reads each run CSV referenced by csv_path
to produce additional GT-binned plots for:
    - average inlier correspondences
    - average outlier correspondences
    - inlier / outlier ratio
    - average RTE over successful runs
    - average RRE over successful runs
    - timing summaries

Examples:
    python3 plot/plot_from_sweep.py \
        --input_csv results/sr_gt_methods_kitti_mulran/sr_gt_summary.csv
        --lang ENG
"""

import os
import csv
import re
import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from i18n import default_i18n_path, load_i18n_labels, tr


FEAT_STYLE = {
    'FasterPFH': {'color': '#1f77b4', 'marker': 'o'},
    'FPFH_PCL': {'color': '#d62728', 'marker': 's'},
}
METHOD_LABEL = {
    'quatro': 'QUATRO',
    'mac': 'MAC',
    'kiss': 'KISS-Matcher',
    'teaser': 'TEASER++',
}
METHOD_STYLE = {
    'QUATRO': {'color': '#1f77b4', 'marker': 'o'},
    'MAC': {'color': '#d62728', 'marker': 's'},
    'KISS-Matcher': {'color': '#2ca02c', 'marker': '^'},
    'TEASER++': {'color': '#ff7f0e', 'marker': 'D'},
}
METHOD_ORDER = ['quatro', 'mac', 'kiss', 'teaser']
TIMING_STAGE_COLS = ['ds_time_s', 'feat_time_s', 'corr_time_s', 'reg_time_s', 'total_time_s']
SUMMARY_METRIC_SPECS = [
    {
        'metric_key': 'sr_percent',
        'title_key': 'title_success_rate',
        'ylabel_key': 'ylabel_success_rate_percent',
        'suffix': 'sr',
        'ylim': (0, 100),
    },
    {
        'metric_key': 'n_inliers',
        'title_key': 'title_average_inliers',
        'ylabel_key': 'ylabel_average_inliers',
        'suffix': 'inliers',
        'ylim': None,
    },
    {
        'metric_key': 'n_outliers',
        'title_key': 'title_average_outliers',
        'ylabel_key': 'ylabel_average_outliers',
        'suffix': 'outliers',
        'ylim': None,
    },
    {
        'metric_key': 'ratio',
        'title_key': 'title_ratio',
        'ylabel_key': 'ylabel_ratio',
        'suffix': 'ratio',
        'ylim': None,
    },
    {
        'metric_key': 'rte_success',
        'title_key': 'title_average_rte',
        'ylabel_key': 'ylabel_rte_m',
        'suffix': 'rte_success',
        'ylim': None,
    },
    {
        'metric_key': 'rre_success',
        'title_key': 'title_average_rre',
        'ylabel_key': 'ylabel_rre_deg',
        'suffix': 'rre_success',
        'ylim': None,
    },
]

def bin_sort_key(label: str) -> Tuple[float, float, str]:
    cleaned = label.strip().strip('[]').replace('_', '-')
    try:
        lo_s, hi_s = cleaned.split('-', 1)
        return (float(lo_s), float(hi_s), label)
    except ValueError:
        return (float('inf'), float('inf'), label)


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def _extract_hms_timestamp(path: str) -> str:
    # Prefer the last HH-MM-SS fragment found in the path.
    matches = re.findall(r'\b\d{2}-\d{2}-\d{2}\b', path)
    return matches[-1] if matches else ''


def _default_output_png_from_input(input_csv: str) -> str:
    csv_dir = os.path.dirname(input_csv) or '.'
    # Keep all plot outputs inside the same run folder as the input CSV.
    return os.path.join(csv_dir, 'plots', 'sr_vs_gt_kitti_grid.png')


def _is_legacy_schema(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    needed = {'seq', 'reg', 'feat', 'gt_bin', 'sr_percent'}
    return needed.issubset(rows[0].keys())


def _is_summary_schema(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    needed = {'dataset', 'scene', 'method', 'dist_tag', 'sr_percent'}
    return needed.issubset(rows[0].keys())


def index_legacy_data(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    data: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for r in rows:
        key = (r['seq'], r['reg'], r['feat'], r['gt_bin'])
        try:
            data[key] = {
                'sr_percent': float(r['sr_percent']),
                'n_pairs_in_bin': float(r['n_pairs_in_bin']),
            }
        except ValueError:
            continue
    return data


def _scene_sort_key(scene: str) -> Tuple[int, str]:
    if scene.isdigit():
        return (0, f'{int(scene):04d}')
    return (1, scene)


def _dist_tag_to_label(tag: str) -> str:
    cleaned = tag.strip().strip('[]')
    parts = cleaned.replace('_', '-').split('-', 1)
    if len(parts) == 2 and all(p.replace('.', '', 1).isdigit() for p in parts):
        return f'{parts[0]}~{parts[1]}'
    return cleaned


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _canonical_method_name(method: str) -> str:
    raw = method.strip().lower()
    aliases = {
        'quatro': 'quatro',
        'mac': 'mac',
        'kiss': 'kiss',
        'kiss-matcher': 'kiss',
        'teaser': 'teaser',
        'teaser++': 'teaser',
    }
    return aliases.get(raw, raw)


def _method_display_name(method: str) -> str:
    canonical = _canonical_method_name(method)
    return METHOD_LABEL.get(canonical, method)


def _method_style(method: str) -> Dict[str, str]:
    display = _method_display_name(method)
    return METHOD_STYLE.get(display, {'color': '#333333', 'marker': 'o'})


def _safe_ratio(num: float, den: float) -> float:
    """Return inlier percentage: num / (num + den) * 100."""
    if not np.isfinite(num) or not np.isfinite(den) or (num + den) <= 0.0:
        return float('nan')
    return float(num / (num + den) * 100.0)


def _is_success_row(row: Dict[str, str]) -> bool:
    # Accept common success flag spellings and treat non-zero as success.
    for key in ('success', 'is_success', 'succ'):
        if key not in row:
            continue
        value = _safe_float(row.get(key, 'nan'))
        if np.isfinite(value):
            return value > 0.0
    return False


def _get_first_finite(row: Dict[str, str], keys: Tuple[str, ...]) -> float:
    for key in keys:
        if key in row:
            value = _safe_float(row.get(key, 'nan'))
            if np.isfinite(value):
                return value
    return float('nan')


def _empty_run_stats() -> Dict[str, float]:
    return {
        'n_corr_init': float('nan'),
        'n_inliers': float('nan'),
        'n_outliers': float('nan'),
        'rte_success': float('nan'),
        'rre_success': float('nan'),
    }


def _analyze_run_csv(run_csv: str) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        'stats': _empty_run_stats(),
        'timing': {k: [] for k in TIMING_STAGE_COLS},
    }

    if not run_csv or not os.path.isfile(run_csv):
        return analysis

    rows = read_rows(run_csv)
    if not rows:
        return analysis

    summary = analysis['stats']

    pair_rows = [row for row in rows if row.get('pair_id') != 'SUMMARY']
    success_rows = [row for row in pair_rows if _is_success_row(row)]

    rte_vals = [
        _get_first_finite(row, ('TE_m', 'RTE_m', 'rte_m', 'te_m'))
        for row in success_rows
    ]
    rte_vals = [v * v for v in rte_vals if np.isfinite(v)]
    if rte_vals:
        summary['rte_success'] = float(np.mean(rte_vals))

    rre_vals = [
        _get_first_finite(row, ('RE_deg', 'RRE_deg', 'rre_deg', 're_deg'))
        for row in success_rows
    ]
    rre_vals = [v for v in rre_vals if np.isfinite(v)]
    if rre_vals:
        summary['rre_success'] = float(np.mean(rre_vals))

    summary_row = next((row for row in rows if row.get('pair_id') == 'SUMMARY'), None)
    if summary_row is not None:
        for key in ('n_corr_init', 'n_inliers', 'n_outliers'):
            summary[key] = _safe_float(summary_row.get(key, 'nan'))
    elif pair_rows:
        for key in ('n_corr_init', 'n_inliers', 'n_outliers'):
            vals = [_safe_float(row.get(key, 'nan')) for row in pair_rows]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                summary[key] = float(np.mean(vals))

    for row in pair_rows:
        for col in TIMING_STAGE_COLS:
            value = _safe_float(row.get(col, 'nan'))
            if np.isfinite(value):
                analysis['timing'][col].append(value)

    return analysis


def _build_run_cache(summary_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        run_csv = row.get('csv_path', '').strip()
        if not run_csv or run_csv in cache:
            continue
        cache[run_csv] = _analyze_run_csv(run_csv)
    return cache


def _collect_run_metric_data(
    summary_rows: List[Dict[str, str]],
    run_cache: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    data: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for row in summary_rows:
        dataset = row.get('dataset', '').strip()
        scene = row.get('scene', '').strip()
        dist_tag = row.get('dist_tag', '').strip()
        method = _canonical_method_name(row.get('method', ''))
        if not dataset or not scene or not dist_tag or not method:
            continue

        run_csv = row.get('csv_path', '').strip()
        run_stats = dict(run_cache.get(run_csv, {}).get('stats', _empty_run_stats()))
        run_stats['sr_percent'] = _safe_float(row.get('sr_percent', 'nan'))
        run_stats['ratio'] = _safe_ratio(run_stats['n_inliers'], run_stats['n_outliers'])
        data[(dataset, scene, method, dist_tag)] = run_stats
    return data


def _metric_output_path(output_png: str, suffix: str, dataset: str, dataset_count: int) -> str:
    base, ext = os.path.splitext(output_png)
    ext = ext or '.png'
    metric_base = base if suffix == 'sr' else f'{base}_{suffix}'
    if dataset_count == 1:
        return f'{metric_base}{ext}'
    return f'{metric_base}_{dataset.lower()}{ext}'


def _create_scene_axes(n_scenes: int):
    ncols = 2 if n_scenes > 1 else 1
    nrows = (n_scenes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 3.8 * nrows), sharey=True)
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    elif ncols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row_axes in axes for ax in row_axes]
    return fig, axes_list


def _collect_metric_values(
    metric_data: Dict[Tuple[str, str, str, str], Dict[str, float]],
    dataset: str,
    scenes: List[str],
    methods: List[str],
    dist_tags: List[str],
    metric_key: str,
) -> List[float]:
    finite_vals: List[float] = []
    for scene in scenes:
        for method in methods:
            for dist_tag in dist_tags:
                value = metric_data.get((dataset, scene, method, dist_tag), {}).get(metric_key, float('nan'))
                if np.isfinite(value):
                    finite_vals.append(float(value))
    return finite_vals


def _plot_scene_metric_bars(
    ax,
    x: np.ndarray,
    dist_tags: List[str],
    metric_data: Dict[Tuple[str, str, str, str], Dict[str, float]],
    dataset: str,
    scene: str,
    methods: List[str],
    metric_key: str,
    bar_width: float,
) -> None:
    for m_idx, method in enumerate(methods):
        y = [metric_data.get((dataset, scene, method, t), {}).get(metric_key, float('nan')) for t in dist_tags]
        style = _method_style(method)
        offset = (m_idx - (len(methods) - 1) / 2.0) * bar_width
        ax.bar(x + offset, y, width=bar_width * 0.78, color=style['color'], label=_method_display_name(method))


def _plot_summary_metric(
    ds_rows: List[Dict[str, str]],
    metric_data: Dict[Tuple[str, str, str, str], Dict[str, float]],
    dataset: str,
    output_png: str,
    dataset_count: int,
    metric_key: str,
    title_key: str,
    ylabel_key: str,
    suffix: str,
    labels: Dict[str, str],
    ylim: Tuple[float, float] | None = None,
) -> None:
    scenes = sorted({r.get('scene', '') for r in ds_rows if r.get('scene', '')}, key=_scene_sort_key)
    methods_present = sorted({_canonical_method_name(r.get('method', '')) for r in ds_rows if r.get('method', '')})
    methods = [m for m in METHOD_ORDER if m in methods_present]
    if not methods:
        methods = methods_present

    dist_tags = sorted({r.get('dist_tag', '') for r in ds_rows if r.get('dist_tag', '')}, key=bin_sort_key)
    if not dist_tags:
        raise RuntimeError(f'No dist_tag values found for dataset {dataset}')

    n_scenes = len(scenes)
    fig, axes_list = _create_scene_axes(len(scenes))

    x = np.arange(len(dist_tags), dtype=float)
    x_labels = [_dist_tag_to_label(t) for t in dist_tags]
    bar_group_width = 0.56
    bar_width = bar_group_width / max(1, len(methods))

    finite_vals = _collect_metric_values(metric_data, dataset, scenes, methods, dist_tags, metric_key)

    for i, scene in enumerate(scenes):
        ax = axes_list[i]
        _plot_scene_metric_bars(ax, x, dist_tags, metric_data, dataset, scene, methods, metric_key, bar_width)

        ax.set_title(f'{tr(labels, "scene_prefix")} {scene}')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(tr(labels, 'xlabel_gt_bin_m'))
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(tr(labels, ylabel_key))

        if ylim is not None:
            ax.set_ylim(*ylim)
        elif finite_vals:
            ymax = max(finite_vals)
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

    for j in range(n_scenes, len(axes_list)):
        axes_list[j].axis('off')

    if axes_list:
        handles, legend_labels = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc='upper center', ncol=len(legend_labels), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = _metric_output_path(output_png, suffix, dataset, dataset_count)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f'saved {tr(labels, title_key).lower()} plot: {out_path}')


def _collect_timing_by_method(
    summary_rows: List[Dict[str, str]],
    run_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    timing: Dict[str, Dict[str, List[float]]] = {}
    stage_cols = TIMING_STAGE_COLS

    for r in summary_rows:
        method = _canonical_method_name(r.get('method', ''))
        run_csv = r.get('csv_path', '').strip()
        if not method or not run_csv:
            continue

        if method not in timing:
            timing[method] = {k: [] for k in stage_cols}

        analysis = run_cache.get(run_csv)
        if analysis is None:
            continue

        for col in stage_cols:
            timing[method][col].extend(analysis.get('timing', {}).get(col, []))

    return timing


def _plot_timing_summary(
    summary_rows: List[Dict[str, str]],
    output_png: str,
    run_cache: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
) -> None:
    timing = _collect_timing_by_method(summary_rows, run_cache)
    if not timing:
        print('[warn] no timing data found via csv_path, skipping timing plots')
        return

    methods_present = list(timing.keys())
    methods = [m for m in METHOD_ORDER if m in methods_present]
    if not methods:
        methods = sorted(methods_present)

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
            vals = timing[method][key]
            stage_means[key].append(float(np.mean(vals)) if vals else 0.0)

        reg_vals = timing[method]['reg_time_s']
        reg_means.append(float(np.mean(reg_vals)) if reg_vals else 0.0)

    y = np.arange(len(methods), dtype=float)

    # 1) Stacked horizontal timing plot by pipeline stage.
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(methods), dtype=float)
    for key, label, color in stage_map:
        vals = np.array(stage_means[key], dtype=float)
        ax1.barh(y, vals, left=left, color=color, label=label, height=0.5)
        left += vals

    ax1.set_yticks(y)
    ax1.set_yticklabels([_method_display_name(m) for m in methods])
    ax1.set_xlabel(tr(labels, 'xlabel_time_s'))
    ax1.set_title(tr(labels, 'title_pipeline_time'))
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.legend(loc='lower right', frameon=False)
    fig1.tight_layout()

    # 2) Registration-only horizontal timing plot.
    fig2, ax2 = plt.subplots(figsize=(8.5, 3.8))
    ax2.barh(y, reg_means, color='#E45756', height=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([_method_display_name(m) for m in methods])
    ax2.set_xlabel(tr(labels, 'xlabel_reg_time_s'))
    ax2.set_title(tr(labels, 'title_registration_time'))
    ax2.grid(True, axis='x', alpha=0.3)
    fig2.tight_layout()

    base, ext = os.path.splitext(output_png)
    ext = ext or '.png'
    out_stacked = f'{base}_timing_stacked{ext}'
    out_reg = f'{base}_timing_registration{ext}'

    out_dir = os.path.dirname(out_stacked)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig1.savefig(out_stacked, dpi=180)
    fig2.savefig(out_reg, dpi=180)
    print(f'saved timing plot: {out_stacked}')
    print(f'saved timing plot: {out_reg}')


def plot_legacy(rows: List[Dict[str, str]], output_png: str, labels: Dict[str, str]) -> None:
    seq_order = ['01', '04']
    reg_order = ['MAC', 'TEASER++', 'KISS-Matcher']
    data = index_legacy_data(rows)
    bin_order = sorted({r.get('gt_bin', '') for r in rows if r.get('gt_bin', '')}, key=bin_sort_key)
    if not bin_order:
        raise RuntimeError('No GT bins found in the input CSV')

    feat_order = [f for f in ['FasterPFH', 'FPFH_PCL'] if any(r.get('feat') == f for r in rows)]
    if not feat_order:
        feat_order = sorted({r.get('feat', '') for r in rows if r.get('feat', '')})

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharey=True)
    # fig.suptitle('KITTI SR vs GT Distance Bins', fontsize=14)

    for i, seq in enumerate(seq_order):
        for j, reg in enumerate(reg_order):
            ax = axes[i, j]
            x = list(range(len(bin_order)))

            for feat in feat_order:
                y = []
                for b in bin_order:
                    v = data.get((seq, reg, feat, b))
                    y.append(v['sr_percent'] if v is not None else float('nan'))

                style = FEAT_STYLE.get(feat, {'color': '#333333', 'marker': 'o'})
                ax.plot(x, y, label=feat, color=style['color'], marker=style['marker'], linewidth=2)

            ax.set_title(f'Seq {seq} | {reg}')
            ax.set_xticks(x)
            ax.set_xticklabels([_dist_tag_to_label(b) for b in bin_order])
            ax.set_xlabel(tr(labels, 'xlabel_gt_bin_m'))
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel(tr(labels, 'ylabel_success_rate_percent'))

            ax.set_ylim(0, 100)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    print(f'saved plot: {output_png}')


def plot_summary(rows: List[Dict[str, str]], output_png: str, labels: Dict[str, str]) -> None:
    datasets = sorted({r.get('dataset', '') for r in rows if r.get('dataset', '')})
    if not datasets:
        raise RuntimeError('No dataset values found in summary CSV')

    run_cache = _build_run_cache(rows)
    metric_data = _collect_run_metric_data(rows, run_cache)

    for dataset in datasets:
        ds_rows = [r for r in rows if r.get('dataset', '') == dataset]
        for spec in SUMMARY_METRIC_SPECS:
            _plot_summary_metric(
                ds_rows,
                metric_data,
                dataset,
                output_png,
                len(datasets),
                metric_key=spec['metric_key'],
                title_key=spec['title_key'],
                ylabel_key=spec['ylabel_key'],
                suffix=spec['suffix'],
                labels=labels,
                ylim=spec['ylim'],
            )

    _plot_timing_summary(rows, output_png, run_cache, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='results/sr_gt_sweep/analysis/sr_gt_by_setup.csv')
    parser.add_argument('--output_png', default='')
    parser.add_argument('--lang', default='ENG', choices=['ENG', 'RUS', 'EN', 'RU'])
    parser.add_argument('--i18n_json', default=default_i18n_path(os.path.dirname(__file__)))
    args = parser.parse_args()

    output_png = args.output_png.strip() if args.output_png else ''
    if not output_png:
        output_png = _default_output_png_from_input(args.input_csv)

    labels = load_i18n_labels(args.i18n_json, args.lang)

    rows = read_rows(args.input_csv)
    if not rows:
        raise RuntimeError('Input CSV has no rows')

    if _is_legacy_schema(rows):
        plot_legacy(rows, output_png, labels)
    elif _is_summary_schema(rows):
        plot_summary(rows, output_png, labels)
    else:
        raise RuntimeError(
            'Unsupported CSV schema'
        )


if __name__ == '__main__':
    main()
