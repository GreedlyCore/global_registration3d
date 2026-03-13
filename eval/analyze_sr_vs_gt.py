#!/usr/bin/env python3
"""
Parse result folders and compute SR vs GT-distance bin statistics.

Expected result structure (from eval/test.py):
  results/.../<run_name>/
    - <run_name>.csv
    - config.json

Outputs are written to <results_root>/analysis/:
  - sr_gt_pairs_long.csv
  - sr_gt_by_run.csv
  - sr_gt_by_setup.csv
"""

import os
import csv
import json
import glob
import argparse
import math
from typing import Dict, List, Tuple, Any, Optional


def parse_bins(spec: str) -> List[Tuple[float, float, str]]:
    bins = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        lo_s, hi_s = token.split('-')
        lo = float(lo_s)
        hi = float(hi_s)
        bins.append((lo, hi, f"[{lo:g}-{hi:g}]"))
    return bins


def infer_bins(values: List[float], target_bin_count: int = 6) -> List[Tuple[float, float, str]]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        lo = math.floor(min_value)
        hi = math.ceil(max_value)
        if math.isclose(lo, hi):
            hi = lo + 1.0
        return [(lo, hi, f"[{lo:g}-{hi:g}]")]

    raw_width = (max_value - min_value) / max(target_bin_count, 1)
    magnitude = 10 ** math.floor(math.log10(raw_width))
    normalized = raw_width / magnitude
    if normalized <= 1:
        nice_step = 1 * magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(min_value / nice_step) * nice_step
    end = math.ceil(max_value / nice_step) * nice_step

    bins = []
    current = start
    while current < end:
        next_edge = current + nice_step
        bins.append((current, next_edge, f"[{current:g}-{next_edge:g}]"))
        current = next_edge
    return bins


def find_bin_label(value: float, bins: List[Tuple[float, float, str]]) -> Optional[str]:
    for lo, hi, label in bins:
        if lo <= value <= hi:
            return label
    return None


def read_run_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('pair_id') == 'SUMMARY':
                continue
            try:
                rows.append({
                    'pair_id': int(r['pair_id']),
                    'src_idx': int(r['src_idx']),
                    'tgt_idx': int(r['tgt_idx']),
                    'success': int(r['success']),
                    'RE_deg': float(r['RE_deg']),
                    'TE_m': float(r['TE_m']),
                    'gt_dist_m': float(r['gt_dist_m']),
                })
            except (ValueError, KeyError):
                continue
    return rows


def find_result_csvs(results_root: str) -> List[str]:
    pattern = os.path.join(results_root, '**', '*.csv')
    all_csvs = glob.glob(pattern, recursive=True)
    return sorted([p for p in all_csvs if os.path.basename(p) != 'sr_gt_pairs_long.csv' and
                   os.path.basename(p) != 'sr_gt_by_run.csv' and
                   os.path.basename(p) != 'sr_gt_by_setup.csv'])


def aggregate_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='results/sr_gt_sweep')
    parser.add_argument(
        '--bins',
        default=None,
        help='Comma-separated GT-distance bins like 0-20,20-40. If omitted, bins are inferred from the observed gt_dist_m range.',
    )
    args = parser.parse_args()

    results_root = args.results_root

    analysis_dir = os.path.join(results_root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    raw_pair_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    by_run: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, Any]]] = {}

    csv_paths = find_result_csvs(results_root)
    if not csv_paths:
        raise RuntimeError(f'No CSV files found under: {results_root}')

    for csv_path in csv_paths:
        run_dir = os.path.dirname(csv_path)
        run_name = os.path.basename(run_dir)
        config_path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(config_path):
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        seq = str(cfg.get('seq', ''))
        dataset = str(cfg.get('dataset', '')).upper()
        if dataset != 'KITTI':
            continue

        reg = str(cfg.get('reg', ''))
        feat = str(cfg.get('feat', ''))
        dist_idx = str(cfg.get('dist_idx', ''))
        if not dist_idx and cfg.get('dist_min') is not None and cfg.get('dist_max') is not None:
            dist_idx = f"{cfg.get('dist_min')}-{cfg.get('dist_max')}"
        voxel_size = str(cfg.get('voxel_size', ''))
        test_type = str(cfg.get('test_type', ''))

        run_rows = read_run_rows(csv_path)
        for r in run_rows:
            raw_pair_rows.append({
                'run_name': run_name,
                'csv_path': csv_path,
                'dataset': dataset,
                'seq': seq,
                'reg': reg,
                'feat': feat,
                'dist_idx': dist_idx,
                'voxel_size': voxel_size,
                'test_type': test_type,
                'pair_id': r['pair_id'],
                'src_idx': r['src_idx'],
                'tgt_idx': r['tgt_idx'],
                'success': r['success'],
                'RE_deg': r['RE_deg'],
                'TE_m': r['TE_m'],
                'gt_dist_m': r['gt_dist_m'],
            })

    if not raw_pair_rows:
        raise RuntimeError('No KITTI pair rows were found under the requested results root.')

    bins = parse_bins(args.bins) if args.bins else infer_bins([r['gt_dist_m'] for r in raw_pair_rows])

    for raw_row in raw_pair_rows:
        bin_label = find_bin_label(raw_row['gt_dist_m'], bins)
        if bin_label is None:
            continue

        row = dict(raw_row)
        row['gt_bin'] = bin_label
        pair_rows.append(row)

        key = (
            row['seq'],
            row['reg'],
            row['feat'],
            row['dist_idx'],
            row['voxel_size'],
            row['test_type'],
            bin_label,
        )
        by_run.setdefault(key, []).append(row)

    if not pair_rows:
        observed_min = min(r['gt_dist_m'] for r in raw_pair_rows)
        observed_max = max(r['gt_dist_m'] for r in raw_pair_rows)
        raise RuntimeError(
            'No rows matched the requested bins. '
            f'Observed gt_dist_m range is [{observed_min:.2f}, {observed_max:.2f}] m.'
        )

    # 1) Long per-pair table
    pairs_csv = os.path.join(analysis_dir, 'sr_gt_pairs_long.csv')
    with open(pairs_csv, 'w', newline='') as f:
        fieldnames = [
            'run_name', 'csv_path', 'dataset', 'seq', 'reg', 'feat',
            'dist_idx', 'voxel_size', 'test_type',
            'pair_id', 'src_idx', 'tgt_idx', 'success', 'RE_deg', 'TE_m', 'gt_dist_m', 'gt_bin',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    # 2) Per-setup (including init params like dist_idx/voxel_size)
    run_summary_rows: List[Dict[str, Any]] = []
    for (seq, reg, feat, dist_idx, voxel_size, test_type, gt_bin), rows in sorted(by_run.items()):
        successes = [r['success'] for r in rows]
        gt_vals = [r['gt_dist_m'] for r in rows]
        run_summary_rows.append({
            'seq': seq,
            'reg': reg,
            'feat': feat,
            'dist_idx': dist_idx,
            'voxel_size': voxel_size,
            'test_type': test_type,
            'gt_bin': gt_bin,
            'n_pairs_in_bin': len(rows),
            'sr_percent': 100.0 * aggregate_mean(successes),
            'gt_dist_mean': aggregate_mean(gt_vals),
        })

    by_run_csv = os.path.join(analysis_dir, 'sr_gt_by_run.csv')
    with open(by_run_csv, 'w', newline='') as f:
        fieldnames = [
            'seq', 'reg', 'feat', 'dist_idx', 'voxel_size', 'test_type',
            'gt_bin', 'n_pairs_in_bin', 'sr_percent', 'gt_dist_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summary_rows)

    # 3) Aggregate by (seq, reg, feat, gt_bin) across init params
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in run_summary_rows:
        grouped.setdefault((r['seq'], r['reg'], r['feat'], r['gt_bin']), []).append(r)

    setup_rows: List[Dict[str, Any]] = []
    for (seq, reg, feat, gt_bin), rows in sorted(grouped.items()):
        sr_weighted = sum(r['sr_percent'] * r['n_pairs_in_bin'] for r in rows)
        n_total = sum(r['n_pairs_in_bin'] for r in rows)
        gt_weighted = sum(r['gt_dist_mean'] * r['n_pairs_in_bin'] for r in rows)

        setup_rows.append({
            'seq': seq,
            'reg': reg,
            'feat': feat,
            'gt_bin': gt_bin,
            'n_runs': len(rows),
            'n_pairs_in_bin': n_total,
            'sr_percent': (sr_weighted / n_total) if n_total > 0 else float('nan'),
            'gt_dist_mean': (gt_weighted / n_total) if n_total > 0 else float('nan'),
        })

    by_setup_csv = os.path.join(analysis_dir, 'sr_gt_by_setup.csv')
    with open(by_setup_csv, 'w', newline='') as f:
        fieldnames = [
            'seq', 'reg', 'feat', 'gt_bin', 'n_runs', 'n_pairs_in_bin', 'sr_percent', 'gt_dist_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(setup_rows)

    print(f'[done] Wrote {pairs_csv}')
    print(f'[done] Wrote {by_run_csv}')
    print(f'[done] Wrote {by_setup_csv}')


if __name__ == '__main__':
    main()
