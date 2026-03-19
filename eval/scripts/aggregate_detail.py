#!/usr/bin/env python3
"""
Aggregate an overall_detail.csv (one row per scene per grid cell) into
overall_<method>_<feat>.csv files (one row per grid cell, averaged across scenes).

Usage:
    python3 eval/scripts/aggregate_detail.py <detail_csv> <out_dir>

The detail CSV must have at least these columns:
    method, feat, voxel_size, alpha, beta, rnormal, rFPFH, sr_percent, time_s



python3 eval/scripts/aggregate_detail.py results/feat_research_user/15-07-29/overall_detail.csv results/feat_research_user/15-07-29/overall
    
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict


def to_float(v):
    try:
        return float(v)
    except Exception:
        return float('nan')


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <detail_csv> <out_dir>', file=sys.stderr)
        sys.exit(1)

    in_csv = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with open(in_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        key = (
            r['method'].strip(),
            r['feat'].strip(),
            r['voxel_size'].strip(),
            r['alpha'].strip(),
            r['beta'].strip(),
            r['rnormal'].strip(),
            r['rFPFH'].strip(),
        )
        groups[key].append(r)

    headers = [
        'method', 'feat', 'voxel_size', 'alpha', 'beta', 'rnormal', 'rFPFH',
        'sr_percent_mean', 'time_s_mean', 'n_runs', 'n_valid_sr', 'n_valid_time',
    ]

    summary_rows = []
    for key, grp in groups.items():
        sr_finite = [v for v in (to_float(r.get('sr_percent', 'nan')) for r in grp)
                     if math.isfinite(v)]
        tm_finite = [v for v in (to_float(r.get('time_s', 'nan')) for r in grp)
                     if math.isfinite(v)]
        summary_rows.append({
            'method':          key[0],
            'feat':            key[1],
            'voxel_size':      key[2],
            'alpha':           key[3],
            'beta':            key[4],
            'rnormal':         key[5],
            'rFPFH':           key[6],
            'sr_percent_mean': statistics.fmean(sr_finite) if sr_finite else float('nan'),
            'time_s_mean':     statistics.fmean(tm_finite) if tm_finite else float('nan'),
            'n_runs':          len(grp),
            'n_valid_sr':      len(sr_finite),
            'n_valid_time':    len(tm_finite),
        })

    summary_rows.sort(key=lambda r: (
        r['method'], r['feat'],
        float(r['voxel_size']), float(r['alpha']), float(r['beta'])
    ))

    def write_csv(path, rows_to_write):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows_to_write)
        print(f'[info] wrote {path}  ({len(rows_to_write)} rows)')

    write_csv(os.path.join(out_dir, 'overall_all.csv'), summary_rows)

    by_pair = defaultdict(list)
    for r in summary_rows:
        by_pair[(r['method'], r['feat'])].append(r)

    for (method, feat), pair_rows in by_pair.items():
        write_csv(os.path.join(out_dir, f'overall_{method}_{feat}.csv'), pair_rows)


if __name__ == '__main__':
    main()
