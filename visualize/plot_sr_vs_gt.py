#!/usr/bin/env python3
"""
Plot SR vs GT-distance bins from eval/analyze_sr_vs_gt.py output.

Default input:
  python3 visualize/plot_sr_vs_gt.py --input_csv results/sr_gt_sweep/analysis/sr_gt_by_setup.csv

Default output:
  results/sr_gt_sweep/analysis/sr_vs_gt_kitti_grid.png
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


SEQ_ORDER = ['01', '04']
REG_ORDER = ['mac', 'teaser', 'kiss']
FEAT_STYLE = {
    'FasterPFH': {'color': '#1f77b4', 'marker': 'o'},
    'FPFH_PCL': {'color': '#d62728', 'marker': 's'},
}


def bin_sort_key(label: str) -> Tuple[float, float, str]:
    cleaned = label.strip().strip('[]')
    try:
        lo_s, hi_s = cleaned.split('-', 1)
        return (float(lo_s), float(hi_s), label)
    except ValueError:
        return (float('inf'), float('inf'), label)


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def index_data(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='results/sr_gt_sweep/analysis/sr_gt_by_setup.csv')
    parser.add_argument('--output_png', default='results/sr_gt_sweep/analysis/sr_vs_gt_kitti_grid.png')
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    data = index_data(rows)
    bin_order = sorted({r.get('gt_bin', '') for r in rows if r.get('gt_bin', '')}, key=bin_sort_key)
    if not bin_order:
        raise RuntimeError('No GT bins found in the input CSV.')

    feat_order = [f for f in ['FasterPFH', 'FPFH_PCL'] if any(r.get('feat') == f for r in rows)]
    if not feat_order:
        feat_order = sorted({r.get('feat', '') for r in rows if r.get('feat', '')})

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharey=True)
    fig.suptitle('KITTI SR vs GT Distance Bins', fontsize=14)

    for i, seq in enumerate(SEQ_ORDER):
        for j, reg in enumerate(REG_ORDER):
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
            ax.set_xticklabels(bin_order)
            ax.set_xlabel('GT Distance Bin (m)')
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel('Success Rate (%)')

            ax.set_ylim(0, 100)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = os.path.dirname(args.output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    print(f'[done] saved plot: {args.output_png}')


if __name__ == '__main__':
    main()
