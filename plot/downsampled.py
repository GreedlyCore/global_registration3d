#!/usr/bin/env python3
"""
Plot downsampled point-count curves (mean ± std) vs voxel size.

This script:
1) loads scan files using dataset loaders from eval/dataset_loader.py
2) downsamples each scan with eval/reg_pipe._downsample_tbb
3) aggregates #points across fixed benchmark scenes and plots mean/std bands


python3 plot/downsampled.py
# Faster preview by subsampling scans per scene
python3 plot/downsampled.py --max_scans_per_scene 200

"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
EVAL_DIR = os.path.join(REPO_ROOT, 'eval')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

try:  # noqa: E402
    from eval.dataset_loader import (
        load_kitti_dataset,
        load_kitti_velodyne_pcd,
        load_mulran_dataset,
        load_mulran_ouster_pcd,
        load_oxford_lidar_pcd,
        load_oxford_dataset,
    )
    from eval.reg_pipe import _downsample_tbb
except ImportError:  # noqa: E402
    _dataset_loader = importlib.import_module('dataset_loader')
    load_kitti_dataset = _dataset_loader.load_kitti_dataset
    load_kitti_velodyne_pcd = _dataset_loader.load_kitti_velodyne_pcd
    load_mulran_dataset = _dataset_loader.load_mulran_dataset
    load_mulran_ouster_pcd = _dataset_loader.load_mulran_ouster_pcd
    load_oxford_lidar_pcd = _dataset_loader.load_oxford_lidar_pcd
    _reg_pipe = importlib.import_module('reg_pipe')
    _downsample_tbb = _reg_pipe._downsample_tbb


VOXEL_SIZES = np.array([0.05, 0.10, 0.25, 0.50], dtype=float)
DEFAULT_SCENES: Dict[str, List[str]] = {
    'OXFORD': ['2024-03-18-christ-church-01', '2024-03-18-christ-church-02', '2024-03-20-christ-church-06'],
    'KITTI': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
    'MulRan': ['DCC02', 'RIVERSIDE02', 'KAIST02'],
}
DATASET_DISPLAY = {
    'KITTI': 'KITTI',
    'MulRan': 'MulRan',
    'OXFORD': 'Oxford',
}
DATASET_STYLE = {
    'KITTI': {'color': '#1f77b4', 'marker': 'D'},
    'MulRan': {'color': '#2ca02c', 'marker': '^'},
    'OXFORD': {'color': '#d62728', 'marker': 's'},
}


def _default_output_path() -> str:
    out_dir = os.path.join(REPO_ROOT, 'results', 'downsampled')
    return os.path.join(out_dir, 'downsampled_points.png')


def _load_scan_files(dataset: str, scene: str) -> List[str]:
    if dataset == 'KITTI':
        scan_files, _, _ = load_kitti_dataset(scene)
        return scan_files
    if dataset == 'MulRan':
        scan_files, _, _ = load_mulran_dataset(scene)
        return scan_files
    if dataset == 'OXFORD':
        scan_files, _, _ = load_oxford_dataset(scene)
        return scan_files
    raise ValueError(f'Unsupported dataset: {dataset}')


def _load_scan_pcd(dataset: str, scan_file: str):
    if dataset == 'KITTI':
        return load_kitti_velodyne_pcd(scan_file)
    if dataset == 'MulRan':
        return load_mulran_ouster_pcd(scan_file)
    if dataset == 'OXFORD':
        return load_oxford_lidar_pcd(scan_file)
    raise ValueError(f'Unsupported dataset: {dataset}')


def _select_scan_files(scan_files: List[str], max_scans: int) -> List[str]:
    if max_scans <= 0 or len(scan_files) <= max_scans:
        return scan_files
    idx = np.linspace(0, len(scan_files) - 1, max_scans, dtype=int)
    return [scan_files[i] for i in idx.tolist()]


def _collect_counts_for_dataset(dataset: str, scenes: Sequence[str], max_scans_per_scene: int) -> Dict[float, List[int]]:
    counts: Dict[float, List[int]] = {float(v): [] for v in VOXEL_SIZES}

    for scene in scenes:
        scan_files = _load_scan_files(dataset, scene)
        scan_files = _select_scan_files(scan_files, max_scans=max_scans_per_scene)
        if not scan_files:
            continue

        print(f'[{dataset}:{scene}] processing {len(scan_files)} scans')
        for i, scan_file in enumerate(scan_files, 1):
            if i % 100 == 0:
                print(f'  processed {i}/{len(scan_files)}')

            pcd = _load_scan_pcd(dataset, scan_file)
            for voxel in VOXEL_SIZES:
                ds_pcd = _downsample_tbb(pcd, float(voxel))
                counts[float(voxel)].append(len(ds_pcd.points))

    return counts


def _stats_from_counts(counts: Dict[float, List[int]]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for voxel in VOXEL_SIZES:
        vals = np.asarray(counts[float(voxel)], dtype=float)
        if vals.size == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _plot_curves(dataset_stats: Dict[str, tuple[np.ndarray, np.ndarray]], output_png: str) -> None:
    plt.figure(figsize=(5.5, 4.2))
    x = VOXEL_SIZES

    for dataset, (mean_vals, std_vals) in dataset_stats.items():
        style = DATASET_STYLE[dataset]
        label = DATASET_DISPLAY[dataset]

        lower = mean_vals - std_vals
        upper = mean_vals + std_vals
        plt.fill_between(x, lower, upper, color=style['color'], alpha=0.2)
        plt.plot(
            x,
            mean_vals,
            marker=style['marker'],
            linewidth=2,
            color=style['color'],
            label=label,
        )

    plt.xlabel(r'Voxel size, $v$ [m]')
    plt.ylabel(r'# of points')
    plt.xlim(0.05, 0.5)
    plt.ylim(bottom=0)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    print(f'saved plot: {output_png}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_scans_per_scene',
        type=int,
        default=0,
        help='If >0, uniformly subsample this many scans per scene (0 means all scans).',
    )
    parser.add_argument('--output_png', default='')
    args = parser.parse_args()
    datasets = list(DEFAULT_SCENES.keys())
    selected_scenes = {dataset: list(DEFAULT_SCENES[dataset]) for dataset in datasets}

    output_png = args.output_png.strip() if args.output_png else ''
    if not output_png:
        output_png = _default_output_path()

    dataset_stats: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        scenes = selected_scenes[dataset]
        print(f'[{dataset}] scenes: {", ".join(scenes)}')
        counts = _collect_counts_for_dataset(dataset, scenes, args.max_scans_per_scene)
        means, stds = _stats_from_counts(counts)
        dataset_stats[dataset] = (means, stds)
        print(f'[{dataset}] mean points: {np.array2string(means, precision=1)}')
        print(f'[{dataset}] std points : {np.array2string(stds, precision=1)}')

    _plot_curves(dataset_stats, output_png)


if __name__ == '__main__':
    main()