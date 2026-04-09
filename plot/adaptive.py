#!/usr/bin/env python3
"""
Estimate adaptive geometric parameters per scan and export scene-level stats to JSON.

python3 plot/adaptive.py
python3 plot/adaptive.py --max_scans_per_scene 200
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence

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
        load_mulran_dataset,
        load_oxford_dataset,
    )
except ImportError:  # noqa: E402
    _dataset_loader = importlib.import_module('dataset_loader')
    load_kitti_dataset = _dataset_loader.load_kitti_dataset
    load_mulran_dataset = _dataset_loader.load_mulran_dataset
    load_oxford_dataset = _dataset_loader.load_oxford_dataset

try:  # noqa: E402
    adaptive_bootstrap = importlib.import_module('adaptive_bootstrap')
except ImportError as exc:  # noqa: E402
    raise ImportError(
        'Could not import adaptive_bootstrap. Build adaptive python module with '\
        'cmake -DBUILD_PYTHON=ON in adaptive/.') from exc


# DEFAULT_SCENES: Dict[str, List[str]] = {
#     'OXFORD': ['2024-03-18-christ-church-01', '2024-03-18-christ-church-02', '2024-03-20-christ-church-06'],
#     'KITTI': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
#     'MulRan': ['DCC02', 'RIVERSIDE02', 'KAIST02'],
# }

DEFAULT_SCENES: Dict[str, List[str]] = {
    'OXFORD': [],
    'KITTI': ['01', '04', ],
    'MulRan': ['DCC02', 'RIVERSIDE02', 'KAIST02'],
}

VECTOR_KEYS = ('voxel_size', 'r_local', 'r_middle', 'r_global')


def _default_output_path() -> str:
    out_dir = os.path.join(REPO_ROOT, 'results', 'adaptive')
    return os.path.join(out_dir, 'adaptive_params.json')


def _default_progress_path() -> str:
    out_dir = os.path.join(REPO_ROOT, 'results', 'adaptive')
    return os.path.join(out_dir, 'adaptive_progress.ndjson')


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


def _select_scan_files(scan_files: List[str], max_scans: int) -> List[str]:
    if max_scans <= 0 or len(scan_files) <= max_scans:
        return scan_files
    idx = np.linspace(0, len(scan_files) - 1, max_scans, dtype=int)
    return [scan_files[i] for i in idx.tolist()]


def _dataset_to_bind_name(dataset: str) -> str:
    if dataset == 'KITTI':
        return 'kitti'
    if dataset == 'MulRan':
        return 'mulran'
    if dataset == 'OXFORD':
        return 'oxford'
    raise ValueError(f'Unsupported dataset: {dataset}')


def _append_progress(progress_path: str, payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(progress_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(progress_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')


def _collect_scene_vectors(
    dataset: str,
    scene: str,
    scan_files: Sequence[str],
    args: argparse.Namespace,
    progress_path: str,
) -> np.ndarray:
    vectors: List[List[float]] = []
    bind_dataset = _dataset_to_bind_name(dataset)

    for idx, scan_file in enumerate(scan_files, 1):
        if idx % 100 == 0:
            print(f'  processed {idx}/{len(scan_files)}')

        result = adaptive_bootstrap.get_adaptive_params(
            scan_path=scan_file,
            dataset=bind_dataset,
            kappa_spheric=args.kappa_spheric,
            kappa_disc=args.kappa_disc,
            tau_v=args.tau_v,
            delta_v=args.delta_v,
            tau_l=args.tau_l,
            tau_m=args.tau_m,
            tau_g=args.tau_g,
            N_r=args.N_r,
            r_max=args.r_max,
        )

        vectors.append([
            float(result['voxel_size']),
            float(result['r_local']),
            float(result['r_middle']),
            float(result['r_global']),
        ])

    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'dataset': dataset,
        'scene': scene,
        'num_scans': len(vectors),
        'status': 'scene_completed',
    }
    _append_progress(progress_path, payload)

    if not vectors:
        return np.empty((0, 4), dtype=float)
    return np.asarray(vectors, dtype=float)


def _scene_stats(vectors: np.ndarray) -> Dict[str, float]:
    if vectors.size == 0:
        stats: Dict[str, float] = {}
        for name in VECTOR_KEYS:
            stats[f'{name}_mean'] = float('nan')
            stats[f'{name}_std'] = float('nan')
        return stats

    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0)

    stats = {}
    for i, name in enumerate(VECTOR_KEYS):
        stats[f'{name}_mean'] = float(means[i])
        stats[f'{name}_std'] = float(stds[i])
    return stats


def _write_json(output_json: str, results: Dict[str, Dict[str, Dict[str, object]]]) -> None:
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, allow_nan=True)
    print(f'saved json: {output_json}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_scans_per_scene',
        type=int,
        default=0,
        help='If >0, uniformly subsample this many scans per scene (0 means all scans).',
    )
    parser.add_argument('--output_json', default='')
    parser.add_argument('--progress_ndjson', default='')

    parser.add_argument('--kappa_spheric', type=float, default=0.10)
    parser.add_argument('--kappa_disc', type=float, default=0.15)
    parser.add_argument('--tau_v', type=float, default=0.05)
    parser.add_argument('--delta_v', type=float, default=0.10)
    parser.add_argument('--tau_l', type=float, default=0.005)
    parser.add_argument('--tau_m', type=float, default=0.02)
    parser.add_argument('--tau_g', type=float, default=0.05)
    parser.add_argument('--N_r', type=int, default=2000)
    parser.add_argument('--r_max', type=float, default=5.0)
    args = parser.parse_args()

    output_json = args.output_json.strip() if args.output_json else ''
    if not output_json:
        output_json = _default_output_path()

    progress_path = args.progress_ndjson.strip() if args.progress_ndjson else ''
    if not progress_path:
        progress_path = _default_progress_path()

    datasets = list(DEFAULT_SCENES.keys())
    selected_scenes = {dataset: list(DEFAULT_SCENES[dataset]) for dataset in datasets}

    _append_progress(
        progress_path,
        {
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            'status': 'start',
            'datasets': datasets,
        },
    )

    output: Dict[str, Dict[str, Dict[str, object]]] = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        scenes = selected_scenes[dataset]
        print(f'[{dataset}] scenes: {", ".join(scenes)}')

        for scene in scenes:
            scan_files = _load_scan_files(dataset, scene)
            scan_files = _select_scan_files(scan_files, max_scans=args.max_scans_per_scene)
            if not scan_files:
                print(f'[{dataset}:{scene}] no scans')
                output[dataset][scene] = {
                    'num_scans': 0,
                    **_scene_stats(np.empty((0, 4), dtype=float)),
                }
                continue

            print(f'[{dataset}:{scene}] processing {len(scan_files)} scans')
            vectors = _collect_scene_vectors(dataset, scene, scan_files, args, progress_path)
            stats = _scene_stats(vectors)
            output[dataset][scene] = {
                'num_scans': int(vectors.shape[0]),
                **stats,
            }

            stats_preview = np.array([output[dataset][scene][f'{k}_mean'] for k in VECTOR_KEYS], dtype=float)
            print(f'[{dataset}:{scene}] mean vector: {np.array2string(stats_preview, precision=4)}')

    _append_progress(
        progress_path,
        {
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            'status': 'complete',
        },
    )

    _write_json(output_json, output)


if __name__ == '__main__':
    main()
