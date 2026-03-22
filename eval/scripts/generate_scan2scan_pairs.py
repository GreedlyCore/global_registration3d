#!/usr/bin/env python3
"""Generate fixed scan2scan pairs per dataset/scene/dist-bin for sweep runs"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EVAL_ROOT = os.path.join(REPO_ROOT, 'eval')
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from eval.test_utils import generate_pairs, gt_transform, load_dataset_loader


def _validate_pairs(
    pairs: List[Tuple[int, int]],
    test_count: int,
    total_scans: int,
    poses: np.ndarray,
    Tr: np.ndarray,
    dist_min: float,
    dist_max: float,
    dataset: str,
    scene: str,
    dist_tag: str,
) -> None:
    if len(pairs) != test_count:
        raise ValueError(
            f'[{dataset}/{scene}/{dist_tag}] expected {test_count} pairs, got {len(pairs)}'
        )

    if len(set(pairs)) != len(pairs):
        raise ValueError(
            f'[{dataset}/{scene}/{dist_tag}] duplicate pairs found in generated list'
        )

    for src_idx, tgt_idx in pairs:
        if not (0 <= src_idx < total_scans and 0 <= tgt_idx < total_scans):
            raise ValueError(
                f'[{dataset}/{scene}/{dist_tag}] pair out of bounds: ({src_idx}, {tgt_idx}), total_scans={total_scans}'
            )
        if src_idx >= tgt_idx:
            raise ValueError(
                f'[{dataset}/{scene}/{dist_tag}] expected src_idx < tgt_idx, got ({src_idx}, {tgt_idx})'
            )

        T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)
        gt_dist = float(np.linalg.norm(T_gt[:3, 3]))
        if not (dist_min <= gt_dist <= dist_max):
            raise ValueError(
                f'[{dataset}/{scene}/{dist_tag}] pair ({src_idx}, {tgt_idx}) has gt_dist={gt_dist:.6f}, '
                f'outside [{dist_min}, {dist_max}]'
            )


def _build_for_scene(
    dataset: str,
    scene: str,
    dist_bins: List[Tuple[float, float, str]],
    test_count: int,
    seed: int,
) -> Dict[str, List[List[int]]]:
    scan_files, poses, Tr, _load_pcd, resolved_seq = load_dataset_loader(dataset, scene)
    total_scans = len(scan_files)

    out: Dict[str, List[List[int]]] = {}
    for dist_min, dist_max, dist_tag in dist_bins:
        args = SimpleNamespace(
            test_scans=[],
            dist_min=float(dist_min),
            dist_max=float(dist_max),
            test_count=int(test_count),
            seed=int(seed),
        )
        pairs = generate_pairs('random', args, total_scans, poses, Tr)
        _validate_pairs(
            pairs=pairs,
            test_count=test_count,
            total_scans=total_scans,
            poses=poses,
            Tr=Tr,
            dist_min=float(dist_min),
            dist_max=float(dist_max),
            dataset=dataset,
            scene=resolved_seq,
            dist_tag=dist_tag,
        )
        out[dist_tag] = [[int(a), int(b)] for a, b in pairs]

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--test_count', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--kitti_seqs', nargs='*', default=[])
    parser.add_argument('--mulran_seqs', nargs='*', default=[])
    parser.add_argument('--oxford_seqs', nargs='*', default=[])
    parser.add_argument('--dist_mins', nargs='+', required=True, type=float)
    parser.add_argument('--dist_maxs', nargs='+', required=True, type=float)
    parser.add_argument('--dist_tags', nargs='+', required=True)
    args = parser.parse_args()

    if not (len(args.dist_mins) == len(args.dist_maxs) == len(args.dist_tags)):
        raise ValueError('dist_mins, dist_maxs, and dist_tags must have equal lengths')
    if args.test_count <= 0:
        raise ValueError(f'test_count must be positive, got {args.test_count}')

    dist_bins = list(zip(args.dist_mins, args.dist_maxs, args.dist_tags))
    for dmin, dmax, dtag in dist_bins:
        if dmin > dmax:
            raise ValueError(f'Invalid dist bin {dtag}: dist_min ({dmin}) > dist_max ({dmax})')

    result: Dict[str, Dict[str, Dict[str, List[List[int]]]]] = {'pairs': {}}

    if args.kitti_seqs:
        result['pairs']['KITTI'] = {}
        for seq in args.kitti_seqs:
            result['pairs']['KITTI'][seq] = _build_for_scene(
                dataset='KITTI',
                scene=seq,
                dist_bins=dist_bins,
                test_count=args.test_count,
                seed=args.seed,
            )

    if args.mulran_seqs:
        result['pairs']['MulRan'] = {}
        for seq in args.mulran_seqs:
            result['pairs']['MulRan'][seq] = _build_for_scene(
                dataset='MulRan',
                scene=seq,
                dist_bins=dist_bins,
                test_count=args.test_count,
                seed=args.seed,
            )

    if args.oxford_seqs:
        result['pairs']['OXFORD'] = {}
        for seq in args.oxford_seqs:
            result['pairs']['OXFORD'][seq] = _build_for_scene(
                dataset='OXFORD',
                scene=seq,
                dist_bins=dist_bins,
                test_count=args.test_count,
                seed=args.seed,
            )

    result['meta'] = {
        'seed': args.seed,
        'test_count': args.test_count,
        'dist_bins': [
            {'dist_tag': tag, 'dist_min': float(dmin), 'dist_max': float(dmax)}
            for dmin, dmax, tag in dist_bins
        ],
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)

    total_lists = sum(len(scene_bins) for ds in result['pairs'].values() for scene_bins in ds.values())
    print(f'[info] wrote {args.out_json} ({total_lists} lists)')


if __name__ == '__main__':
    main()
