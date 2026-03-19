import os
import json
import argparse
from typing import Tuple, Dict, List, Any, Callable

import numpy as np

from dataset_loader import (
    load_kitti_dataset, load_kitti_velodyne_pcd,
    load_nclt_dataset, load_nclt_velodyne_pcd,
    load_mulran_dataset, load_mulran_ouster_pcd,
    load_oxford_dataset, load_oxford_lidar_pcd,
)
from helpers import gt_transform

class Metrics:
    """Evaluation metrics for pose estimation."""

    def __init__(self, re_threshold: float = 5.0, te_threshold: float = 2.0) -> None:
        """
        Args:
            re_threshold: rotation error threshold in degrees
            te_threshold: translation error threshold in meters
        """
        self.re_threshold = re_threshold
        self.te_threshold = te_threshold

    def rotation_error(self, R_pred: np.ndarray, R_gt: np.ndarray) -> float:
        """
        Compute Rotation Error (RRE) in degrees
        Explanation of arccos formula ...
        https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
        https://math.stackexchange.com/questions/3510272/why-should-the-trace-of-a-3d-rotation-matrix-have-these-properties#3510284
        """
        cos_angle = np.clip( (np.trace(R_pred.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0 )
        return float(np.degrees(np.arccos(cos_angle)))

    def translation_error(self, t_pred: np.ndarray, t_gt: np.ndarray) -> float:
        """Compute per-pair translation error as L2 norm in meters."""
        return float(np.linalg.norm(t_pred.ravel() - t_gt.ravel()))

    def is_success(self, re: float, te: float) -> int:
        """Return 1 if success (within thresholds), 0 otherwise."""
        return int((re < self.re_threshold) and (te < self.te_threshold))

    def compute_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from result rows."""
        successful = [r for r in rows if r['success'] == 1]
        failed = [r for r in rows if r['success'] == 0]
        
        n_success = len(successful)
        sr = n_success / len(rows) * 100 if rows else 0.0

        mean_rre = float(np.mean([r['RE_deg'] for r in successful])) if successful else float('nan')
        mean_rte = float(np.mean([r['TE_m'] for r in successful])) if successful else float('nan')
        fail_rre = float(np.mean([r['RE_deg'] for r in failed])) if failed else float('nan')
        fail_rte = float(np.mean([r['TE_m'] for r in failed])) if failed else float('nan')
        mean_time = float(np.mean([r['total_time_s'] for r in rows])) if rows else float('nan')
        mean_gt_dist = float(np.mean([r['gt_dist_m'] for r in rows])) if rows else float('nan')

        return {
            'n_success': n_success,
            'sr_percent': sr,
            'mean_rre': mean_rre,
            'mean_rte': mean_rte,
            'fail_rre': fail_rre,
            'fail_rte': fail_rte,
            'mean_time': mean_time,
            'mean_gt_dist': mean_gt_dist,
            'n_failed': len(failed),
            'n_total': len(rows),
        }


def load_dataset_loader(dataset: str, seq: str) -> Tuple[List[str], np.ndarray, np.ndarray, Callable, str]:
    """
    Load dataset and return scan files, poses, transformation matrix, loader function.
    
    Args:
           dataset: 'KITTI', 'NCLT', 'MULRAN', or 'OXFORD'
    Returns:
        (scan_files, poses, Tr, load_pcd_fn, normalized_seq)
    """
    dataset = dataset.upper()
    
    if dataset == 'KITTI':
        seq = seq.zfill(2)
        scan_files, poses, Tr = load_kitti_dataset(seq)
        load_pcd = load_kitti_velodyne_pcd
    elif dataset == 'NCLT':
        scan_files, poses, Tr = load_nclt_dataset(seq)
        load_pcd = load_nclt_velodyne_pcd
    elif dataset == 'MULRAN':
        seq = seq.upper()
        scan_files, poses, Tr = load_mulran_dataset(seq)
        load_pcd = load_mulran_ouster_pcd
    elif dataset == 'OXFORD':
        scan_files, poses, Tr = load_oxford_dataset(seq)
        load_pcd = load_oxford_lidar_pcd
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    
    return scan_files, poses, Tr, load_pcd, seq


def generate_pairs(
    test_type: str,
    args: argparse.Namespace,
    total_scans: int,
    poses: np.ndarray,
    Tr: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Generate list of (src_idx, tgt_idx) scan pairs for evaluation.
    
    Args:
        test_type: 'random' or 'scan2scan'
        args: parsed arguments
        total_scans: total number of scans in sequence
    
    Returns:
        List of (src_idx, tgt_idx) tuples
    """
    if test_type == 'scan2scan':
        if not args.test_scans:
            raise ValueError('test_type="scan2scan" requires "test_scans" list in config')
        return [tuple(p) for p in args.test_scans]
    
    # random pairs with GT-distance filtering.
    if args.dist_min is None or args.dist_max is None:
        raise ValueError('random mode requires both --dist_min and --dist_max')
    if args.dist_min > args.dist_max:
        raise ValueError(f'dist_min ({args.dist_min}) must be <= dist_max ({args.dist_max})')

    rng = np.random.default_rng(args.seed)
    offsets = list(range(1, 21))
    pairs: List[Tuple[int, int]] = []
    seen = set()

    # Allow enough attempts to satisfy stricter distance windows.
    max_attempts = max(args.test_count * len(offsets) * 20, 2000)
    for attempt_idx in range(max_attempts):
        offset = offsets[attempt_idx % len(offsets)]
        max_src = total_scans - offset - 1
        if max_src < 0:
            continue

        src_idx = int(rng.integers(0, max_src + 1))
        tgt_idx = src_idx + offset
        pair = (src_idx, tgt_idx)
        if pair in seen:
            continue

        T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)
        gt_dist = float(np.linalg.norm(T_gt[:3, 3]))
        if args.dist_min <= gt_dist <= args.dist_max:
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= args.test_count:
                break

    if len(pairs) < args.test_count:
        raise ValueError(
            f'Could only sample {len(pairs)} pairs in GT-distance range '
            f'[{args.dist_min}, {args.dist_max}] with offsets 1..20 '
            f'after {max_attempts} attempts. Try wider bounds or lower test_count.'
        )

    return pairs



def merge_cli_with_json_config(json_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge JSON config with CLI argument overrides.
    
    CLI arguments take precedence over JSON values.
    
    Returns:
        Merged config dict
    """
    merged = json_cfg.copy()
    merged.update({
        'dataset': args.dataset,
        'seq': args.seq,
        'dist_min': args.dist_min,
        'dist_max': args.dist_max,
        'test_count': args.test_count,
        'feat': args.feat,
        'reg': args.reg,
        'voxel_size': args.voxel_size,
        're_thre': args.re_thre,
        'te_thre': args.te_thre,
        'out_dir': args.out_dir,
        'seed': args.seed,
        'test_type': args.test_type,
    })
    if args.teaser:
        merged['teaser'] = args.teaser
    if args.mac:
        merged['mac'] = args.mac
    if getattr(args, 'quatro', None):
        merged['quatro'] = args.quatro
    if getattr(args, 'feat_cfg', None):
        merged['feat_cfg'] = args.feat_cfg
    if args.test_scans:
        merged['test_scans'] = args.test_scans
    
    return merged


def create_result_directory(args: argparse.Namespace, test_type: str, dataset: str, 
                           seq: str, cfg: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Create result directory and generate merged config.
    
    Args:
        args: parsed arguments
        test_type: 'random' or 'scan2scan'
        dataset: dataset name
        seq: sequence ID
        cfg: original config dict from JSON
    
    Returns:
        (csv_path, config_path, merged_config)
    """
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create result subdirectory
    csv_base = f'{dataset.lower()}_{seq}_{test_type}_{args.feat}_{args.reg}'
    result_dir = os.path.join(args.out_dir, csv_base)
    os.makedirs(result_dir, exist_ok=True)
    
    csv_path = os.path.join(result_dir, f'{csv_base}.csv')
    config_path = os.path.join(result_dir, 'config.json')
    
    # Merge config with CLI overrides
    merged_config = merge_cli_with_json_config(cfg, args)
    
    return csv_path, config_path, merged_config


def compute_metrics(T_pred: np.ndarray, T_gt: np.ndarray, metrics: Metrics) -> Dict[str, float]:
    """
    Compute rotation and translation errors from predicted and ground-truth transforms.
    
    Args:
        T_pred: (4, 4) predicted transform
        T_gt: (4, 4) ground-truth transform
        metrics: Metrics instance
    
    Returns:
        Dict with 'RE_deg', 'TE_m', 'success', 'gt_dist_m'
    """
    re = metrics.rotation_error(T_pred[:3, :3], T_gt[:3, :3])
    te = metrics.translation_error(T_pred[:3, 3], T_gt[:3, 3])
    success = metrics.is_success(re, te)
    gt_dist = float(np.linalg.norm(T_gt[:3, 3]))
    
    return {
        'RE_deg': re,
        'TE_m': te,
        'success': success,
        'gt_dist_m': gt_dist,
    }
