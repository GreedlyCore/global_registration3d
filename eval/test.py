"""
Evaluate a registration pipeline on KITTI / NCLT / ... sequences.

Usage:
    cd ~/thesis/global_registration3d
    source start.sh
    python eval/test.py --config eval/config/KITTI.json
"""

import sys
sys.path.append('.')
import os
import csv
import json
import logging
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from test_utils import (
    Metrics,
    load_dataset_loader, generate_pairs, create_result_directory,
    gt_transform, compute_metrics,
)
from reg_pipe import run_registration
from test_args import parse_test_args


# Toggle file logging without adding a CLI argument.
ENABLE_FILE_LOGGING = os.getenv('EVAL_ENABLE_FILE_LOGGING', '0').strip().lower() in ('1', 'true', 'yes', 'on')
# Toggle CSV output without adding a CLI argument.
ENABLE_CSV_OUTPUT = os.getenv('EVAL_ENABLE_CSV_OUTPUT', '1').strip().lower() in ('1', 'true', 'yes', 'on')


def eval_sequence(args, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate registration pipeline on a dataset sequence.
    
    Args:
        args: parsed command-line arguments
        cfg: config loaded from JSON, before CLI overrides
    
    Returns:
        List of result rows (each row is a dict with metrics for one pair)
    """
    # Load dataset
    scan_files, poses, Tr, load_pcd, seq = load_dataset_loader(args.dataset, args.seq)
    dataset = args.dataset.upper()
    total_scans = len(scan_files)

    # Generate pairs
    test_type = getattr(args, 'test_type', 'random')
    pairs = generate_pairs(test_type, args, total_scans, poses, Tr)

    csv_path = None
    if ENABLE_CSV_OUTPUT:
        csv_path, config_path, merged_config = create_result_directory(
            args, test_type, dataset, seq, cfg)
        with open(config_path, 'w') as f:
            json.dump(merged_config, f, indent=2)

    # Initialize metrics
    metrics = Metrics(re_threshold=args.re_thre, te_threshold=args.te_thre)

    # CSV fields
    csv_fields = [
        'pair_id', 'src_idx', 'tgt_idx',
        'success', 'RE_deg', 'TE_m', 'gt_dist_m',
        'n_corr_init', 'n_inliers', 'n_outliers',
        'ds_time_s', 'feat_time_s', 'corr_time_s', 'reg_time_s', 'total_time_s',
    ]

    rows: List[Dict[str, Any]] = []
    writer = None
    csv_fh = None

    if ENABLE_CSV_OUTPUT:
        csv_fh = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()

    # Process each pair
    for pair_id, (src_idx, tgt_idx) in enumerate(tqdm(pairs, desc=f'{dataset} {seq}')):
        # Load point clouds
        src_pcd = load_pcd(scan_files[src_idx])
        tgt_pcd = load_pcd(scan_files[tgt_idx])

        # Ground-truth transform
        T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)

        # Run registration
        T_pred, timings, corr_stats = run_registration(
            src_pcd, tgt_pcd,
            voxel_size=args.voxel_size,
            reg_method=args.reg,
            feat_method=args.feat,
            corr_method='nn',
            teaser_cfg=args.teaser,
            mac_cfg=args.mac,
            quatro_cfg=args.quatro,
            feat_cfg=args.feat_cfg,
        )

        # Compute metrics
        metric_dict = compute_metrics(T_pred, T_gt, metrics)

        # Build row
        row = {
            'pair_id':      pair_id,
            'src_idx':      src_idx,
            'tgt_idx':      tgt_idx,
            'success':      metric_dict['success'],
            'RE_deg':       round(metric_dict['RE_deg'], 4),
            'TE_m':         round(metric_dict['TE_m'], 4),
            'gt_dist_m':    round(metric_dict['gt_dist_m'], 4),
            'n_corr_init':  int(corr_stats['n_corr_init']),
            'n_inliers':    int(corr_stats['n_inliers']),
            'n_outliers':   int(corr_stats['n_outliers']),
            'ds_time_s':    round(timings['downsample'], 6),
            'feat_time_s':  round(timings['feature'], 6),
            'corr_time_s':  round(timings['correspondence'], 6),
            'reg_time_s':   round(timings['registration'], 6),
            'total_time_s': round(sum(timings.values()), 6),
        }
        if writer is not None:
            writer.writerow(row)
        rows.append(row)

    # Write summary row
    summary_stats = metrics.compute_summary(rows)
    if writer is not None:
        writer.writerow({
            'pair_id':      'SUMMARY',
            'src_idx':      '',
            'tgt_idx':      f'{summary_stats["n_success"]}/{summary_stats["n_total"]}',
            'success':      round(summary_stats['sr_percent'], 2),
            'RE_deg':       round(summary_stats['mean_rre'], 4),
            'TE_m':         round(summary_stats['mean_rte'], 4),
            'gt_dist_m':    round(summary_stats['mean_gt_dist'], 4),
            'n_corr_init':  round(float(np.mean([r['n_corr_init'] for r in rows])), 2),
            'n_inliers':    round(float(np.mean([r['n_inliers'] for r in rows])), 2),
            'n_outliers':   round(float(np.mean([r['n_outliers'] for r in rows])), 2),
            'ds_time_s':    round(float(np.mean([r['ds_time_s'] for r in rows])), 6),
            'feat_time_s':  round(float(np.mean([r['feat_time_s'] for r in rows])), 6),
            'corr_time_s':  round(float(np.mean([r['corr_time_s'] for r in rows])), 6),
            'reg_time_s':   round(float(np.mean([r['reg_time_s'] for r in rows])), 6),
            'total_time_s': round(summary_stats['mean_time'], 6),
        })
        csv_fh.close()

    logging.info('*' * 50)
    mode_info = (
        f'dist_range=[{args.dist_min},{args.dist_max}]m'
        if test_type == 'random'
        else f'scan2scan({len(pairs)} pairs)'
    )
    logging.info(
        f"[{dataset}] Seq {seq} | {mode_info} | {args.feat}+{args.reg} | N={len(rows)}")
    logging.info(
        f"  SR = {summary_stats['sr_percent']:.2f}%  ({summary_stats['n_success']}/{summary_stats['n_total']})  |  "
        f"Mean RRE = {summary_stats['mean_rre']:.2f} deg  |  Mean RTE = {summary_stats['mean_rte']:.4f} m  |  "
        f"Mean GT dist = {summary_stats['mean_gt_dist']:.2f} m")
    logging.info(
        f"  Failed ({summary_stats['n_failed']})  |  "
        f"Mean RRE = {summary_stats['fail_rre']:.2f} deg  |  Mean RTE = {summary_stats['fail_rte']:.4f} m")
    if ENABLE_CSV_OUTPUT:
        logging.info(f"  Results -> {csv_path}")
    else:
        logging.info('  CSV output disabled (EVAL_ENABLE_CSV_OUTPUT=0).')

    return rows

if __name__ == '__main__':
    args, cfg, _ = parse_test_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if ENABLE_FILE_LOGGING:
        os.makedirs('logs', exist_ok=True)
        if args.test_type == 'random':
            dist_tag = f"d{args.dist_min:g}_{args.dist_max:g}"
        else:
            dist_tag = 'scan2scan'
        tag = f"{args.dataset.lower()}_{args.seq}_{dist_tag}_{args.feat}_{args.reg}"
        handlers.insert(0, logging.FileHandler(f'logs/{tag}.log', mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='',
        handlers=handlers,
    )

    eval_sequence(args, cfg)
