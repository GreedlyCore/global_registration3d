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
import argparse
import logging
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from test_utils import (
    Metrics,
    load_dataset_loader, generate_pairs, create_result_directory,
    gt_transform, compute_metrics,
)
from helpers import resolve_feature_cfg
from reg_pipe import run_registration


def eval_sequence(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Evaluate registration pipeline on a dataset sequence.
    
    Args:
        args: parsed command-line arguments
    
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

    # Setup result directory and config
    csv_path, config_path, merged_config = create_result_directory(
        args, test_type, dataset, seq, cfg)
    
    # Write config to file
    with open(config_path, 'w') as f:
        json.dump(merged_config, f, indent=2)

    # Initialize metrics
    metrics = Metrics(re_threshold=args.re_thre, te_threshold=args.te_thre)

    # CSV fields
    csv_fields = [
        'pair_id', 'src_idx', 'tgt_idx',
        'success', 'RE_deg', 'TE_m', 'gt_dist_m',
        'ds_time_s', 'feat_time_s', 'corr_time_s', 'reg_time_s', 'total_time_s',
    ]

    rows: List[Dict[str, Any]] = []
    n_success = 0

    # Process each pair
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for pair_id, (src_idx, tgt_idx) in enumerate(tqdm(pairs, desc=f'{dataset} {seq}')):
            # Load point clouds
            src_pcd = load_pcd(scan_files[src_idx])
            tgt_pcd = load_pcd(scan_files[tgt_idx])

            # Ground-truth transform
            T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)
            
            # Run registration
            T_pred, timings = run_registration(
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
            success = metric_dict['success']
            n_success += success

            # Build row
            row = {
                'pair_id':      pair_id,
                'src_idx':      src_idx,
                'tgt_idx':      tgt_idx,
                'success':      success,
                'RE_deg':       round(metric_dict['RE_deg'], 4),
                'TE_m':         round(metric_dict['TE_m'], 4),
                'gt_dist_m':    round(metric_dict['gt_dist_m'], 4),
                'ds_time_s':    round(timings['downsample'], 4),
                'feat_time_s':  round(timings['feature'], 4),
                'corr_time_s':  round(timings['correspondence'], 4),
                'reg_time_s':   round(timings['registration'], 4),
                'total_time_s': round(sum(timings.values()), 4),
            }
            writer.writerow(row)
            rows.append(row)

        # Write summary row
        summary_stats = metrics.compute_summary(rows)
        writer.writerow({
            'pair_id':      'SUMMARY',
            'src_idx':      '',
            'tgt_idx':      f'{summary_stats["n_success"]}/{summary_stats["n_total"]}',
            'success':      round(summary_stats['sr_percent'], 2),
            'RE_deg':       round(summary_stats['mean_rre'], 4),
            'TE_m':         round(summary_stats['mean_rte'], 4),
            'gt_dist_m':    round(summary_stats['mean_gt_dist'], 4),
            'ds_time_s':    round(float(np.mean([r['ds_time_s'] for r in rows])), 4),
            'feat_time_s':  round(float(np.mean([r['feat_time_s'] for r in rows])), 4),
            'corr_time_s':  round(float(np.mean([r['corr_time_s'] for r in rows])), 4),
            'reg_time_s':   round(float(np.mean([r['reg_time_s'] for r in rows])), 4),
            'total_time_s': round(summary_stats['mean_time'], 4),
        })

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
    logging.info(f"  Results -> {csv_path}")

    return rows

if __name__ == '__main__':
    # --- pass 1: grab --config only, ignore everything else ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', required=True)
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        cfg = json.load(f)

    # --- pass 2: full parser, JSON values are defaults, CLI wins ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--dataset',    default=cfg.get('dataset'))
    parser.add_argument('--seq',        default=str(cfg.get('seq', '')))
    parser.add_argument('--dist_min',   type=float, default=cfg.get('dist_min'))
    parser.add_argument('--dist_max',   type=float, default=cfg.get('dist_max'))
    parser.add_argument('--test_count', type=int,   default=cfg.get('test_count', 100))
    parser.add_argument('--feat',       default=cfg.get('feat',       'FPFH'))
    parser.add_argument('--reg',        default=cfg.get('reg',        'teaser'))
    parser.add_argument('--voxel_size', type=float, default=cfg.get('voxel_size', 0.5))
    parser.add_argument('--re_thre',    type=float, default=cfg.get('re_thre',    10.0))
    parser.add_argument('--te_thre',    type=float, default=cfg.get('te_thre',    2.0))
    parser.add_argument('--out_dir',    default=cfg.get('out_dir',    'results'))
    parser.add_argument('--seed',       type=int,   default=cfg.get('seed',       42))
    parser.add_argument('--test_type',  default=cfg.get('test_type',  'random'),
                        choices=['random', 'scan2scan'])
    # teaser/mac/quatro sub-configs are not overridable from CLI (use JSON for those)
    args = parser.parse_args()
    args.teaser     = cfg.get('teaser',     {})
    args.mac        = cfg.get('mac',        {})
    args.quatro     = cfg.get('quatro',     {})
    args.feat_cfg   = resolve_feature_cfg(cfg, args.feat)
    args.test_scans = cfg.get('test_scans', [])

    if args.test_type == 'random' and (args.dist_min is None or args.dist_max is None):
        parser.error('random mode requires both --dist_min and --dist_max (in meters)')

    os.makedirs('logs', exist_ok=True)
    if args.test_type == 'random':
        dist_tag = f"d{args.dist_min:g}_{args.dist_max:g}"
    else:
        dist_tag = 'scan2scan'
    tag = f"{args.dataset.lower()}_{args.seq}_{dist_tag}_{args.feat}_{args.reg}"
    logging.basicConfig(
        level=logging.INFO,
        format='',
        handlers=[
            logging.FileHandler(f'logs/{tag}.log', mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    eval_sequence(args)
