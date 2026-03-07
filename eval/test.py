"""
Evaluate a registration pipeline on KITTI / NCLT sequences.

Usage:
    cd ~/thesis/investigate
    source start.sh
    python eval/test.py --config eval/config/config_KITTI.json
"""

import sys
sys.path.append('.')
import os
import csv
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm

from dataset_loader import (
    load_kitti_dataset, load_kitti_velodyne_pcd,
    load_nclt_dataset,  load_nclt_velodyne_pcd,
)
from reg_pipe import run_registration


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def rotation_error(R_pred, R_gt):
    """RRE in degrees."""
    cos_angle = np.clip((np.trace(R_pred.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_pred, t_gt):
    """RTE (L2 norm)."""
    return float(np.linalg.norm(t_pred.ravel() - t_gt.ravel()))


def gt_transform(poses, Tr, src_idx, tgt_idx):
    """
    Ground-truth relative transform src → tgt.
    KITTI: poses are camera-frame, Tr is velodyne→camera.
    NCLT:  poses are world-frame (body→world), Tr=eye(4).
    """
    Tr_inv = np.linalg.inv(Tr)
    return Tr_inv @ np.linalg.inv(poses[tgt_idx]) @ poses[src_idx] @ Tr


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def eval_sequence(args):
    dataset = args.dataset.upper()

    if dataset == 'KITTI':
        seq = args.seq.zfill(2)
        scan_files, poses, Tr = load_kitti_dataset(seq)
        load_pcd = load_kitti_velodyne_pcd
    elif dataset == 'NCLT':
        seq = args.seq
        scan_files, poses, Tr = load_nclt_dataset(seq)
        load_pcd = load_nclt_velodyne_pcd
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    total_scans = len(scan_files)

    # ------------------------------------------------------------------ #
    # Sample random pairs
    # ------------------------------------------------------------------ #
    max_src = total_scans - args.dist_idx - 1
    if max_src < 0:
        raise ValueError(
            f'dist_idx={args.dist_idx} too large for {total_scans} scans')

    rng = np.random.default_rng(args.seed)
    src_indices = rng.choice(max_src + 1, size=args.test_count,
                             replace=args.test_count > max_src + 1)

    # ------------------------------------------------------------------ #
    # Output CSV
    # ------------------------------------------------------------------ #
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(
        args.out_dir,
        f'{dataset.lower()}_{seq}_dist{args.dist_idx}_{args.feat}_{args.reg}.csv')

    csv_fields = [
        'pair_id', 'src_idx', 'tgt_idx',
        'success', 'RE_deg', 'TE_m', 'gt_dist_m',
        'feat_time_s', 'corr_time_s', 'reg_time_s', 'total_time_s',
    ]

    rows = []
    n_success = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for pair_id, src_idx in enumerate(tqdm(src_indices, desc=f'{dataset} {seq}')):
            src_idx = int(src_idx)
            tgt_idx = src_idx + args.dist_idx

            src_pcd = load_pcd(scan_files[src_idx])
            tgt_pcd = load_pcd(scan_files[tgt_idx])

            T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)
            T_pred, timings = run_registration(
                src_pcd, tgt_pcd,
                voxel_size=args.voxel_size,
                reg_method=args.reg,
                feat_method=args.feat,
                corr_method='nn',
                teaser_cfg=args.teaser,
                mac_cfg=args.mac,
            )

            Re = rotation_error(T_pred[:3, :3], T_gt[:3, :3])
            Te = translation_error(T_pred[:3, 3], T_gt[:3, 3])
            gt_dist = float(np.linalg.norm(T_gt[:3, 3]))
            success = int((Re < args.re_thre) and (Te < args.te_thre))
            n_success += success

            row = {
                'pair_id':      pair_id,
                'src_idx':      src_idx,
                'tgt_idx':      tgt_idx,
                'success':      success,
                'RE_deg':       round(Re, 4),
                'TE_m':         round(Te, 4),
                'gt_dist_m':    round(gt_dist, 4),
                'feat_time_s':  round(timings['feature'], 4),
                'corr_time_s':  round(timings['correspondence'], 4),
                'reg_time_s':   round(timings['registration'], 4),
                'total_time_s': round(sum(timings.values()), 4),
            }
            writer.writerow(row)
            rows.append(row)

        # ---------------------------------------------------------------- #
        # Summary row
        # ---------------------------------------------------------------- #
        succ = [r for r in rows if r['success'] == 1]
        sr   = n_success / len(rows) * 100

        mean_rre     = float(np.mean([r['RE_deg'] for r in succ])) if succ else float('nan')
        mean_rte     = float(np.mean([r['TE_m']   for r in succ])) if succ else float('nan')
        mean_t       = float(np.mean([r['total_time_s'] for r in rows]))
        mean_gt_dist = float(np.mean([r['gt_dist_m']    for r in rows]))

        writer.writerow({
            'pair_id':      'SUMMARY',
            'src_idx':      '',
            'tgt_idx':      f'{n_success}/{len(rows)}',
            'success':      round(sr, 2),
            'RE_deg':       round(mean_rre, 4),
            'TE_m':         round(mean_rte, 4),
            'gt_dist_m':    round(mean_gt_dist, 4),
            'feat_time_s':  round(float(np.mean([r['feat_time_s'] for r in rows])), 4),
            'corr_time_s':  round(float(np.mean([r['corr_time_s'] for r in rows])), 4),
            'reg_time_s':   round(float(np.mean([r['reg_time_s']  for r in rows])), 4),
            'total_time_s': round(mean_t, 4),
        })

    failed   = [r for r in rows if r['success'] == 0]
    fail_rre = float(np.mean([r['RE_deg'] for r in failed])) if failed else float('nan')
    fail_rte = float(np.mean([r['TE_m']   for r in failed])) if failed else float('nan')

    logging.info('*' * 50)
    logging.info(
        f"[{dataset}] Seq {seq} | dist_idx={args.dist_idx} | {args.feat}+{args.reg} | N={len(rows)}")
    logging.info(
        f"  SR = {sr:.2f}%  ({n_success}/{len(rows)})  |  "
        f"Mean RRE = {mean_rre:.2f} deg  |  Mean RTE = {mean_rte:.4f} m  |  "
        f"Mean GT dist = {mean_gt_dist:.2f} m")
    logging.info(
        f"  Failed ({len(failed)})  |  "
        f"Mean RRE = {fail_rre:.2f} deg  |  Mean RTE = {fail_rte:.4f} m")
    logging.info(f"  Results -> {csv_path}")

    return rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = json.load(f)

    args = argparse.Namespace(
        dataset    = cfg['dataset'],
        seq        = str(cfg['seq']),
        dist_idx   = int(cfg['dist_idx']),
        test_count = int(cfg['test_count']),
        feat       = cfg.get('feat',       'FPFH'),
        reg        = cfg.get('reg',        'teaser'),
        voxel_size = float(cfg.get('voxel_size', 0.5)),
        re_thre    = float(cfg.get('re_thre',    10.0)),
        te_thre    = float(cfg.get('te_thre',    2.0)),
        out_dir    = cfg.get('out_dir',    'results'),
        seed       = int(cfg.get('seed',   42)),
        teaser     = cfg.get('teaser',     {}),
        mac        = cfg.get('mac',        {}),
    )

    os.makedirs('logs', exist_ok=True)
    tag = f"{args.dataset.lower()}_{args.seq}_dist{args.dist_idx}_{args.feat}_{args.reg}"
    logging.basicConfig(
        level=logging.INFO,
        format='',
        handlers=[
            logging.FileHandler(f'logs/{tag}.log', mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    eval_sequence(args)
