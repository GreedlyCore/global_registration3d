"""
MAC parameter sweep: one (tcmp, dcmp, dist_bin) configuration.

Pair distances are pre-computed once and cached; subsequent runs for the
same dataset/seq skip the expensive distance scan.

Usage (called by eval/scripts/sweep_mac_params.sh):
    python eval/sweep_mac.py \
        --config eval/config/KITTI.json \
        --tcmp 0.99 --dcmp 2.0 --dist_bin 1 \
        --out_dir results/sweep_mac

Distance bins:
    0 : [2,  6)  m
    1 : [6, 10)  m
    2 : [10, 12] m
"""

import sys
sys.path.append('.')

import csv
import json
import logging
import math
import os
import time
import argparse

import numpy as np
from tqdm import tqdm

from dataset_loader import (
    load_kitti_dataset, load_kitti_velodyne_pcd,
    load_nclt_dataset,  load_nclt_velodyne_pcd,
)
from helpers import extract_fpfh, find_correspondences, gt_transform, pcd2xyz

# (lo_m, hi_m, hi_inclusive)
DIST_BINS = [
    (2.0,  6.0,  False),
    (6.0,  10.0, False),
    (10.0, 12.0, True),
]
BIN_LABELS = ['[2-6m)', '[6-10m)', '[10-12m]']

def in_bin(dist: float, b: int) -> bool:
    lo, hi, inclusive = DIST_BINS[b]
    return lo <= dist <= hi if inclusive else lo <= dist < hi

def build_or_load_pair_cache(scan_files, poses, Tr, cache_path: str,
                              max_pairs: int, seed: int,
                              max_dist_idx: int = 60,
                              src_step: int = 5) -> list:
    """
    Return list of 3 sublists, one per bin, each containing
    (src_idx, tgt_idx, gt_dist_m) tuples (up to max_pairs entries).

    If cache_path already exists it is loaded directly; otherwise the
    function scans all (src, src+dist_idx) pairs for dist_idx in
    [1, max_dist_idx] with src stepping by src_step.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        bins = [[(p[0], p[1], p[2]) for p in raw['bins'][str(b)]]
                for b in range(3)]
        logging.info('Loaded pair cache %s  bins=%s',
                     cache_path, [len(b) for b in bins])
        return bins

    logging.info('Building pair distance cache (max_dist_idx=%d, src_step=%d) ...',
                 max_dist_idx, src_step)
    candidates = [[] for _ in range(3)]
    n = len(scan_files)

    for dist_idx in range(1, min(max_dist_idx + 1, n)):
        for src_idx in range(0, n - dist_idx, src_step):
            tgt_idx = src_idx + dist_idx
            T_gt = gt_transform(poses, Tr, src_idx, tgt_idx)
            d = float(np.linalg.norm(T_gt[:3, 3]))
            for b in range(3):
                if in_bin(d, b):
                    candidates[b].append((src_idx, tgt_idx, d))
                    break

    rng = np.random.default_rng(seed)
    bins = []
    for b in range(3):
        c = candidates[b]
        idx = rng.permutation(len(c)).tolist()
        kept = [c[i] for i in idx[:max_pairs]]
        bins.append(kept)
        logging.info('  Bin %d %s: %d candidates → %d kept',
                     b, BIN_LABELS[b], len(c), len(kept))

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({'bins': {str(b): bins[b] for b in range(3)}}, f)
    logging.info('Pair cache saved → %s', cache_path)
    return bins

def extract_correspondences(src_pcd, tgt_pcd, voxel_size: float,
                             feat_method: str):
    """
    Downsample → features → mutual-NN correspondences.

    Returns
    -------
    src_corr : (K, 3) float32
    tgt_corr : (K, 3) float32
    feat_time : float  (seconds)
    corr_time : float  (seconds)
    """
    src_ds = src_pcd.voxel_down_sample(voxel_size)
    tgt_ds = tgt_pcd.voxel_down_sample(voxel_size)

    t0 = time.time()
    if feat_method == 'FasterPFH':
        from kiss_matcher._kiss_matcher import FasterPFH
        ext = FasterPFH(normal_radius=voxel_size * 2,
                        fpfh_radius=voxel_size * 5,
                        thr_linearity=0.9)
        src_xyz, src_feats = ext.compute(np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = ext.compute(np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, M)
        tgt_xyz = tgt_xyz.T
    elif feat_method == 'FPFH_PCL':
        from kiss_matcher._kiss_matcher import FPFH
        ext = FPFH(normal_radius=voxel_size * 2,
                   fpfh_radius=voxel_size * 5)
        src_xyz, src_feats = ext.compute(np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = ext.compute(np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, M)
        tgt_xyz = tgt_xyz.T
    else:   # FPFH (Open3D)
        src_xyz   = pcd2xyz(src_ds)                 # (3, N)
        tgt_xyz   = pcd2xyz(tgt_ds)
        src_feats = extract_fpfh(src_ds, voxel_size)  # (N, 33)
        tgt_feats = extract_fpfh(tgt_ds, voxel_size)
    feat_time = time.time() - t0

    t0 = time.time()
    ci, cj = find_correspondences(src_feats, tgt_feats, mutual_filter=True)
    # src_xyz is (3, M) → src_xyz[:, ci] is (3, K) → .T gives (K, 3)
    src_corr = src_xyz[:, ci].T.astype(np.float32)
    tgt_corr = tgt_xyz[:, cj].T.astype(np.float32)
    corr_time = time.time() - t0

    return src_corr, tgt_corr, feat_time, corr_time
def _fmt(x, ndigits=4):
    """Return rounded float or empty string for NaN."""
    if isinstance(x, float) and math.isnan(x):
        return ''
    if isinstance(x, (int, float)):
        return round(float(x), ndigits)
    return x


def _mean(rows, col):
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    return round(float(np.mean(vals)), 4) if vals else ''

def run_sweep(args):
    dataset = args.dataset.upper()

    if dataset == 'KITTI':
        seq      = args.seq.zfill(2)
        scan_files, poses, Tr = load_kitti_dataset(seq)
        load_pcd = load_kitti_velodyne_pcd
    elif dataset == 'NCLT':
        seq      = args.seq
        scan_files, poses, Tr = load_nclt_dataset(seq)
        load_pcd = load_nclt_velodyne_pcd
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    cache_path = os.path.join(args.cache_dir,
                              f'pair_cache_{dataset.lower()}_{seq}.json')
    all_bins = build_or_load_pair_cache(
        scan_files, poses, Tr, cache_path,
        max_pairs=args.max_pairs,
        seed=args.seed,
        max_dist_idx=args.max_dist_idx,
        src_step=args.dist_step,
    )

    pairs = all_bins[args.dist_bin]
    if not pairs:
        logging.warning('No pairs for bin %d %s — skipping.',
                        args.dist_bin, BIN_LABELS[args.dist_bin])
        return

    logging.info('Bin %d %s | %d pairs | tcmp=%.4f  dcmp=%.2f  inlier_thresh=%.4f m',
                 args.dist_bin, BIN_LABELS[args.dist_bin], len(pairs),
                 args.tcmp, args.dcmp, args.dcmp * args.voxel_size)

    os.makedirs(args.out_dir, exist_ok=True)
    tcmp_tag = f'{args.tcmp}'.replace('.', 'p')   # 0.99 → 0p99
    dcmp_tag = f'{args.dcmp}'.replace('.', 'p')
    csv_name = (f'{dataset.lower()}_{seq}'
                f'_bin{args.dist_bin}'
                f'_tcmp{tcmp_tag}'
                f'_dcmp{dcmp_tag}.csv')
    csv_path = os.path.join(args.out_dir, csv_name)

    csv_fields = [
        'pair_id', 'src_idx', 'tgt_idx', 'gt_dist_m',
        'tcmp', 'dcmp', 'dist_bin',
        # registration result
        'success', 'RE_deg', 'TE_m', 'time_ms',
        # graph-state metrics
        'n_edges', 'graph_density',
        'n_cliques_total', 'n_cliques_selected',
        'mean_deg_inlier', 'mean_deg_outlier', 'sep',
        'f_pure', 'r_star',
    ]

    import mac_solver as _mac

    rows      = []
    n_success = 0
    inlier_thresh = args.dcmp * args.voxel_size

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()

        for pair_id, (src_idx, tgt_idx, gt_dist) in enumerate(
                tqdm(pairs,
                     desc=f'bin{args.dist_bin} tcmp={args.tcmp} dcmp={args.dcmp}')):
            src_idx = int(src_idx)
            tgt_idx = int(tgt_idx)

            src_pcd = load_pcd(scan_files[src_idx])
            tgt_pcd = load_pcd(scan_files[tgt_idx])
            T_gt    = gt_transform(poses, Tr, src_idx, tgt_idx)
            R_gt    = T_gt[:3, :3]
            t_gt    = T_gt[:3,  3]

            # Feature extraction + correspondences
            t_wall = time.time()
            src_corr, tgt_corr, _, _ = extract_correspondences(
                src_pcd, tgt_pcd, args.voxel_size, args.feat)

            # GT inlier mask  ||R_gt @ p_src + t_gt − p_tgt|| < 0.3 m
            residuals = np.linalg.norm(
                src_corr @ R_gt.T + t_gt - tgt_corr, axis=1)
            is_inlier = (residuals < 0.3)

            # MAC verbose solve
            res = _mac.mac_solve_verbose(
                src_corr, tgt_corr,
                inlier_thresh=float(inlier_thresh),
                cmp_thresh=float(args.tcmp),
                min_clique_sz=int(args.min_clique_sz),
                is_inlier=is_inlier,
            )
            time_ms = (time.time() - t_wall) * 1e3

            T_pred = res['transform']
            R_pred = T_pred[:3, :3]
            t_pred = T_pred[:3,  3]

            cos_a = float(np.clip((np.trace(R_pred.T @ R_gt) - 1.0) / 2.0,
                                  -1.0, 1.0))
            Re = float(np.degrees(np.arccos(cos_a)))
            Te = float(np.linalg.norm(t_pred - t_gt))
            ok = int(Re < args.re_thre and Te < args.te_thre)
            n_success += ok

            row = {
                'pair_id':            pair_id,
                'src_idx':            src_idx,
                'tgt_idx':            tgt_idx,
                'gt_dist_m':          round(float(gt_dist), 4),
                'tcmp':               args.tcmp,
                'dcmp':               args.dcmp,
                'dist_bin':           args.dist_bin,
                'success':            ok,
                'RE_deg':             round(Re, 4),
                'TE_m':               round(Te, 4),
                'time_ms':            round(time_ms, 2),
                'n_edges':            res['n_edges'],
                'graph_density':      _fmt(res['graph_density'], 6),
                'n_cliques_total':    res['n_cliques_total'],
                'n_cliques_selected': res['n_cliques_selected'],
                'mean_deg_inlier':    _fmt(res['mean_deg_inlier']),
                'mean_deg_outlier':   _fmt(res['mean_deg_outlier']),
                'sep':                _fmt(res['sep']),
                'f_pure':             _fmt(res['f_pure']),
                'r_star':             res['r_star'],
            }
            writer.writerow(row)
            rows.append(row)

        succ = [r for r in rows if r['success'] == 1]
        sr   = n_success / len(rows) * 100.0 if rows else 0.0

        writer.writerow({
            'pair_id':            'SUMMARY',
            'src_idx':            '',
            'tgt_idx':            f'{n_success}/{len(rows)}',
            'gt_dist_m':          _mean(rows, 'gt_dist_m'),
            'tcmp':               args.tcmp,
            'dcmp':               args.dcmp,
            'dist_bin':           args.dist_bin,
            'success':            round(sr, 2),
            'RE_deg':             _mean(succ, 'RE_deg'),
            'TE_m':               _mean(succ, 'TE_m'),
            'time_ms':            _mean(rows, 'time_ms'),
            'n_edges':            _mean(rows, 'n_edges'),
            'graph_density':      _mean(rows, 'graph_density'),
            'n_cliques_total':    _mean(rows, 'n_cliques_total'),
            'n_cliques_selected': _mean(rows, 'n_cliques_selected'),
            'mean_deg_inlier':    _mean(rows, 'mean_deg_inlier'),
            'mean_deg_outlier':   _mean(rows, 'mean_deg_outlier'),
            'sep':                _mean(rows, 'sep'),
            'f_pure':             _mean(rows, 'f_pure'),
            'r_star':             _mean(rows, 'r_star'),
        })

    logging.info('SR=%.1f%%  (%d/%d)  → %s',
                 sr, n_success, len(rows), csv_path)
    return rows

if __name__ == '__main__':
    # pass-1: grab --config only
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', required=True)
    pre_args, _ = pre.parse_known_args()
    with open(pre_args.config) as f:
        cfg = json.load(f)

    parser = argparse.ArgumentParser(
        description='MAC parameter sweep (one tcmp × dcmp × dist_bin config)')
    parser.add_argument('--config',       required=True)
    parser.add_argument('--dataset',      default=cfg.get('dataset', 'KITTI'))
    parser.add_argument('--seq',          default=str(cfg.get('seq', '')))
    parser.add_argument('--feat',         default=cfg.get('feat', 'FPFH'))
    parser.add_argument('--voxel_size',   type=float, default=cfg.get('voxel_size', 0.5))
    parser.add_argument('--re_thre',      type=float, default=cfg.get('re_thre',   10.0))
    parser.add_argument('--te_thre',      type=float, default=cfg.get('te_thre',    2.0))
    parser.add_argument('--seed',         type=int,   default=cfg.get('seed',        42))
    # sweep-specific
    parser.add_argument('--tcmp',         type=float, required=True,
                        help='MAC cmp_thresh')
    parser.add_argument('--dcmp',         type=float, required=True,
                        help='inlier_thresh multiplier: inlier_thresh = dcmp * voxel_size')
    parser.add_argument('--dist_bin',     type=int,   required=True, choices=[0, 1, 2],
                        help='0=[2-6m)  1=[6-10m)  2=[10-12m]')
    parser.add_argument('--out_dir',      default='results/sweep_mac')
    parser.add_argument('--cache_dir',    default='results/sweep_mac/cache')
    parser.add_argument('--max_pairs',    type=int, default=200)
    parser.add_argument('--max_dist_idx', type=int, default=60)
    parser.add_argument('--dist_step',    type=int, default=5,
                        help='Stride over src scan indices when building pair cache')
    args = parser.parse_args()

    # min_clique_sz comes from the mac block in the JSON config
    args.min_clique_sz = cfg.get('mac', {}).get('min_clique_sz', 3)

    os.makedirs('logs', exist_ok=True)
    tag = (f'{args.dataset.lower()}_{args.seq}'
           f'_sweep_mac_bin{args.dist_bin}'
           f'_tcmp{args.tcmp}_dcmp{args.dcmp}')
    logging.basicConfig(
        level=logging.INFO,
        format='',
        handlers=[
            logging.FileHandler(f'logs/{tag}.log', mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run_sweep(args)
