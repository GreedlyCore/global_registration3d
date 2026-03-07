"""
Registration pipeline: feature extraction -> correspondences -> solver.
"""
import time
import numpy as np
import open3d as o3d
import mac_solver
from helpers import extract_fpfh, find_correspondences, get_teaser_solver, get_mac_solver_params, Rt2T, pcd2xyz


def run_registration(src_pcd, tgt_pcd, voxel_size=0.5,
                     reg_method='teaser', feat_method='FPFH', corr_method='nn',
                     teaser_cfg=None, mac_cfg=None):
    """
    Feature-based point cloud registration pipeline.

    Args:
        src_pcd     : open3d.geometry.PointCloud  (source)
        tgt_pcd     : open3d.geometry.PointCloud  (target)
        voxel_size  : voxel size for downsampling and feature radii
        reg_method  : 'teaser' | 'mac' | ...  # TODO: add also MAC++ / KISS-Matcher supremacy?
        feat_method : 'FPFH'   | 'FasterPFH' | ... # TODO: try more features?
        corr_method : 'nn'  (mutual nearest-neighbor in feature space) | ... # TODO: try more NNs related methods?

    Returns:
        T_pred  : (4, 4) numpy float64 — predicted transform src -> tgt
        timings : dict  keys 'feature', 'correspondence', 'registration' (seconds)
    """
    timings = {}

    # ------------------------------------------------------------------ #
    # Downsample
    # ------------------------------------------------------------------ #
    src_ds = src_pcd.voxel_down_sample(voxel_size)
    tgt_ds = tgt_pcd.voxel_down_sample(voxel_size)

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    t0 = time.time()
    if feat_method == 'FasterPFH':
        from kiss_matcher._kiss_matcher import FasterPFH
        extractor = FasterPFH(normal_radius=voxel_size * 2,
                              fpfh_radius=voxel_size * 5,
                              thr_linearity=0.9)
        src_xyz, src_feats = extractor.compute(
            np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = extractor.compute(
            np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, M)
        tgt_xyz = tgt_xyz.T   # (3, M)
    else:  # FPFH via open3d
        src_xyz = pcd2xyz(src_ds)   # (3, N)
        tgt_xyz = pcd2xyz(tgt_ds)
        src_feats = extract_fpfh(src_ds, voxel_size)   # (N, 33)
        tgt_feats = extract_fpfh(tgt_ds, voxel_size)
    timings['feature'] = time.time() - t0

    # ------------------------------------------------------------------ #
    # Correspondences
    # ------------------------------------------------------------------ #
    t0 = time.time()
    # corr_method == 'nn'
    corrs_src, corrs_tgt = find_correspondences(src_feats, tgt_feats, mutual_filter=True)
    src_corr = src_xyz[:, corrs_src]   # (3, K)
    tgt_corr = tgt_xyz[:, corrs_tgt]   # (3, K)
    timings['correspondence'] = time.time() - t0

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    t0 = time.time()
    if reg_method == 'teaser':
        solver = get_teaser_solver(noise_bound=voxel_size, cfg=teaser_cfg)
        solver.solve(src_corr, tgt_corr)
        sol = solver.getSolution()
        T_pred = Rt2T(sol.rotation, sol.translation)
    elif reg_method == 'mac':
        mac_kwargs = get_mac_solver_params(noise_bound=voxel_size, cfg=mac_cfg)
        # src_corr / tgt_corr are (3, K) — mac_solve expects (K, 3)
        T_pred = mac_solver.mac_solve(
            src_corr.T.astype(np.float32),
            tgt_corr.T.astype(np.float32),
            **mac_kwargs,
        )
    else:
        raise ValueError(f"Unknown reg_method: {reg_method}")
    timings['registration'] = time.time() - t0

    return T_pred, timings
