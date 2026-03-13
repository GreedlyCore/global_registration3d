"""
Registration pipeline: feature extraction -> correspondences -> solver.
"""
import os
import sys
import time
import ctypes
import numpy as np
import open3d as o3d
from helpers import (
    extract_fpfh,
    find_correspondences,
    get_teaser_solver,
    get_mac_solver_params,
    get_quatro_solver_params,
    Rt2T,
    pcd2xyz,
)


def _downsample_tbb(pcd, voxel_size):
    """Downsample an open3d PointCloud using kiss_matcher TBB VoxelgridSampling."""
    from kiss_matcher._kiss_matcher import voxelgrid_sampling
    pts_in = np.asarray(pcd.points).astype(np.float32)
    pts_out = voxelgrid_sampling(pts_in, float(voxel_size))   # (M, 3) float32
    ds = o3d.geometry.PointCloud()
    ds.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
    return ds


def _import_quatro_solver():
    """Import quatro_solver, trying local build/install locations first."""
    try:
        import quatro_solver  # type: ignore
        return quatro_solver
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate_py_dirs = [
        os.path.join(repo_root, 'Quatro', 'install', 'quatro_ros2', 'lib'),
        os.path.join(repo_root, 'Quatro', 'build', 'quatro_ros2'),
    ]
    candidate_libs = [
        os.path.join(repo_root, 'Quatro', 'build', 'quatro_ros2', 'pmc', 'lib', 'libpmc.so'),
        os.path.join(repo_root, 'Quatro', 'install', 'quatro_ros2', 'lib', 'libquatro_core.so'),
    ]

    for d in candidate_py_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    # Preload dependent shared libraries so Python extension import can resolve symbols.
    for lib_path in candidate_libs:
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    import quatro_solver  # type: ignore
    return quatro_solver


def run_registration(src_pcd, tgt_pcd, voxel_size=0.5,
                     reg_method='teaser', feat_method='FPFH', corr_method='nn',
                     downsample_method='o3d',
                     teaser_cfg=None, mac_cfg=None, quatro_cfg=None):
    """
    Feature-based point cloud registration pipeline.

    Args:
        src_pcd           : open3d.geometry.PointCloud  (source)
        tgt_pcd           : open3d.geometry.PointCloud  (target)
        voxel_size        : voxel size for downsampling and feature radius
        reg_method        : 'teaser' | 'mac' | 'quatro' | 'kiss'
        feat_method       : 'FPFH' (open3d) | 'FPFH_PCL' (kiss_matcher PCL binding) | 'FasterPFH'
        corr_method       : 'nn'  (mutual nearest-neighbor in feature space)
        downsample_method : 'o3d' (open3d voxel_down_sample) | 'tbb_vector' (kiss_matcher TBB)

    Returns:
        T_pred  : (4, 4) numpy float64 — predicted transform src -> tgt
        timings : dict  keys 'feature', 'correspondence', 'registration' (seconds)
    """
    timings = {}

    # ------------------------------------------------------------------ #
    # KISS-Matcher: full self-contained pipeline
    # ------------------------------------------------------------------ #
    if reg_method == 'kiss':
        import kiss_matcher
        src_np = np.asarray(src_pcd.points).astype(np.float64)
        tgt_np = np.asarray(tgt_pcd.points).astype(np.float64)
        t0 = time.time()
        params = kiss_matcher.KISSMatcherConfig(voxel_size)
        matcher = kiss_matcher.KISSMatcher(params)
        result = matcher.estimate(src_np, tgt_np)
        timings['registration'] = time.time() - t0
        timings['downsample'] = 0.0
        timings['feature'] = 0.0
        timings['correspondence'] = 0.0
        T_pred = Rt2T(np.array(result.rotation), np.array(result.translation))
        return T_pred, timings

    # ------------------------------------------------------------------ #
    # Downsample
    # ------------------------------------------------------------------ #
    t0 = time.time()
    if downsample_method == 'tbb_vector': # show be superior faster comparing to o3d 
        src_ds = _downsample_tbb(src_pcd, voxel_size)
        tgt_ds = _downsample_tbb(tgt_pcd, voxel_size)
    else:  # 'o3d'
        src_ds = src_pcd.voxel_down_sample(voxel_size)
        tgt_ds = tgt_pcd.voxel_down_sample(voxel_size)
    timings['downsample'] = time.time() - t0

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
    elif feat_method == 'FPFH_PCL':  # FPFH via kiss_matcher pybinding 
        from kiss_matcher._kiss_matcher import FPFH
        extractor = FPFH(normal_radius=voxel_size * 2,
                         fpfh_radius=voxel_size * 5)
        src_xyz, src_feats = extractor.compute(
            np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = extractor.compute(
            np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, N)
        tgt_xyz = tgt_xyz.T   # (3, N)
    else:  # FPFH via open3d
        # src_xyz = pcd2xyz(src_ds)   # (3, N)
        # tgt_xyz = pcd2xyz(tgt_ds)
        # src_feats = extract_fpfh(src_ds, voxel_size)   # (N, 33)
        # tgt_feats = extract_fpfh(tgt_ds, voxel_size)
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
        import mac_solver
        mac_kwargs = get_mac_solver_params(noise_bound=voxel_size, cfg=mac_cfg)
        # src_corr / tgt_corr are (3, K) — mac_solve expects (K, 3)
        T_pred = mac_solver.mac_solve(
            src_corr.T.astype(np.float32),
            tgt_corr.T.astype(np.float32),
            **mac_kwargs,
        )
    elif reg_method == 'quatro':
        quatro_solver = _import_quatro_solver()
        quatro_kwargs = get_quatro_solver_params(noise_bound=voxel_size, cfg=quatro_cfg)
        # quatro_solve expects Kx3 float64 arrays.
        T_pred = np.asarray(
            quatro_solver.quatro_solve(
                src_corr.T.astype(np.float64),
                tgt_corr.T.astype(np.float64),
                **quatro_kwargs,
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown reg_method: {reg_method}")
    timings['registration'] = time.time() - t0

    return T_pred, timings
