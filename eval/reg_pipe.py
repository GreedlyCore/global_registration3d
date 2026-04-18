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
    get_gmor_solver_params,
    get_teaser_solver,
    get_mac_solver_params,
    get_macpp_solver_params,
    get_quatro_solver_params,
    get_trde_solver_params,
    resolve_descriptor_params,
    Rt2T,
    pcd2xyz,
)
from helpers_graph import (
    graph_stats_kiss,
    graph_stats_mac,
    graph_stats_quatro,
    graph_stats_quatro_result,
    graph_stats_teaser,
    graph_stats_teaser_solver,
)


# TODO: remove
def _identity_result(timings, n_corr_init=0):
    """Return a safe fallback result when feature/correspondence data is unusable."""
    timings.setdefault('registration', 0.0)
    return (
        np.eye(4, dtype=np.float64),
        timings,
        {
            'n_corr_init': int(n_corr_init),
            'n_inliers': 0,
            'n_outliers': int(n_corr_init),
        },
    )

# TODO: remove
def _sanitize_feature_outputs(xyz, feats):
    """Drop non-finite descriptor rows and keep xyz/feature arrays aligned."""
    xyz = np.asarray(xyz, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float32)

    if xyz.ndim != 2 or feats.ndim != 2:
        return np.empty((3, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float32)

    if xyz.shape[0] != 3 and xyz.shape[1] == 3:
        xyz = xyz.T

    if xyz.shape[0] != 3:
        return np.empty((3, 0), dtype=np.float64), np.empty((0, feats.shape[1]), dtype=np.float32)

    n = min(xyz.shape[1], feats.shape[0])
    if n <= 0:
        return np.empty((3, 0), dtype=np.float64), np.empty((0, feats.shape[1]), dtype=np.float32)

    xyz = xyz[:, :n]
    feats = feats[:n, :]

    finite_xyz = np.all(np.isfinite(xyz.T), axis=1)
    finite_feats = np.all(np.isfinite(feats), axis=1)
    mask = finite_xyz & finite_feats

    if not np.any(mask):
        return np.empty((3, 0), dtype=np.float64), np.empty((0, feats.shape[1]), dtype=np.float32)

    return xyz[:, mask], feats[mask, :]

def _downsample_tbb(pcd, voxel_size):
    """Downsample an open3d PointCloud using kiss_matcher TBB VoxelgridSampling."""
    from kiss_matcher._kiss_matcher import voxelgrid_sampling
    pts_in = np.asarray(pcd.points).astype(np.float32)
    pts_out = voxelgrid_sampling(pts_in, float(voxel_size))   # (M, 3) float32
    ds = o3d.geometry.PointCloud()
    ds.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
    return ds


def _import_macpp_solver():
    """Import macpp_solver, trying local build/eval locations first."""
    try:
        import macpp_solver  # type: ignore
        return macpp_solver
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate_py_dirs = [
        os.path.join(repo_root, 'eval'),
        os.path.join(repo_root, 'MAC-PLUS-PLUS', 'src', 'build_pybind'),
    ]

    for d in candidate_py_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import macpp_solver  # type: ignore
    return macpp_solver


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


def _import_gmor_trde_solver():
    """Import gmor_trde_solver, trying local build/eval locations first."""
    try:
        import gmor_trde_solver  # type: ignore
        return gmor_trde_solver
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate_py_dirs = [
        os.path.join(repo_root, 'eval'),
        os.path.join(repo_root, 'GMOR', 'build_pybind'),
    ]

    for d in candidate_py_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import gmor_trde_solver  # type: ignore
    return gmor_trde_solver


def _import_std_solver():
    """Import std_solver from the local eval path or build output."""
    try:
        import std_solver  # type: ignore
        return std_solver
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate_py_dirs = [
        os.path.join(repo_root, 'eval'),
        os.path.join(repo_root, 'STD_REG', 'build_pybind'),
    ]

    for d in candidate_py_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    import std_solver  # type: ignore
    return std_solver


def run_registration(src_pcd, tgt_pcd, voxel_size=0.5,
                     reg_method='teaser', feat_method='FPFH', corr_method='nn',
                     downsample_method='o3d',
                     teaser_cfg=None, mac_cfg=None, macpp_cfg=None, quatro_cfg=None,
                     gmor_cfg=None, trde_cfg=None,
                     feat_cfg=None, dataset_name=None):
    """
    Feature-based point cloud registration pipeline.

    Args:
        src_pcd           : open3d.geometry.PointCloud  (source)
        tgt_pcd           : open3d.geometry.PointCloud  (target)
        voxel_size        : voxel size for downsampling and feature radius
        reg_method        : 'teaser' | 'mac' | 'macpp' | 'quatro' | 'kiss' | 'gmor' | 'trde'
        feat_method       : 'FPFH' (o3d) | 'FPFH_PCL' / FasterPFH' (kiss_matcher PCL binding) | 'SHOT_PCL'
        corr_method       : 'nn'  (mutual nearest-neighbor in feature space)
        downsample_method : 'o3d' (open3d voxel_down_sample) | 'tbb_vector' (kiss_matcher TBB)
        dataset_name      : dataset tag forwarded to method-specific solvers (used by macpp)

    Returns:
        T_pred     : (4, 4) numpy float64 — predicted transform src -> tgt
        timings    : dict  keys 'feature', 'correspondence', 'registration' (seconds)
        corr_stats : dict  keys 'n_corr_init', 'n_inliers', 'n_outliers'
    """
    timings = {}
    
    # fine for now ///
    if feat_method == 'STD' or reg_method == 'STD':
        std_solver = _import_std_solver()

        src_np = np.asarray(src_pcd.points, dtype=np.float32)
        tgt_np = np.asarray(tgt_pcd.points, dtype=np.float32)

        cfg = std_solver.ConfigSetting()
        cfg.ds_size_ = float(voxel_size)
        cfg.voxel_size_ = float(voxel_size)

        matcher = std_solver.STDescManager(cfg)
        result = matcher.match_pairwise(src_np, tgt_np)

        rot = np.asarray(result['transform_rotation'], dtype=np.float64)
        trans = np.asarray(result['transform_translation'], dtype=np.float64)
        if rot.shape != (3, 3):
            rot = np.eye(3, dtype=np.float64)
        if trans.shape != (3,):
            trans = np.zeros(3, dtype=np.float64)
        T_pred = Rt2T(rot, trans)

        timings = {
            'downsample': float(result['timings_ms'].get('downsample', 0.0)) / 1000.0,
            'feature': float(result['timings_ms'].get('feature', 0.0)) / 1000.0,
            'correspondence': float(result['timings_ms'].get('candidate', 0.0)) / 1000.0,
            'registration': float(result['timings_ms'].get('registration', 0.0)) / 1000.0,
        }

        corr_count = len(result.get('correspondences', []))
        corr_stats = {
            'n_corr_init': corr_count,
            'n_inliers': corr_count,
            'n_outliers': 0,
        }
        return T_pred, timings, corr_stats

    # Full self-contained pipeline
    if reg_method == 'kiss':
        import kiss_matcher
        src_np = np.asarray(src_pcd.points).astype(np.float64)
        tgt_np = np.asarray(tgt_pcd.points).astype(np.float64)

        def _safe_kiss_time(getter_name, fallback=0.0):
            getter = getattr(matcher, getter_name, None)
            if getter is None:
                return float(fallback)
            try:
                v = float(getter())
            except Exception:
                return float(fallback)
            return v if v >= 0.0 else float(fallback)

        t0 = time.time()
        params = kiss_matcher.KISSMatcherConfig(voxel_size)
        matcher = kiss_matcher.KISSMatcher(params)
        result = matcher.estimate(src_np, tgt_np)

        wall_t = time.time() - t0
        timings['downsample'] = _safe_kiss_time('get_processing_time', fallback=0.0)
        timings['feature'] = _safe_kiss_time('get_extraction_time', fallback=0.0)
        # KISS internally has:
        # 1) descriptor matching (NN search),
        # 2) ROBIN outlier pruning,
        # 3) final robust solve.
        #
        # For parity with other methods in this repo, we treat (2)+(3) as
        # "registration" and keep only (1) as "correspondence".
        # This avoids misleading near-zero reg_time_s values for KISS.
        timings['correspondence'] = _safe_kiss_time('get_matching_time', fallback=0.0)
        timings['registration'] = (
            _safe_kiss_time('get_rejection_time', fallback=0.0)
            + _safe_kiss_time('get_solver_time', fallback=0.0)
        )

        # If timing getters are unavailable (older wheel), preserve prior behavior.
        if (
            timings['downsample'] == 0.0
            and timings['feature'] == 0.0
            and timings['correspondence'] == 0.0
            and timings['registration'] == 0.0
        ):
            timings['registration'] = wall_t

        T_pred = Rt2T(np.array(result.rotation), np.array(result.translation))
        corr_stats = graph_stats_kiss(matcher)
        return T_pred, timings, corr_stats

    # Downsample
    t0 = time.time()
    if downsample_method == 'tbb_vector': # show be superior faster comparing to o3d 
        src_ds = _downsample_tbb(src_pcd, voxel_size)
        tgt_ds = _downsample_tbb(tgt_pcd, voxel_size)
    else:  # 'o3d'
        src_ds = src_pcd.voxel_down_sample(voxel_size)
        tgt_ds = tgt_pcd.voxel_down_sample(voxel_size)
    timings['downsample'] = time.time() - t0

    # Feature extraction
    t0 = time.time()
    feat_params = resolve_descriptor_params(voxel_size, cfg=feat_cfg)
    if feat_method == 'FasterPFH':
        from kiss_matcher._kiss_matcher import FasterPFH
        extractor = FasterPFH(normal_radius=feat_params['normal_radius'],
                              fpfh_radius=feat_params['fpfh_radius'],
                              thr_linearity=feat_params['thr_linearity'])
        src_xyz, src_feats = extractor.compute(
            np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = extractor.compute(
            np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, M)
        tgt_xyz = tgt_xyz.T   # (3, M)
    elif feat_method == 'FPFH_PCL':  # FPFH via kiss_matcher pybinding 
        from kiss_matcher._kiss_matcher import FPFH
        extractor = FPFH(normal_radius=feat_params['normal_radius'],
                         fpfh_radius=feat_params['fpfh_radius'])
        src_xyz, src_feats = extractor.compute(
            np.asarray(src_ds.points).astype(np.float32))
        tgt_xyz, tgt_feats = extractor.compute(
            np.asarray(tgt_ds.points).astype(np.float32))
        src_xyz = src_xyz.T   # (3, N)
        tgt_xyz = tgt_xyz.T   # (3, N)
    elif feat_method == 'SHOT_PCL':
        from kiss_matcher._kiss_matcher import SHOT
        extractor = SHOT(
            normal_radius=feat_params['normal_radius'],
            shot_radius=feat_params['shot_radius'],
            n_threads=feat_params['shot_threads'],
        )
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

    src_xyz, src_feats = _sanitize_feature_outputs(src_xyz, src_feats)
    tgt_xyz, tgt_feats = _sanitize_feature_outputs(tgt_xyz, tgt_feats)
    timings['feature'] = time.time() - t0

    if src_feats.shape[0] == 0 or tgt_feats.shape[0] == 0:
        return _identity_result(timings, n_corr_init=0)

    # Correspondences
    t0 = time.time()
    # corr_method == 'nn'
    corrs_src, corrs_tgt = find_correspondences(src_feats, tgt_feats, mutual_filter=True)
    src_corr = src_xyz[:, corrs_src]   # (3, K)
    tgt_corr = tgt_xyz[:, corrs_tgt]   # (3, K)

    if src_corr.size == 0 or tgt_corr.size == 0:
        timings['correspondence'] = time.time() - t0
        return _identity_result(timings, n_corr_init=0)

    finite_corr_mask = np.all(np.isfinite(src_corr.T), axis=1) & np.all(np.isfinite(tgt_corr.T), axis=1)
    src_corr = src_corr[:, finite_corr_mask]
    tgt_corr = tgt_corr[:, finite_corr_mask]
    timings['correspondence'] = time.time() - t0

    min_corr_required = 3
    if src_corr.shape[1] < min_corr_required:
        return _identity_result(timings, n_corr_init=src_corr.shape[1])

    # Registration
    t0 = time.time()
    if reg_method == 'teaser':
        solver = get_teaser_solver(noise_bound=voxel_size, cfg=teaser_cfg)
        solver.solve(src_corr, tgt_corr)
        sol = solver.getSolution()
        T_pred = Rt2T(sol.rotation, sol.translation)
        corr_stats = graph_stats_teaser_solver(solver, src_corr.shape[1])
        if corr_stats is None:
            corr_stats = graph_stats_teaser(
                src_corr.T, tgt_corr.T, T_pred, inlier_thresh=voxel_size)
    elif reg_method == 'mac':
        import mac_solver
        mac_kwargs = get_mac_solver_params(noise_bound=voxel_size, cfg=mac_cfg)
        # src_corr / tgt_corr are (3, K) — mac_solve expects (K, 3)
        T_pred = mac_solver.mac_solve(
            src_corr.T.astype(np.float32),
            tgt_corr.T.astype(np.float32),
            **mac_kwargs,
        )
        corr_stats = graph_stats_mac(
            src_corr.T, tgt_corr.T, T_pred, inlier_thresh=mac_kwargs['inlier_thresh'])
    elif reg_method == 'macpp':
        macpp_solver = _import_macpp_solver()

        if feat_method in ('FPFH', 'FPFH_PCL', 'FasterPFH', 'SHOT_PCL'):
            descriptor_for_macpp = 'fpfh'
        else:
            descriptor_for_macpp = feat_method.lower()

        ds = (dataset_name or 'KITTI').upper()
        if ds == 'KITTI':
            dataset_for_macpp = 'KITTI'
        elif ds == '3DMATCH':
            dataset_for_macpp = '3dmatch'
        elif ds == '3DLOMATCH':
            dataset_for_macpp = '3dlomatch'
        elif ds == 'U3M':
            dataset_for_macpp = 'U3M'
        else:
            dataset_for_macpp = 'KITTI'

        macpp_kwargs = get_macpp_solver_params(
            noise_bound=voxel_size,
            cfg=macpp_cfg,
            dataset_name=dataset_for_macpp,
            descriptor=descriptor_for_macpp,
        )
        T_pred = np.asarray(
            macpp_solver.macpp_solve(
                src_corr.T.astype(np.float32),
                tgt_corr.T.astype(np.float32),
                **macpp_kwargs,
            ),
            dtype=np.float64,
        )
        corr_stats = graph_stats_mac(
            src_corr.T, tgt_corr.T, T_pred, inlier_thresh=macpp_kwargs['inlier_thresh'])
    elif reg_method == 'quatro':
        quatro_solver = _import_quatro_solver()
        quatro_kwargs = get_quatro_solver_params(noise_bound=voxel_size, cfg=quatro_cfg)
        src_corr64 = src_corr.T.astype(np.float64)
        tgt_corr64 = tgt_corr.T.astype(np.float64)
        solve_with_stats = getattr(quatro_solver, 'quatro_solve_with_stats', None)
        if solve_with_stats is not None:
            result = solve_with_stats(
                src_corr64,
                tgt_corr64,
                **quatro_kwargs,
            )
            T_pred = np.asarray(result['transform'], dtype=np.float64)
            corr_stats = graph_stats_quatro_result(result)
        else:
            # quatro_solve expects Kx3 float64 arrays.
            T_pred = np.asarray(
                quatro_solver.quatro_solve(
                    src_corr64,
                    tgt_corr64,
                    **quatro_kwargs,
                ),
                dtype=np.float64,
            )
            corr_stats = graph_stats_quatro(
                src_corr.T, tgt_corr.T, T_pred, inlier_thresh=quatro_kwargs['noise_bound'])
    elif reg_method == 'gmor':
        gmor_trde_solver = _import_gmor_trde_solver()
        gmor_kwargs = get_gmor_solver_params(noise_bound=voxel_size, cfg=gmor_cfg)
        T_pred = np.asarray(
            gmor_trde_solver.gmor_solve(
                src_corr.T.astype(np.float32),
                tgt_corr.T.astype(np.float32),
                **gmor_kwargs,
            ),
            dtype=np.float64,
        )
        corr_stats = graph_stats_mac(
            src_corr.T, tgt_corr.T, T_pred, inlier_thresh=gmor_kwargs['noise_bound'])
    elif reg_method == 'trde':
        gmor_trde_solver = _import_gmor_trde_solver()
        trde_kwargs = get_trde_solver_params(noise_bound=voxel_size, cfg=trde_cfg)
        T_pred = np.asarray(
            gmor_trde_solver.trde_solve(
                src_corr.T.astype(np.float32),
                tgt_corr.T.astype(np.float32),
                **trde_kwargs,
            ),
            dtype=np.float64,
        )
        corr_stats = graph_stats_mac(
            src_corr.T, tgt_corr.T, T_pred, inlier_thresh=trde_kwargs['noise_bound'])
    else:
        raise ValueError(f"Unknown reg_method: {reg_method}")
    timings['registration'] = time.time() - t0

    return T_pred, timings, corr_stats
