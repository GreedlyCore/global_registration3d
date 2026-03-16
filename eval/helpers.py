import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree


def _find_correspondences_pybind(feats0, feats1, mutual_filter=True):
    """ pybindings correspondence search """
    from kiss_matcher import find_correspondences as _kiss_find_correspondences
    
    try:
        idx0, idx1 = _kiss_find_correspondences(
            np.ascontiguousarray(feats0, dtype=np.float32),
            np.ascontiguousarray(feats1, dtype=np.float32),
            bool(mutual_filter),
            False,
            0,
        )
        return np.asarray(idx0, dtype=np.int64), np.asarray(idx1, dtype=np.int64)
    except Exception:
        return None


def gt_transform(poses, Tr, src_idx, tgt_idx):
    """Compute ground-truth relative transform src -> tgt."""
    Tr_inv = np.linalg.inv(Tr)
    return Tr_inv @ np.linalg.inv(poses[tgt_idx]) @ poses[src_idx] @ Tr


def rotation_angle_deg(R):
    """Return rotation angle (degrees) from a 3x3 rotation matrix."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def resolve_feature_cfg(cfg, feat_name):
    """Resolve descriptor config with support for shared + per-feature blocks.

    Supported layouts:
      1) Top-level per-feature blocks (e.g. cfg['FPFH'] = {...})
      2) Nested descriptor block:
         cfg['descriptor'] = {
             'common': {...},
             'FPFH': {...},
             'FPFH_PCL': {...},
             'FasterPFH': {...}
         }
      3) Flat descriptor block used as common defaults:
         cfg['descriptor'] = { ... descriptor keys ... }
    """
    merged = {}
    known_keys = {
        'normal_radius', 'rnormal',
        'fpfh_radius', 'rFPFH',
        'shot_radius', 'rSHOT',
        'max_nn_normal', 'max_nn_feature',
        'shot_threads',
        'thr_linearity',
    }

    descriptor_block = cfg.get('descriptor', {})
    if isinstance(descriptor_block, dict):
        if any(k in descriptor_block for k in known_keys):
            merged.update({k: descriptor_block[k] for k in known_keys if k in descriptor_block})

        common_block = descriptor_block.get('common', {})
        if isinstance(common_block, dict):
            merged.update(common_block)

        per_feat = descriptor_block.get(feat_name, {})
        if isinstance(per_feat, dict):
            merged.update(per_feat)

    top_level_feat_block = cfg.get(feat_name, {})
    if isinstance(top_level_feat_block, dict):
        merged.update(top_level_feat_block)

    # Backward-compatible convenience:
    # let top-level "FPFH" drive KISS descriptor bindings as well.
    if feat_name in ('FPFH_PCL', 'FasterPFH', 'SHOT_PCL') and not top_level_feat_block:
        fallback_fpfh = cfg.get('FPFH', {})
        if isinstance(fallback_fpfh, dict):
            merged.update(fallback_fpfh)

    return merged


def resolve_descriptor_params(voxel_size, cfg=None):
    """Resolve descriptor hyper-parameters with backward-compatible aliases.

    Supported keys:
        - normal_radius or rnormal
        - fpfh_radius or rFPFH
        - shot_radius or rSHOT
        - max_nn_normal (Open3D FPFH only)
        - max_nn_feature (Open3D FPFH only)
        - shot_threads (SHOT_PCL only, 0 means PCL default)
        - thr_linearity (FasterPFH only)
    """
    if cfg is None:
        cfg = {}

    base_voxel = float(voxel_size)
    normal_radius = float(cfg.get('normal_radius', cfg.get('rnormal', base_voxel * 2.0)))
    fpfh_radius = float(cfg.get('fpfh_radius', cfg.get('rFPFH', base_voxel * 5.0)))
    shot_radius = float(cfg.get('shot_radius', cfg.get('rSHOT', fpfh_radius)))

    return {
        'normal_radius': normal_radius,
        'fpfh_radius': fpfh_radius,
        'shot_radius': shot_radius,
        'max_nn_normal': int(cfg.get('max_nn_normal', 30)),
        'max_nn_feature': int(cfg.get('max_nn_feature', 100)),
        'shot_threads': int(cfg.get('shot_threads', 0)),
        'thr_linearity': float(cfg.get('thr_linearity', 0.9)),
    }

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
    # Keep Open3D FPFH path fixed and unchanged.
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
#   feat1tree = cKDTree(feat1, compact_nodes=False, balanced_tree=False)
  feat1tree = cKDTree(feat1, compact_nodes=False, balanced_tree=False)
  dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
    # pybind_match = _find_correspondences_pybind(feats0, feats1, mutual_filter=mutual_filter)
    # if pybind_match is not None:
        # return pybind_match

    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1

def get_teaser_solver(noise_bound, cfg=None):
    """Build a TEASER++ solver.

    Args:
        noise_bound : noise bound (usually voxel_size)
        cfg         : optional dict of overrides from the JSON "teaser" block.
                      Supported keys and their defaults:
                        cbar2                      (1.0)
                        estimate_scaling           (False)
                        inlier_selection_mode      ("PMC_EXACT")
                        rotation_tim_graph         ("CHAIN")
                        rotation_estimation_algorithm ("GNC_TLS")
                        rotation_gnc_factor        (1.4)
                        rotation_max_iterations    (10000)
                        rotation_cost_threshold    (1e-16)
    """
    import teaserpp_python
    if cfg is None:
        cfg = {}

    _INLIER_MODE = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE
    _GRAPH_FORM  = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION
    _ROT_ALG     = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM

    inlier_mode_map = {
        'PMC_EXACT': _INLIER_MODE.PMC_EXACT,
        'PMC_HEU':   _INLIER_MODE.PMC_HEU,
        'KCORE_HEU': _INLIER_MODE.KCORE_HEU,
        'NONE':      _INLIER_MODE.NONE,
    }
    graph_form_map = {
        'CHAIN':    _GRAPH_FORM.CHAIN,
        'COMPLETE': _GRAPH_FORM.COMPLETE,
    }
    rot_alg_map = {
        'GNC_TLS': _ROT_ALG.GNC_TLS,
        'FGR':     _ROT_ALG.FGR,
    }

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2         = cfg.get('cbar2', 1.0)
    solver_params.noise_bound   = noise_bound
    solver_params.estimate_scaling = cfg.get('estimate_scaling', False)
    solver_params.inlier_selection_mode = inlier_mode_map[
        cfg.get('inlier_selection_mode', 'PMC_EXACT')]
    solver_params.rotation_tim_graph = graph_form_map[
        cfg.get('rotation_tim_graph', 'CHAIN')]
    solver_params.rotation_estimation_algorithm = rot_alg_map[
        cfg.get('rotation_estimation_algorithm', 'GNC_TLS')]
    solver_params.rotation_gnc_factor      = cfg.get('rotation_gnc_factor', 1.4)
    solver_params.rotation_max_iterations  = cfg.get('rotation_max_iterations', 10000)
    solver_params.rotation_cost_threshold  = cfg.get('rotation_cost_threshold', 1e-16)

    return teaserpp_python.RobustRegistrationSolver(solver_params)


def get_mac_solver_params(noise_bound, cfg=None):
    """Return kwargs dict for mac_solver.mac_solve from config.

    Args:
        noise_bound : fallback for inlier_thresh when cfg['inlier_thresh'] is null/None
        cfg         : optional dict of overrides from the JSON "mac" block.
                      Supported keys and their defaults:
                        inlier_thresh  (null  -> noise_bound)
                        cmp_thresh     (0.99)
                        min_clique_sz  (3)
    """
    if cfg is None:
        cfg = {}
    inlier_thresh = cfg.get('inlier_thresh') or noise_bound
    return {
        'inlier_thresh': float(inlier_thresh),
        'cmp_thresh':    float(cfg.get('cmp_thresh',    0.99)),
        'min_clique_sz': int(  cfg.get('min_clique_sz', 3)),
    }


def get_quatro_solver_params(noise_bound, cfg=None):
    """Return kwargs dict for quatro_solver.quatro_solve from config.

    Args:
        noise_bound : fallback for noise_bound when cfg['noise_bound'] is null/None
        cfg         : optional dict of overrides from the JSON "quatro" block.
                      Supported keys and their defaults:
                        noise_bound              (null -> noise_bound)
                        cbar2                    (1.0)
                        estimate_scaling         (False)
                        rotation_gnc_factor      (1.4)
                        rotation_max_iterations  (100)
                        rotation_cost_threshold  (1e-6)
                        inlier_selection_mode    (1)  # 0=PMC_EXACT, 1=PMC_HEU, 2=KCORE_HEU, 3=NONE
                        rotation_tim_graph       (0)  # 0=CHAIN, 1=COMPLETE
    """
    if cfg is None:
        cfg = {}
    quattro_noise_bound = cfg.get('noise_bound') or noise_bound
    return {
        'noise_bound': float(quattro_noise_bound),
        'cbar2': float(cfg.get('cbar2', 1.0)),
        'estimate_scaling': bool(cfg.get('estimate_scaling', False)),
        'rotation_gnc_factor': float(cfg.get('rotation_gnc_factor', 1.4)),
        'rotation_max_iterations': int(cfg.get('rotation_max_iterations', 100)),
        'rotation_cost_threshold': float(cfg.get('rotation_cost_threshold', 1e-6)),
        'inlier_selection_mode': int(cfg.get('inlier_selection_mode', 1)),
        'rotation_tim_graph': int(cfg.get('rotation_tim_graph', 0)),
    }


def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 