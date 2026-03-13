import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
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