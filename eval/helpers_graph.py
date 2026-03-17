"""
Unified correspondence-graph statistics across registration methods.

Core metric contract:
    n_corr_init = n_inliers + n_outliers
"""

import numpy as np


def _finalize_graph_stats(n_corr_init, n_inliers):
    """Return normalized integer graph stats with a strict invariant."""
    n_init = max(0, int(n_corr_init))
    n_inl = max(0, int(n_inliers))
    if n_inl > n_init:
        n_init = n_inl
    n_out = n_init - n_inl
    ratio = float('inf') if n_out == 0 else (n_inl / n_out)
    return {
        'n_corr_init': n_init,
        'n_inliers': n_inl,
        'n_outliers': n_out,
        'ratio': ratio,
    }


def _count_inliers_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh):
    """Count inliers from geometric residuals on correspondence arrays."""
    src = np.asarray(src_corr, dtype=np.float64)
    tgt = np.asarray(tgt_corr, dtype=np.float64)

    if src.ndim != 2 or tgt.ndim != 2 or src.shape != tgt.shape or src.shape[1] != 3:
        return 0
    if src.shape[0] == 0:
        return 0

    R = np.asarray(T_pred[:3, :3], dtype=np.float64)
    t = np.asarray(T_pred[:3, 3], dtype=np.float64)

    residuals = np.linalg.norm((src @ R.T) + t - tgt, axis=1)
    n_inliers = int(np.sum(residuals < float(inlier_thresh)))
    return n_inliers


def _stats_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh):
    """Build graph stats from raw correspondences and a predicted transform."""
    src = np.asarray(src_corr)
    n_init = int(src.shape[0]) if src.ndim == 2 else 0
    n_inliers = _count_inliers_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh)
    return _finalize_graph_stats(n_init, n_inliers)


def graph_stats_mac(src_corr, tgt_corr, T_pred, inlier_thresh):
    """MAC graph stats using residual inliers on initial correspondences."""
    return _stats_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh)


def graph_stats_teaser(src_corr, tgt_corr, T_pred, inlier_thresh):
    """TEASER graph stats using residual inliers on initial correspondences."""
    return _stats_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh)


def graph_stats_teaser_solver(solver, n_corr_init):
    """TEASER graph stats from binding-native translation inlier indices."""
    # Preferred path: binding-native standardized stats.
    if hasattr(solver, 'get_graph_stats'):
        try:
            stats = solver.get_graph_stats(int(n_corr_init))
            return _finalize_graph_stats(stats['n_corr_init'], stats['n_inliers'])
        except Exception:
            pass

    # Fallback path: count input-ordered translation inlier indices.
    if hasattr(solver, 'getInputOrderedTranslationInliers'):
        try:
            indices = solver.getInputOrderedTranslationInliers()
            return _finalize_graph_stats(n_corr_init, len(indices))
        except Exception:
            pass

    return None


def graph_stats_quatro(src_corr, tgt_corr, T_pred, inlier_thresh):
    """Quatro graph stats using residual inliers on initial correspondences."""
    return _stats_from_residual(src_corr, tgt_corr, T_pred, inlier_thresh)


def graph_stats_quatro_result(result):
    """Quatro graph stats from binding-native solve_with_stats output."""
    if result is None:
        return None
    try:
        return _finalize_graph_stats(result['n_corr_init'], result['n_inliers'])
    except Exception:
        return None


def graph_stats_kiss(matcher):
    """KISS graph stats from exposed matcher counters/APIs.

    n_corr_init comes from initial correspondences.
    n_inliers comes from final translation inliers.
    """
    # Preferred path: binding-native standardized stats.
    if hasattr(matcher, 'get_graph_stats'):
        try:
            stats = matcher.get_graph_stats()
            return _finalize_graph_stats(stats['n_corr_init'], stats['n_inliers'])
        except Exception:
            pass

    # Fallback path: compute counts from existing KISS getters.
    n_init = 0
    if hasattr(matcher, 'get_initial_correspondences'):
        try:
            n_init = len(matcher.get_initial_correspondences())
        except Exception:
            n_init = 0

    n_inl = None
    if hasattr(matcher, 'get_num_final_inliers'):
        try:
            n_inl = int(matcher.get_num_final_inliers())
        except Exception:
            n_inl = None

    if n_inl is None:
        # Fallback to final correspondences if inlier counter is unavailable.
        if hasattr(matcher, 'get_final_correspondences'):
            try:
                n_inl = len(matcher.get_final_correspondences())
            except Exception:
                n_inl = 0
        else:
            n_inl = 0

    return _finalize_graph_stats(n_init, n_inl)
