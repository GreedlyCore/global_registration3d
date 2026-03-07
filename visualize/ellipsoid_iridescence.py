#!/usr/bin/env python
"""
GMM ellipsoid visualizer using iridescence

Each Gaussian component is drawn as a scaled/rotated sphere primitive:
  T[:3, :3] = eigenvectors @ diag(n_sigma * sqrt(eigenvalues))   <- rotation + scale
  T[:3,  3] = mean                                               <- translation
This maps the unit sphere exactly onto the n-sigma ellipsoid.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

import numpy as np
import argparse
import glob
import yaml
import matplotlib.pyplot as plt
from pyridescence import guik, glk, imgui

from utils.nclt_loader import get_nclt_sync_files, load_nclt_pointcloud
from utils.tum_loader import get_tum_scan_files, load_tum_pointcloud
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter

NCLT_BASE_PATH = os.path.expanduser('~/thesis/data')
TUM_BASE_PATH  = os.path.expanduser('~/thesis/sogmm_registration/data')

_PCL_SHADER_NAMES = ['rainbow', 'flat_red', 'flat_green', 'flat_blue', 'flat_orange']


def load_gmm_file(gmm_file):
    """Load GMM from .gmm file.

    Returns:
        dict with 'means', 'covariances', 'weights' (each as numpy arrays)
    """
    data = np.loadtxt(gmm_file, delimiter=',')
    n_components = data.shape[0]
    return {
        'means':       data[:, 0:3],
        'covariances': data[:, 3:12].reshape(n_components, 3, 3),
        'weights':     data[:, 12],
        'n_components': n_components
    }


def weights_to_viridis(weights):
    """Convert weights to Viridis RGBA colors, shape (N, 4)."""
    weights = np.array(weights)
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-10:
        normalized = np.full_like(weights, 0.5)
    else:
        normalized = (weights - w_min) / (w_max - w_min)
    return plt.cm.viridis(normalized)


def ellipsoid_transform(mean, covariance, n_sigma):
    """
    Build a 4x4 transform that maps the unit sphere to the n-sigma ellipsoid
    defined by (mean, covariance).

        T[:3, :3] = V @ diag(n_sigma * sqrt(λ))   (rotation × scale)
        T[:3,  3] = mean                            (translation)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues))
    T[:3, 3] = mean
    return T


def load_meta_yaml(gmm_dir):
    """Load meta.yaml from GMM directory. Returns empty dict if not found."""
    meta_path = os.path.join(gmm_dir, 'meta.yaml')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return yaml.safe_load(f) or {}


def apply_nclt_preprocessing(points, meta):
    """Same pipeline as create_and_save_gmm_nclt.py (mahal_distance is EM-internal)."""
    if meta.get('every_n'):
        points = every_n_filter(points, n=meta['every_n'], verbose=False)
    if meta.get('radius') is not None:
        points = radius_filter(points, 0.5, meta['radius'], verbose=False)
    if meta.get('voxel') is not None:
        points = voxel_filter(points, meta['voxel'], verbose=False)
    return points


def apply_tum_preprocessing(points, meta):
    """Same pipeline as create_and_save_gmm_tum.py (radius min=0.0, mahal_distance is EM-internal)."""
    if meta.get('every_n'):
        points = every_n_filter(points, n=meta['every_n'], verbose=False)
    if meta.get('radius') is not None:
        points = radius_filter(points, 0.0, meta['radius'], verbose=False)
    if meta.get('voxel') is not None:
        points = voxel_filter(points, meta['voxel'], verbose=False)
    return points


class IridescenceEllipsoidVisualizer:
    """Interactive GMM ellipsoid visualizer using pyridescence."""

    def __init__(self, gmm_dir, start_index=0, dataset=None, scene=None):
        self.gmm_dir = gmm_dir
        self.gmm_files = self._get_gmm_files()
        self.current_index = start_index

        # Ellipsoid rendering params
        self.n_sigma = 2
        self.sigma_options = [1, 2, 3]
        self.wire_mode = True          # wire_sphere vs solid sphere
        self.use_viridis = True
        self._prev_n_components = 0   # tracks how many drawables to clean up

        # Scene
        self.point_size = 0.03
        self.show_axes = True
        self.n_components = 0

        # Auto-play
        self.auto_play = False
        self.play_speed = 1
        self.frame_counter = 0

        # PCL state
        self.dataset = dataset
        self.meta = load_meta_yaml(gmm_dir)
        self.show_pcl = False
        self.pcl_color_idx = 0
        self.pcl_files = None

        if dataset == 'nclt' and scene:
            try:
                self.pcl_files = get_nclt_sync_files(NCLT_BASE_PATH, scene)
                self.show_pcl = True
                print(f"NCLT: {len(self.pcl_files)} scans, scene={scene}")
            except FileNotFoundError as e:
                print(f"WARNING: Could not load NCLT data: {e}")

        elif dataset == 'tum' and scene:
            try:
                self.pcl_files = get_tum_scan_files(TUM_BASE_PATH, scene)
                self.show_pcl = True
                print(f"TUM: {len(self.pcl_files)} scans, dataset={scene}")
            except FileNotFoundError as e:
                print(f"WARNING: Could not load TUM data: {e}")

        if self.pcl_files is not None and self.meta:
            print(f"Preprocessing from meta.yaml: {self.meta}")

        print(f"Found {len(self.gmm_files)} GMM files in {gmm_dir}")

    def _get_gmm_files(self):
        gmm_files = sorted(
            glob.glob(os.path.join(self.gmm_dir, "*.gmm")),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        if not gmm_files:
            raise ValueError(f"No .gmm files found in {self.gmm_dir}")
        return gmm_files

    def _get_pcl_shader(self):
        shaders = {
            'rainbow':     guik.Rainbow,
            'flat_red':    guik.FlatRed,
            'flat_green':  guik.FlatGreen,
            'flat_blue':   guik.FlatBlue,
            'flat_orange': guik.FlatOrange,
        }
        return shaders.get(_PCL_SHADER_NAMES[self.pcl_color_idx], guik.Rainbow)()

    def _load_pointcloud(self, index):
        if self.pcl_files is None:
            return None
        scan_idx = int(os.path.basename(self.gmm_files[index]).split('.')[0])
        if scan_idx >= len(self.pcl_files):
            print(f"WARNING: scan_idx {scan_idx} >= {len(self.pcl_files)} available scans")
            return None
        if self.dataset == 'nclt':
            points = load_nclt_pointcloud(self.pcl_files[scan_idx])
            points = apply_nclt_preprocessing(points, self.meta)
        else:  # tum
            points = load_tum_pointcloud(self.pcl_files[scan_idx])
            points = apply_tum_preprocessing(points, self.meta)
        return points[:, :3].astype(np.float32)

    def _render_gmm(self, viewer, index):
        """
        Render each Gaussian component as a scaled/rotated sphere primitive.

        Each ellipsoid gets its own named drawable ('ellipsoid_0', 'ellipsoid_1', ...).
        Extra drawables from the previous frame are removed if component count decreased.
        """
        gmm_file = self.gmm_files[index]
        gmm_data = load_gmm_file(gmm_file)

        if self.dataset == 'tum':
            # Invert Z to match point cloud (depth camera points down)
            gmm_data['means'][:, 2] *= -1
            D = np.array([1, 1, -1])
            gmm_data['covariances'] = (D[:, None] * gmm_data['covariances']) * D[None, :]

        self.n_components = gmm_data['n_components']

        print(f"[{index + 1}/{len(self.gmm_files)}] {os.path.basename(gmm_file)} "
              f"({self.n_components} components)")

        # --- DEBUG: reference unit sphere at origin (bright green) ---
        T_ref = np.eye(4, dtype=np.float64)
        viewer.update_drawable('_debug_origin',
                               glk.primitives.wire_sphere(),
                               guik.FlatColor(0.0, 1.0, 0.0, 1.0, T_ref))

        # --- DEBUG: raw covariance and eigenvalues of first component ---
        cov0 = gmm_data['covariances'][0]
        eigvals, _ = np.linalg.eigh(cov0)
        print(f"  [DEBUG] first component mean:       {gmm_data['means'][0]}")
        print(f"  [DEBUG] raw covariance (row 0):     {cov0[0]}")
        print(f"  [DEBUG] raw covariance (row 1):     {cov0[1]}")
        print(f"  [DEBUG] raw covariance (row 2):     {cov0[2]}")
        print(f"  [DEBUG] eigenvalues:                {eigvals}")
        print(f"  [DEBUG] semi-axes at {self.n_sigma}σ (m):   {self.n_sigma * np.sqrt(np.maximum(eigvals, 1e-9))}")

        colors_rgba = weights_to_viridis(gmm_data['weights'])  # (N, 4)
        primitive = glk.primitives.wire_sphere if self.wire_mode else glk.primitives.sphere

        for i in range(self.n_components):
            T = ellipsoid_transform(
                gmm_data['means'][i],
                gmm_data['covariances'][i],
                self.n_sigma
            )
            if self.use_viridis:
                r, g, b, a = colors_rgba[i].tolist()
            else:
                r, g, b, a = 0.2, 0.6, 1.0, 1.0

            viewer.update_drawable(
                f'ellipsoid_{i}',
                primitive(),
                guik.FlatColor(r, g, b, a, T)
            )

        # Remove drawables left over from a previous frame with more components
        for i in range(self.n_components, self._prev_n_components):
            viewer.remove_drawable(f'ellipsoid_{i}')
        self._prev_n_components = self.n_components

    def run(self):
        viewer = guik.LightViewer.instance()
        viewer.set_title("GMM Ellipsoid Visualizer (iridescence)")
        viewer.set_point_shape(self.point_size, metric=True, circle=True)

        # Initial render
        self._render_gmm(viewer, self.current_index)

        if self.show_axes:
            viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        if self.pcl_files is not None and self.show_pcl:
            pcl = self._load_pointcloud(self.current_index)
            if pcl is not None:
                viewer.update_points("pointcloud", pcl, self._get_pcl_shader())

        def ui_callback():
            imgui.begin("GMM Controls", None)

            imgui.text(f"Frame: {self.current_index + 1} / {len(self.gmm_files)}")
            imgui.text(f"File: {os.path.basename(self.gmm_files[self.current_index])}")
            imgui.text(f"Components: {self.n_components}")
            imgui.separator()

            # Navigation
            if imgui.button("< Prev (E)"):
                self._prev(); self._update_view(viewer)
            imgui.same_line()
            if imgui.button("Next (Q) >"):
                self._next(); self._update_view(viewer)

            imgui.separator()

            # Sigma
            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Sigma Level:")
            sigma_changed = False
            for sigma_val in self.sigma_options:
                if imgui.radio_button(f"{sigma_val}σ", self.n_sigma == sigma_val):
                    if self.n_sigma != sigma_val:
                        self.n_sigma = sigma_val
                        sigma_changed = True
                if sigma_val < 3:
                    imgui.same_line()
            if sigma_changed:
                self._update_view(viewer)

            imgui.separator()

            # Ellipsoid style
            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Ellipsoid style:")
            changed, self.wire_mode = imgui.checkbox("Wireframe", self.wire_mode)
            if changed:
                self._update_view(viewer)
            changed, self.use_viridis = imgui.checkbox("Viridis colormap", self.use_viridis)
            if changed:
                self._update_view(viewer)

            imgui.separator()

            # Point size (for PCL)
            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.01, 0.2)
            if changed:
                self.point_size = new_size
                viewer.set_point_shape(self.point_size, metric=True, circle=True)

            # Show axes
            changed, self.show_axes = imgui.checkbox("Show axes", self.show_axes)
            if changed:
                if self.show_axes:
                    viewer.update_coord("coords", guik.VertexColor().scale(2.0))
                else:
                    viewer.remove_drawable("coords")

            imgui.separator()

            # Auto-play
            changed, self.auto_play = imgui.checkbox("Auto-play", self.auto_play)
            if self.auto_play:
                _, self.play_speed = imgui.slider_int("Speed", self.play_speed, 1, 10)

            imgui.separator()

            # Jump to frame
            changed, new_idx = imgui.slider_int("Go to", self.current_index, 0, len(self.gmm_files) - 1)
            if changed and new_idx != self.current_index:
                self.current_index = new_idx
                self._update_view(viewer)

            # Point cloud section
            if self.pcl_files is not None:
                imgui.separator()
                imgui.text_colored(
                    np.array([0.4, 1.0, 0.5, 1.0], dtype=np.float32),
                    f"Point Cloud ({self.dataset.upper()}):")

                changed, self.show_pcl = imgui.checkbox("Show PCL", self.show_pcl)
                if changed:
                    self._update_view(viewer)

                if self.show_pcl:
                    changed, self.pcl_color_idx = imgui.combo(
                        "PCL color", self.pcl_color_idx, _PCL_SHADER_NAMES)
                    if changed:
                        pcl = self._load_pointcloud(self.current_index)
                        if pcl is not None:
                            viewer.update_points("pointcloud", pcl, self._get_pcl_shader())

                if self.meta:
                    imgui.text_colored(
                        np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32),
                        "Preprocessing (from meta):")
                    if self.meta.get('every_n'):
                        imgui.text(f"  every_n: {self.meta['every_n']}")
                    if self.meta.get('radius') is not None:
                        r_min = 0.5 if self.dataset == 'nclt' else 0.0
                        imgui.text(f"  radius:  {r_min} – {self.meta['radius']} m")
                    if self.meta.get('voxel') is not None:
                        imgui.text(f"  voxel:   {self.meta['voxel']} m")

            imgui.separator()
            imgui.text_colored(np.array([0.5, 0.8, 1.0, 1.0], dtype=np.float32), "Keyboard:")
            imgui.text("  Q - Next frame")
            imgui.text("  E - Previous frame")
            imgui.text("  1/2/3 - Set sigma")
            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        print("\nControls:")
        print("  Q - Next frame")
        print("  E - Previous frame")
        print("  1/2/3 - Set sigma level")
        print("  Mouse - Rotate/Pan/Zoom\n")

        while viewer.spin_once():
            if imgui.is_key_pressed(ord('Q')):
                self._next(); self._update_view(viewer)
            if imgui.is_key_pressed(ord('E')):
                self._prev(); self._update_view(viewer)

            for sigma_val in [1, 2, 3]:
                if imgui.is_key_pressed(ord(str(sigma_val))):
                    if self.n_sigma != sigma_val:
                        self.n_sigma = sigma_val
                        self._update_view(viewer)

            if self.auto_play:
                self.frame_counter += 1
                if self.frame_counter >= (60 // self.play_speed):
                    self.frame_counter = 0
                    if self.current_index < len(self.gmm_files) - 1:
                        self.current_index += 1
                        self._update_view(viewer)
                    else:
                        self.auto_play = False

    def _next(self):
        if self.current_index < len(self.gmm_files) - 1:
            self.current_index += 1
        else:
            print("Already at last frame")

    def _prev(self):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            print("Already at first frame")

    def _update_view(self, viewer):
        self._render_gmm(viewer, self.current_index)

        if self.pcl_files is not None:
            if self.show_pcl:
                pcl = self._load_pointcloud(self.current_index)
                if pcl is not None:
                    viewer.update_points("pointcloud", pcl, self._get_pcl_shader())
            else:
                viewer.update_points(
                    "pointcloud", np.zeros((0, 3), dtype=np.float32), guik.Rainbow())


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GMM as N-sigma ellipsoids using iridescence')
    parser.add_argument('gmm_dir', type=str,
                        help='Path to directory containing .gmm files')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--sigma', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--point-size', type=float, default=0.03)
    parser.add_argument('--dataset', type=str, default=None, choices=['nclt', 'tum'],
                        help='Dataset type for PCL overlay (nclt | tum)')
    parser.add_argument('--scene', type=str, default=None,
                        help='2013-01-10 (nclt) or rgbd_dataset_... (tum)')

    args = parser.parse_args()

    if not os.path.isdir(args.gmm_dir):
        print(f"ERROR: Directory not found: {args.gmm_dir}")
        return

    if args.dataset and not args.scene:
        # Infer scene from gmm_dir: runs/tum_<scene>/... or runs/nclt_<scene>/...
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.gmm_dir)))
        prefix = args.dataset + '_'
        if parent.startswith(prefix):
            args.scene = parent[len(prefix):]
            print(f"Inferred --scene={args.scene} from path")
        else:
            print(f"ERROR: --scene is required (could not infer from '{parent}')")
            return

    visualizer = IridescenceEllipsoidVisualizer(
        gmm_dir=args.gmm_dir,
        start_index=args.start_index,
        dataset=args.dataset,
        scene=args.scene
    )
    visualizer.n_sigma = args.sigma
    visualizer.point_size = args.point_size

    visualizer.run()


if __name__ == '__main__':
    main()
