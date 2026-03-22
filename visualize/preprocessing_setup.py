#!/usr/bin/env python
"""
Preprocessing setup visualizer for KITTI, NCLT, and MulRan datasets.
Visualizes point clouds with configurable filters: every-n, radius, voxel, mahal.

Usage:
    python visualize/preprocessing_setup.py --dataset kitti --scene 01
    python visualize/preprocessing_setup.py --dataset kitti --scene 04
    python visualize/preprocessing_setup.py --dataset nclt  --scene 2013-01-10
"""

"""
~3676 idx: pre indoor scene index
~3880 - 4070 idx: starting indoor scene index (там его ещё шатают вперёд назад неплохо так)
"""

import os
import sys
import argparse
import numpy as np
from pyridescence import guik, imgui

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'eval'))
from dataset_loader import (
    load_kitti_dataset,
    load_kitti_velodyne,
    load_nclt_dataset,
    load_nclt_velodyne,
    load_mulran_dataset,
    load_mulran_ouster,
    load_oxford_dataset,
    load_oxford_lidar,
)

KITTI_BASE_PATH = os.path.join(REPO_ROOT, 'data', 'KITTI')
NCLT_BASE_PATH  = os.path.join(REPO_ROOT, 'data', 'NCLT')


def _get_kitti_scan_files(scene):
    vel_dir = os.path.join(KITTI_BASE_PATH, 'sequences', scene, 'velodyne')
    if not os.path.isdir(vel_dir):
        raise FileNotFoundError(f'KITTI velodyne directory not found: {vel_dir}')
    return sorted(
        os.path.join(vel_dir, f)
        for f in os.listdir(vel_dir)
        if f.endswith('.bin')
    )


def every_n_filter(points: np.ndarray, n: int, verbose: bool = False) -> np.ndarray:
    step = max(int(n), 1)
    return points[::step]


def radius_filter(points: np.ndarray, rmin: float, rmax: float, verbose: bool = False) -> np.ndarray:
    if points.size == 0:
        return points
    xyz = points[:, :3]
    dist = np.linalg.norm(xyz, axis=1)
    keep = (dist >= float(rmin)) & (dist <= float(rmax))
    return points[keep]


def voxel_filter(points: np.ndarray, voxel_size: float, verbose: bool = False) -> np.ndarray:
    if points.size == 0:
        return points
    v = float(voxel_size)
    if v <= 0.0:
        return points

    xyz = points[:, :3]
    keys = np.floor(xyz / v).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx]


def mahal_filter(points: np.ndarray, threshold: float, verbose: bool = False) -> np.ndarray:
    if points.size == 0:
        return points
    xyz = points[:, :3]
    if xyz.shape[0] < 5:
        return points

    mean = np.mean(xyz, axis=0)
    centered = xyz - mean
    cov = np.cov(centered, rowvar=False)
    cov += 1e-6 * np.eye(3)
    inv_cov = np.linalg.inv(cov)
    md2 = np.einsum('ij,jk,ik->i', centered, inv_cov, centered)
    keep = md2 <= float(threshold) ** 2
    return points[keep]


class PreprocessingVisualizer:
    def __init__(self, dataset, scene):
        self.dataset = dataset
        self.scene = scene
        self.current_index = 0

        if dataset == 'kitti':
            self.scan_files, _, _ = load_kitti_dataset(scene)
            self.total_scans = len(self.scan_files)
            print(f'KITTI / seq {scene} ({self.total_scans} scans)')

        elif dataset == 'nclt':
            self.scan_files, _, _ = load_nclt_dataset(scene)
            self.total_scans = len(self.scan_files)
            print(f'NCLT / {scene} ({self.total_scans} scans)')
        elif dataset == 'mulran':
            scene = scene.upper()
            self.scene = scene
            self.scan_files, _, _ = load_mulran_dataset(scene)
            self.total_scans = len(self.scan_files)
            print(f'MulRan / {scene} ({self.total_scans} scans)')
        elif dataset == 'oxford':
            self.scan_files, _, _ = load_oxford_dataset(scene)
            self.total_scans = len(self.scan_files)
            print(f'Oxford / {scene} ({self.total_scans} scans)')
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

        # Filter state (order matches create_and_save_gmm_viral.py pipeline)
        self.every_n_enabled = False
        self.every_n = 20

        self.radius_enabled = False
        self.radius_max = 100.0

        self.voxel_enabled = False
        self.voxel_size = 0.5

        self.mahal_enabled = False
        self.mahal_threshold = 3.0

        # Rendering state
        self.point_size = 0.05
        self.colormap = 'rainbow'
        self.colormap_idx = 0

        # Playback
        self.auto_play = False
        self.play_speed = 1
        self.frame_counter = 0

    def _load_pointcloud(self, index):
        if self.dataset == 'kitti':
            points = load_kitti_velodyne(self.scan_files[index])
        elif self.dataset == 'nclt':
            points = load_nclt_velodyne(self.scan_files[index])
        elif self.dataset == 'mulran':
            points = load_mulran_ouster(self.scan_files[index])
        else:
            points = load_oxford_lidar(self.scan_files[index])
        original = len(points)

        # Apply filters
        if self.every_n_enabled:
            points = every_n_filter(points, n=self.every_n, verbose=False)
        if self.radius_enabled:
            points = radius_filter(points, 0.0, self.radius_max, verbose=False)
        if self.voxel_enabled:
            points = voxel_filter(points, self.voxel_size, verbose=False)
        if self.mahal_enabled:
            points = mahal_filter(points, self.mahal_threshold, verbose=False)

        print(f'[{index + 1}/{self.total_scans}]  {original} --> {len(points)} pts')
        return points[:, :3].astype(np.float32)

    def _get_shader(self):
        shaders = {
            'rainbow': guik.Rainbow,
            'flat_red': guik.FlatRed,
            'flat_green': guik.FlatGreen,
            'flat_blue': guik.FlatBlue,
            'flat_orange': guik.FlatOrange,
        }
        return shaders.get(self.colormap, guik.Rainbow)()

    def _update_view(self, viewer):
        points = self._load_pointcloud(self.current_index)
        viewer.update_points("pointcloud", points, self._get_shader())

    def run(self):
        viewer = guik.LightViewer.instance()
        viewer.set_title(f"Preprocessing Visualizer - {self.dataset.upper()}/{self.scene}")
        viewer.set_point_shape(self.point_size, metric=True, circle=True)

        points = self._load_pointcloud(self.current_index)
        viewer.update_points("pointcloud", points, self._get_shader())
        viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        colormaps = ['rainbow', 'flat_red', 'flat_green', 'flat_blue', 'flat_orange']

        def ui_callback():
            imgui.begin("Controls", None)

            imgui.text(f"{self.dataset.upper()} / {self.scene}")
            imgui.text(f"Frame: {self.current_index + 1} / {self.total_scans}")
            imgui.separator()

            # Navigation
            if imgui.button("< Prev (E)"):
                self._prev()
                self._update_view(viewer)
            imgui.same_line()
            if imgui.button("Next (Q) >"):
                self._next()
                self._update_view(viewer)

            changed, new_idx = imgui.slider_int("Go to", self.current_index, 0, self.total_scans - 1)
            if changed and new_idx != self.current_index:
                self.current_index = new_idx
                self._update_view(viewer)

            imgui.separator()

            # Auto-play
            changed, self.auto_play = imgui.checkbox("Auto-play", self.auto_play)
            if self.auto_play:
                _, self.play_speed = imgui.slider_int("Speed", self.play_speed, 1, 10)

            imgui.separator()

            # Rendering
            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "Rendering:")
            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.01, 2.0)
            if changed:
                self.point_size = new_size
                viewer.set_point_shape(self.point_size, metric=True, circle=True)

            changed, self.colormap_idx = imgui.combo("Colormap", self.colormap_idx, colormaps)
            if changed:
                self.colormap = colormaps[self.colormap_idx]
                self._update_view(viewer)

            imgui.separator()

            # Preprocessing filters (order matches GMM creation pipeline)
            imgui.text_colored(np.array([0.5, 0.8, 1.0, 1.0], dtype=np.float32), "Preprocessing filters:")
            filters_changed = False

            changed, self.every_n_enabled = imgui.checkbox("Every-N", self.every_n_enabled)
            filters_changed |= changed
            if self.every_n_enabled:
                changed, self.every_n = imgui.slider_int("  N (keep 1 in N)", self.every_n, 1, 20)
                filters_changed |= changed

            changed, self.radius_enabled = imgui.checkbox("Radius filter", self.radius_enabled)
            filters_changed |= changed
            if self.radius_enabled:
                changed, self.radius_max = imgui.slider_float("  Max radius (m)", self.radius_max, 0.0, 100.0)
                filters_changed |= changed

            changed, self.voxel_enabled = imgui.checkbox("Voxel filter", self.voxel_enabled)
            filters_changed |= changed
            if self.voxel_enabled:
                changed, self.voxel_size = imgui.slider_float("  Voxel size (m)", self.voxel_size, 0.0, 0.5)
                filters_changed |= changed

            changed, self.mahal_enabled = imgui.checkbox("Mahal outlier removal", self.mahal_enabled)
            filters_changed |= changed
            if self.mahal_enabled:
                changed, self.mahal_threshold = imgui.slider_float("  Distance threshold", self.mahal_threshold, 1.0, 10.0)
                filters_changed |= changed

            if filters_changed:
                self._update_view(viewer)

            imgui.separator()
            imgui.text("Keys: Q=Next, E=Prev")
            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        while viewer.spin_once():
            if imgui.is_key_pressed(ord('Q')):
                self._next()
                self._update_view(viewer)
            if imgui.is_key_pressed(ord('E')):
                self._prev()
                self._update_view(viewer)

            if self.auto_play:
                self.frame_counter += 1
                if self.frame_counter >= (60 // self.play_speed):
                    self.frame_counter = 0
                    if self.current_index < self.total_scans - 1:
                        self.current_index += 1
                        self._update_view(viewer)
                    else:
                        self.auto_play = False

    def _next(self):
        if self.current_index < self.total_scans - 1:
            self.current_index += 1

    def _prev(self):
        if self.current_index > 0:
            self.current_index -= 1


def main():
    parser = argparse.ArgumentParser(description='Preprocessing setup visualizer for KITTI/NCLT/MulRan datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['kitti', 'nclt', 'mulran', 'oxford'])
    parser.add_argument('--scene', type=str, required=True,
                        help='Sequence ID: 01/04 for KITTI, 2013-01-10 for NCLT, DCC02 for MulRan, <seq_name> for Oxford')
    args = parser.parse_args()

    visualizer = PreprocessingVisualizer(args.dataset, args.scene)
    visualizer.run()


if __name__ == '__main__':
    main()
