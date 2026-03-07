#!/usr/bin/env python
"""
Preprocessing setup visualizer for VIRAL and NCLT datasets.
Visualizes point clouds with configurable filters: every-n, radius, voxel, mahal.

Usage:
    python visualize/preprocessing_setup.py --dataset viral --scene eee_03
    python visualize/preprocessing_setup.py --dataset nclt  --scene 2013-01-10
"""

"""
~3676 idx: pre indoor scene index
~3880 - 4070 idx: starting indoor scene index (там его ещё шатают вперёд назад неплохо так)

python3 create_and_save_gmm_nclt.py \
    --scene 2013-01-10 \
    --every-n 5 \
    --voxel 0.05 \
    --start-id 3880 --end-id 4070 \
    --mahal 2.0 \
    --n_components 150
"""

import os
import sys
import argparse
import numpy as np
from pyridescence import guik, imgui

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../../'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from utils.viral_loader import (
    load_viral_lidar_config, load_viral_pointcloud,
    get_viral_scan_count, apply_transform
)
from utils.nclt_loader import (
    get_nclt_sync_files, get_nclt_scan_count, load_nclt_pointcloud
)
from utils.pcl_filters import voxel_filter, radius_filter, every_n_filter, mahal_filter

VIRAL_BASE_PATH = os.path.join(REPO_ROOT, 'dataset/viral')
NCLT_BASE_PATH = os.path.expanduser('~/thesis/data')


class PreprocessingVisualizer:
    def __init__(self, dataset, scene):
        self.dataset = dataset
        self.scene = scene
        self.current_index = 0

        if dataset == 'viral':
            bag_path = os.path.join(VIRAL_BASE_PATH, scene, f'{scene}.bag')
            if not os.path.exists(bag_path):
                raise FileNotFoundError(f'Bag not found: {bag_path}')
            self.bag_path = bag_path

            config_topic, self.T_body_lidar = load_viral_lidar_config(bag_path)
            self.lidar_topic = config_topic or '/os1_cloud_node1/points'

            print(f'Counting scans in {bag_path}...')
            self.total_scans = get_viral_scan_count(bag_path, self.lidar_topic)
            print(f'VIRAL / {scene} ({self.total_scans} scans), topic: {self.lidar_topic}')

        elif dataset == 'nclt':
            self.nclt_files = get_nclt_sync_files(NCLT_BASE_PATH, scene)
            self.total_scans = len(self.nclt_files)
            print(f'NCLT / {scene} ({self.total_scans} scans)')

        # Filter state (order matches create_and_save_gmm_viral.py pipeline)
        self.every_n_enabled = False
        self.every_n = 20

        self.radius_enabled = False
        self.radius_max = 50.0

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
        if self.dataset == 'viral':
            points = load_viral_pointcloud(self.bag_path, index, self.lidar_topic)
            points = apply_transform(points, self.T_body_lidar)
        else:  # nclt
            points = load_nclt_pointcloud(self.nclt_files[index])
        original = len(points)

        # Apply filters in same order as create_and_save_gmm_viral.py
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
                changed, self.radius_max = imgui.slider_float("  Max radius (m)", self.radius_max, 0.0, 50.0)
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
    parser = argparse.ArgumentParser(description='Preprocessing setup visualizer for VIRAL/NCLT datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['viral', 'nclt'])
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene name (e.g., eee_03 for VIRAL, 2013-01-10 for NCLT)')
    args = parser.parse_args()

    visualizer = PreprocessingVisualizer(args.dataset, args.scene)
    visualizer.run()


if __name__ == '__main__':
    main()
