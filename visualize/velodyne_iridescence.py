#!/usr/bin/env python
"""Point cloud visualizer using iridescence. Supports KITTI and VIRAL datasets."""
import os
import sys
import glob
import argparse
import numpy as np
from pyridescence import guik, imgui

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../../'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from utils.kitti_loader import load_kitti_velodyne
from utils.viral_loader import load_viral_pointcloud, get_viral_scan_count
from utils.pcl_filters import voxel_filter, radius_filter, remove_plane_ransac

DATASET_CONFIG = {
    'kitti': {
        'base_path': os.path.join(REPO_ROOT, 'dataset/kitti/data_odometry_velodyne/dataset/sequences'),
        'pattern': '{base}/{seq}/velodyne/*.bin',
    },
    'viral': {
        'base_path': os.path.join(REPO_ROOT, 'dataset/viral'),
        'pattern': '{base}/{seq}/{seq}.bag',
    },
}


class IridescenceVisualizer:
    def __init__(self, dataset, sequence):
        self.dataset = dataset
        self.sequence = sequence
        self.current_index = 0

        config = DATASET_CONFIG[dataset]
        base = config['base_path']

        if dataset == 'kitti':
            pattern = config['pattern'].format(base=base, seq=sequence)
            self.file_list = sorted(glob.glob(pattern))
            if not self.file_list:
                raise ValueError(f'No files found: {pattern}')
            self.total_scans = len(self.file_list)
        else:  # viral
            self.bag_path = config['pattern'].format(base=base, seq=sequence)
            if not os.path.exists(self.bag_path):
                raise FileNotFoundError(f'Bag not found: {self.bag_path}')
            print(f'Counting scans in {self.bag_path}...')
            self.total_scans = get_viral_scan_count(self.bag_path)

        print(f'Dataset: {dataset.upper()} / {sequence} ({self.total_scans} scans)')

        # Filter state (GUI-controlled)
        self.voxel_enabled = False
        self.voxel_size = 0.1
        self.radius_enabled = False
        self.radius = 50.0
        self.plane_enabled = False
        self.ransac_distance = 0.3
        self.ransac_iters = 1000

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
            points = load_kitti_velodyne(self.file_list[index])
        else:
            points = load_viral_pointcloud(self.bag_path, index)

        original = len(points)

        if self.radius_enabled:
            points = radius_filter(points, 0.5, self.radius, verbose=False)
        if self.voxel_enabled:
            points = voxel_filter(points, self.voxel_size, verbose=False)
        if self.plane_enabled:
            points = remove_plane_ransac(points, self.ransac_distance, 3, self.ransac_iters, verbose=False)

        print(f'[{index + 1}/{self.total_scans}] {len(points)} pts ({100 * len(points) / original:.0f}%)')
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
        viewer.set_title(f"Point Cloud Viewer - {self.dataset.upper()}/{self.sequence}")
        viewer.set_point_shape(self.point_size, metric=True, circle=True)

        points = self._load_pointcloud(self.current_index)
        viewer.update_points("pointcloud", points, self._get_shader())
        viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        colormaps = ['rainbow', 'flat_red', 'flat_green', 'flat_blue', 'flat_orange']

        def ui_callback():
            imgui.begin("Controls", None)

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

            # Filters
            imgui.text_colored(np.array([0.5, 0.8, 1.0, 1.0], dtype=np.float32), "Filters:")
            filters_changed = False

            changed, self.radius_enabled = imgui.checkbox("Radius filter", self.radius_enabled)
            filters_changed |= changed
            if self.radius_enabled:
                changed, self.radius = imgui.slider_float("  Max radius (m)", self.radius, 5.0, 100.0)
                filters_changed |= changed

            changed, self.voxel_enabled = imgui.checkbox("Voxel filter", self.voxel_enabled)
            filters_changed |= changed
            if self.voxel_enabled:
                changed, self.voxel_size = imgui.slider_float("  Voxel size (m)", self.voxel_size, 0.01, 1.0)
                filters_changed |= changed

            changed, self.plane_enabled = imgui.checkbox("Remove plane (RANSAC)", self.plane_enabled)
            filters_changed |= changed
            if self.plane_enabled:
                changed, self.ransac_distance = imgui.slider_float("  Distance thresh", self.ransac_distance, 0.1, 1.0)
                filters_changed |= changed
                changed, self.ransac_iters = imgui.slider_int("  Iterations", self.ransac_iters, 100, 5000)
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
    parser = argparse.ArgumentParser(description='Point cloud visualizer (KITTI/VIRAL)')
    parser.add_argument('--dataset', type=str, required=True, choices=['kitti', 'viral'])
    parser.add_argument('--sequence', type=str, required=True, help='Sequence name (e.g., "05" or "eee_03")')
    args = parser.parse_args()

    visualizer = IridescenceVisualizer(args.dataset, args.sequence)
    visualizer.run()


if __name__ == '__main__':
    main()
