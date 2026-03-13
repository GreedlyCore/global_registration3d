#!/usr/bin/env python
"""
Sequence pair generator/visualizer for KITTI and NCLT datasets.
Shows scan pairs [src_idx, src_idx + dist_idx] with tgt aligned into src frame via GT transform.

Usage:
    python visualize/generate_seq.py --dataset kitti --scene 01
    python visualize/generate_seq.py --dataset nclt  --scene 2013-01-10

Controls:
    Space  : advance to next pair  [idx → idx+1, tgt = idx+1 + dist_idx]
"""

# dist_idx = 10
"""
[11356, 11366]  GT dist: 2.627 m  | rot: 2.10 deg
[11545, 11555]  GT dist: 2.224 m  | rot: 25.88 deg
[11735, 11745]  GT dist: 2.750 m  | rot: 4.57 deg
[11924, 11934]  GT dist: 2.802 m  | rot: 4.49 deg
[12113, 12123]  GT dist: 2.731 m  | rot: 5.50 deg
[13059, 13069]  GT dist: 2.859 m  | rot: 12.82 deg
[12870, 12880]  GT dist: 2.593 m  | rot: 7.41 deg
[12681, 12691]  GT dist: 2.764 m  | rot: 5.38 deg
[12492, 12502]  GT dist: 2.319 m  | rot: 20.90 deg
[12302, 12312]  GT dist: 2.589 m  | rot: 5.11 deg
[12492, 12502]  GT dist: 2.319 m  | rot: 20.90 deg
[12681, 12691]  GT dist: 2.764 m  | rot: 5.38 deg
[12870, 12880]  GT dist: 2.593 m  | rot: 7.41 deg
[12681, 12691]  GT dist: 2.764 m  | rot: 5.38 deg
---
[14250, 14260]  GT dist: 2.794 m  | rot: 6.57 deg
[14251, 14261]  GT dist: 2.821 m  | rot: 6.77 deg
[14252, 14262]  GT dist: 2.822 m  | rot: 5.85 deg
[14253, 14263]  GT dist: 2.823 m  | rot: 4.94 deg
[14254, 14264]  GT dist: 2.817 m  | rot: 2.24 deg
[14255, 14265]  GT dist: 2.806 m  | rot: 1.62 deg
[14256, 14266]  GT dist: 2.799 m  | rot: 4.31 deg
[14257, 14267]  GT dist: 2.782 m  | rot: 6.09 deg
[14258, 14268]  GT dist: 2.747 m  | rot: 7.55 deg
[14259, 14269]  GT dist: 2.698 m  | rot: 8.59 deg
[14260, 14270]  GT dist: 2.666 m  | rot: 9.02 deg
[14261, 14271]  GT dist: 2.647 m  | rot: 7.89 deg
[14262, 14272]  GT dist: 2.648 m  | rot: 6.00 deg
[14263, 14273]  GT dist: 2.635 m  | rot: 4.39 deg
[14264, 14274]  GT dist: 2.618 m  | rot: 2.10 deg
[14265, 14275]  GT dist: 2.591 m  | rot: 2.76 deg
[14266, 14276]  GT dist: 2.564 m  | rot: 5.42 deg
[14267, 14277]  GT dist: 2.560 m  | rot: 6.28 deg
[14268, 14278]  GT dist: 2.581 m  | rot: 5.65 deg
---
[15920, 15930]  GT dist: 2.768 m  | rot: 2.33 deg
[15921, 15931]  GT dist: 2.746 m  | rot: 2.51 deg
[15922, 15932]  GT dist: 2.719 m  | rot: 2.64 deg
[15923, 15933]  GT dist: 2.706 m  | rot: 2.99 deg
[15924, 15934]  GT dist: 2.700 m  | rot: 2.40 deg
[15925, 15935]  GT dist: 2.689 m  | rot: 3.26 deg
[15926, 15936]  GT dist: 2.670 m  | rot: 3.34 deg
[15927, 15937]  GT dist: 2.662 m  | rot: 2.94 deg
[15928, 15938]  GT dist: 2.673 m  | rot: 3.52 deg
[15929, 15939]  GT dist: 2.697 m  | rot: 2.75 deg
[15930, 15940]  GT dist: 2.707 m  | rot: 2.34 deg
[15931, 15941]  GT dist: 2.711 m  | rot: 2.72 deg
[15932, 15942]  GT dist: 2.719 m  | rot: 0.88 deg
[15933, 15943]  GT dist: 2.720 m  | rot: 3.42 deg
[15934, 15944]  GT dist: 2.700 m  | rot: 2.73 deg
[15935, 15945]  GT dist: 2.673 m  | rot: 5.97 deg
[15936, 15946]  GT dist: 2.657 m  | rot: 6.14 deg
"""

import os
import sys
import argparse
import numpy as np
from pyridescence import guik, imgui

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'eval'))

from dataset_loader import (
    load_kitti_dataset, load_kitti_velodyne_pcd,
    load_nclt_dataset,  load_nclt_velodyne_pcd,
)


# --------------------------------------------------------------------------- #
# Geometry helpers (mirrors eval/test.py)
# --------------------------------------------------------------------------- #

def gt_transform(poses, Tr, src_idx, tgt_idx):
    """Ground-truth relative transform: src frame → tgt frame."""
    Tr_inv = np.linalg.inv(Tr)
    return Tr_inv @ np.linalg.inv(poses[tgt_idx]) @ poses[src_idx] @ Tr


def rotation_angle_deg(R):
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# --------------------------------------------------------------------------- #
# Visualizer
# --------------------------------------------------------------------------- #

class SeqPairVisualizer:
    def __init__(self, dataset, scene):
        self.dataset = dataset
        self.scene   = scene

        if dataset == 'kitti':
            self.scan_files, self.poses, self.Tr = load_kitti_dataset(scene)
            self.load_pcd = load_kitti_velodyne_pcd
        elif dataset == 'nclt':
            self.scan_files, self.poses, self.Tr = load_nclt_dataset(scene)
            self.load_pcd = load_nclt_velodyne_pcd
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

        self.total_scans = len(self.scan_files)
        print(f'{dataset.upper()} / {scene}  ({self.total_scans} scans)')

        # Pair state
        self.dist_idx   = 10
        self.src_idx    = 0
        self.tgt_idx    = self.dist_idx
        self.gt_dist    = 0.0
        self.gt_rot_deg = 0.0

        # Rendering
        self.point_size = 0.05

    # ------------------------------------------------------------------ #

    def _clamp_src(self):
        """Ensure src_idx is valid for current dist_idx."""
        max_src = self.total_scans - self.dist_idx - 1
        self.src_idx = int(np.clip(self.src_idx, 0, max_src))
        self.tgt_idx = self.src_idx + self.dist_idx

    def _advance(self):
        """Step src_idx forward by 1 (wraps at end)."""
        max_src = self.total_scans - self.dist_idx - 1
        self.src_idx = (self.src_idx + 1) % (max_src + 1)
        self.tgt_idx = self.src_idx + self.dist_idx

    def _load_and_update(self, viewer):
        T_gt = gt_transform(self.poses, self.Tr, self.src_idx, self.tgt_idx)
        self.gt_dist    = float(np.linalg.norm(T_gt[:3, 3]))
        self.gt_rot_deg = rotation_angle_deg(T_gt[:3, :3])

        print(f'[{self.src_idx}, {self.tgt_idx}]  GT dist: {self.gt_dist:.3f} m  '
              f'| rot: {self.gt_rot_deg:.2f} deg')
        # print('GT transform:')
        # with np.printoptions(precision=3, suppress=True):
        #     print(T_gt)

        src_pcd = self.load_pcd(self.scan_files[self.src_idx])
        tgt_pcd = self.load_pcd(self.scan_files[self.tgt_idx])

        src_pts = np.asarray(src_pcd.points).astype(np.float32)
        tgt_pts = np.asarray(tgt_pcd.points).astype(np.float32)

        # Bring tgt into src frame:  p_src = T_gt^-1 @ p_tgt
        T_inv = np.linalg.inv(T_gt)
        R_inv = T_inv[:3, :3].astype(np.float32)
        t_inv = T_inv[:3, 3 ].astype(np.float32)
        tgt_in_src = tgt_pts @ R_inv.T + t_inv

        # NCLT visual fix: sensor frame mirrors NED (z points Down, x≈Backward).
        # Flip to ENU-like (x←y, y←x, z←−z) so the scene appears z-Up in the
        # viewer with ground below and sky above.  This is purely cosmetic —
        # the GT transform and all metrics are computed before this flip.
        if self.dataset == 'nclt':
            R_ned_enu = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]], dtype=np.float32)
            src_pts   = src_pts   @ R_ned_enu.T
            tgt_in_src = tgt_in_src @ R_ned_enu.T

        viewer.update_points("src", src_pts,    guik.FlatGreen())
        viewer.update_points("tgt", tgt_in_src, guik.FlatRed())

    # ------------------------------------------------------------------ #

    def run(self):
        viewer = guik.LightViewer.instance()
        viewer.set_title(f"Sequence Pair Generator — {self.dataset.upper()}/{self.scene}")
        viewer.set_point_shape(self.point_size, metric=True, circle=True)
        viewer.update_coord("coords", guik.VertexColor().scale(2.0))

        self._clamp_src()
        self._load_and_update(viewer)

        def ui_callback():
            imgui.begin("Controls", None)

            imgui.text(f"{self.dataset.upper()} / {self.scene}")
            imgui.text(f"Total scans: {self.total_scans}")
            imgui.separator()

            imgui.text(f"src_idx : {self.src_idx}")
            imgui.text(f"tgt_idx : {self.tgt_idx}")
            imgui.separator()

            imgui.text_colored(np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32), "Green: src")
            imgui.text_colored(np.array([1.0, 0.3, 0.3, 1.0], dtype=np.float32), "Red  : tgt (in src frame via GT)")
            imgui.separator()

            imgui.text_colored(np.array([1.0, 0.8, 0.3, 1.0], dtype=np.float32), "GT relative transform:")
            imgui.text(f"  Distance : {self.gt_dist:.3f} m")
            imgui.text(f"  Rotation : {self.gt_rot_deg:.2f} deg")
            imgui.separator()

            changed, new_dist = imgui.slider_int(
                "dist_idx", self.dist_idx, 1,
                min(200, self.total_scans - 1))
            if changed:
                self.dist_idx = new_dist
                self._clamp_src()
                self._load_and_update(viewer)

            changed, new_src = imgui.slider_int(
                "src_idx", self.src_idx, 0,
                self.total_scans - self.dist_idx - 1)
            if changed and new_src != self.src_idx:
                self.src_idx = new_src
                self.tgt_idx = self.src_idx + self.dist_idx
                self._load_and_update(viewer)

            imgui.separator()

            if imgui.button("Next pair [Space]"):
                self._advance()
                self._load_and_update(viewer)

            imgui.separator()

            changed, new_size = imgui.slider_float("Point size", self.point_size, 0.01, 2.0)
            if changed:
                self.point_size = new_size
                viewer.set_point_shape(self.point_size, metric=True, circle=True)

            imgui.separator()
            imgui.text("Space : next pair")
            imgui.end()

        viewer.register_ui_callback("controls", ui_callback)

        while viewer.spin_once():
            if imgui.is_key_pressed(ord(' ')):
                self._advance()
                self._load_and_update(viewer)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Sequence pair generator/visualizer for KITTI/NCLT datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['kitti', 'nclt'])
    parser.add_argument('--scene',   type=str, required=True,
                        help='Sequence ID: 01/04 for KITTI, 2013-01-10 for NCLT')
    args = parser.parse_args()

    visualizer = SeqPairVisualizer(args.dataset, args.scene)
    visualizer.run()


if __name__ == '__main__':
    main()
