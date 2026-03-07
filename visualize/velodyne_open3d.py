#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.open3d_visualizer import InteractiveVisualizer

def load_velodyne_bin(bin_file):
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points

def main():
    parser = argparse.ArgumentParser(description='Visualize Velodyne point cloud from bin file or KITTI sequence')
    parser.add_argument('input', type=str, help='Path to Velodyne .bin file or KITTI sequence number (e.g., "00", "01")')
    parser.add_argument('--voxel-size', type=float, default=None, 
                       help='Voxel size for downsampling (in meters). If not specified, no voxel filtering is applied.')
    parser.add_argument('--radius', type=float, default=None,
                       help='Maximum radius from origin (in meters). If not specified, no radius filtering is applied.')
    parser.add_argument('--remove-plane', action='store_true',
                       help='Remove ground plane using RANSAC algorithm')
    parser.add_argument('--ransac-distance', type=float, default=0.3,
                       help='Distance threshold for RANSAC plane fitting (default: 0.3m)')
    parser.add_argument('--ransac-n', type=int, default=3,
                       help='Number of points to sample for RANSAC plane (default: 3)')
    parser.add_argument('--ransac-iters', type=int, default=1000,
                       help='Number of RANSAC iterations (default: 1000)')
    args = parser.parse_args()

    # Check if input is a 2-digit sequence number
    if len(args.input) == 2 and args.input.isdigit():
        visualizer = InteractiveVisualizer(load_velodyne_bin, args.input, 
                                          start_index=0,
                                          voxel_size=args.voxel_size,
                                          radius=args.radius,
                                          remove_plane=args.remove_plane,
                                          ransac_distance=args.ransac_distance,
                                          ransac_n=args.ransac_n,
                                          ransac_iters=args.ransac_iters)
        visualizer.run()
    else:
        # Original behavior for bin file path
        bin_dir = os.path.dirname(args.input)
        bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))

        if not bin_files:
            print(f"No .bin files found in {bin_dir}")
            return

        try:
            start_index = bin_files.index(args.input)
        except ValueError:
            print(f"Specified file {args.input} not found in directory")
            return

        print(f"Found {len(bin_files)} point cloud files in {bin_dir}")

        visualizer = InteractiveVisualizer(load_velodyne_bin, bin_files, 
                                          start_index,
                                          voxel_size=args.voxel_size,
                                          radius=args.radius,
                                          remove_plane=args.remove_plane,
                                          ransac_distance=args.ransac_distance,
                                          ransac_n=args.ransac_n,
                                          ransac_iters=args.ransac_iters)
        visualizer.run()

if __name__ == '__main__':
    main()
