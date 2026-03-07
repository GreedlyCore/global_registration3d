import open3d as o3d
import teaserpp_python
import numpy as np
import copy
import time
from helpers import *


# /home/sonieth3/thesis/investigate/KISS-Matcher/cpp/examples/build/data/KITTI00-to-07

# VOXEL_SIZE = 0.01
VOXEL_SIZE = 0.05
# VOXEL_SIZE = 0.08
VISUALIZE = False
USE_FASTER_PFH = False  # From KISS-Matcher bindings

if USE_FASTER_PFH:
    from kiss_matcher._kiss_matcher import FasterPFH

# Load and visualize two point clouds from 3DMatch dataset
A_pcd_raw = o3d.io.read_point_cloud('./data/cloud_bin_0.ply')
B_pcd_raw = o3d.io.read_point_cloud('./data/cloud_bin_4.ply')
A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

# voxel downsample both clouds
A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

# extract features
t0 = time.time()
if USE_FASTER_PFH:
    extractor = FasterPFH(normal_radius=VOXEL_SIZE * 2,
                          fpfh_radius=VOXEL_SIZE * 5,
                          thr_linearity=0.9)
    # compute() takes (N,3) float32 and returns (valid_points (M,3), descriptors (M,33))
    A_xyz, A_feats = extractor.compute(np.asarray(A_pcd.points).astype(np.float32))
    B_xyz, B_feats = extractor.compute(np.asarray(B_pcd.points).astype(np.float32))
    A_xyz = A_xyz.T  # (3, M)
    B_xyz = B_xyz.T  # (3, M)
else:
    A_xyz = pcd2xyz(A_pcd)  # (3, N)
    B_xyz = pcd2xyz(B_pcd)  # (3, M)
    A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)
feat_name = "FasterPFH" if USE_FASTER_PFH else "FPFH"
print(f"Extract features:       {time.time() - t0:.3f}s")

# establish correspondences by nearest neighbour search in feature space
t0 = time.time()
corrs_A, corrs_B = find_correspondences(
    A_feats, B_feats, mutual_filter=True)
A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs
print(f"Find correspondences:   {time.time() - t0:.3f}s")

num_corrs = A_corr.shape[1]
print(f'{feat_name} generates {num_corrs} putative correspondences.')

# visualize the point clouds together with feature correspondences
points = np.concatenate((A_corr.T,B_corr.T),axis=0)
lines = []
for i in range(num_corrs):
    lines.append([i,i+num_corrs])
colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

# robust global registration using TEASER++
NOISE_BOUND = VOXEL_SIZE
teaser_solver = get_teaser_solver(NOISE_BOUND)
t0 = time.time()
teaser_solver.solve(A_corr,B_corr)
print(f"Coarse registration:    {time.time() - t0:.3f}s")
solution = teaser_solver.getSolution()
R_teaser = solution.rotation
t_teaser = solution.translation
T_teaser = Rt2T(R_teaser,t_teaser)

print(T_teaser)

# Visualize the registration results
A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])

# local refinement using ICP
# icp_sol = o3d.pipelines.registration.registration_icp(
#       A_pcd, B_pcd, NOISE_BOUND, T_teaser,
#       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
# T_icp = icp_sol.transformation

# visualize the registration after ICP refinement
# A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
# o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])


