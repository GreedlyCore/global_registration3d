import open3d as o3d
import teaserpp_python
import numpy as np
import copy
import time
import networkx as nx
import matplotlib.pyplot as plt
from helpers import *

# VOXEL_SIZE = 0.01
VOXEL_SIZE = 0.05
# VOXEL_SIZE = 0.08
VISUALIZE = False
USE_FASTER_PFH = False  # From KISS-Matcher bindings

if USE_FASTER_PFH:
    from kiss_matcher._kiss_matcher import FasterPFH

# Load and visualize two point clouds from 3DMatch dataset
A_pcd_raw = o3d.io.read_point_cloud('./eval/data/cloud_bin_0.ply')
B_pcd_raw = o3d.io.read_point_cloud('./eval/data/cloud_bin_4.ply')
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


# =====================================================================
# Correspondence Compatibility Graph + Max Clique Visualization
# =====================================================================
# The compatibility graph is the same structure used by TEASER++, ROBIN
# (inside KISS-Matcher), and MAC:
#   nodes  = putative correspondence pairs (a_i <-> b_i)
#   edge (i,j) exists iff the pair is "geometrically compatible":
#       |dist(a_i, a_j) - dist(b_i, b_j)| < 2 * NOISE_BOUND
# True inliers form a (near-)clique because rigid transforms preserve distances.
# The maximum clique is the cleanest inlier set.

MAX_VIZ_CORRS = 70  # cap for Bron-Kerbosch tractability + readable plot

if num_corrs > MAX_VIZ_CORRS:
    rng = np.random.default_rng(0)
    viz_idx = rng.choice(num_corrs, MAX_VIZ_CORRS, replace=False)
    print(f"\n[Graph] Subsampling {MAX_VIZ_CORRS}/{num_corrs} correspondences for visualization.")
else:
    viz_idx = np.arange(num_corrs)

A_viz = A_corr[:, viz_idx]  # (3, N_VIZ)
B_viz = B_corr[:, viz_idx]  # (3, N_VIZ)
N_VIZ = len(viz_idx)

# --- Build compatibility graph (vectorized) -------------------------
t0 = time.time()
A_T = A_viz.T  # (N, 3)
B_T = B_viz.T  # (N, 3)
# pairwise distances in source and target
diff_A = A_T[:, None, :] - A_T[None, :, :]   # (N, N, 3)
diff_B = B_T[:, None, :] - B_T[None, :, :]
dist_A = np.sqrt((diff_A ** 2).sum(axis=2))   # (N, N)
dist_B = np.sqrt((diff_B ** 2).sum(axis=2))
compat = np.abs(dist_A - dist_B) < 2 * NOISE_BOUND  # (N, N) bool

G = nx.Graph()
G.add_nodes_from(range(N_VIZ))
rows, cols = np.where(np.triu(compat, k=1))
G.add_edges_from(zip(rows.tolist(), cols.tolist()))
print(f"[Graph] Built: {N_VIZ} nodes, {G.number_of_edges()} edges  ({time.time()-t0:.3f}s)")

# --- Maximum clique via Bron-Kerbosch (networkx) --------------------
t0 = time.time()
all_cliques = list(nx.find_cliques(G))
max_clique  = max(all_cliques, key=len) if all_cliques else []
print(f"[Graph] Max clique size: {len(max_clique)}  ({time.time()-t0:.3f}s)")
print(f"[Graph] Total maximal cliques found: {len(all_cliques)}")

max_clique_set = set(max_clique)

pos = nx.spring_layout(G, seed=42, k=1.5 / max(np.sqrt(N_VIZ), 1))

# ── Window 1: raw compatibility graph (no clique info) ───────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Correspondence Compatibility Graph\n"
             f"{N_VIZ} nodes  |  {G.number_of_edges()} edges",
             fontsize=13, fontweight='bold')
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color='#95a5a6')
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#7f8c8d', node_size=40)
ax.axis('off')
plt.tight_layout()
plt.savefig('graph_1_raw.png', dpi=150, bbox_inches='tight')
print("[Graph] Saved → graph_1_raw.png  (close window to continue)")
plt.show()  # blocks until closed

# ── Window 2: MAC — all maximal cliques, each a different color ──────
# Sort by size descending; assign a distinct color to each clique.
# Nodes in multiple cliques get the color of the largest clique they appear in.
all_cliques_sorted = sorted(all_cliques, key=len, reverse=True)
cmap = plt.colormaps['tab20']
node_color_mac = {n: '#d5d8dc' for n in G.nodes()}   # default: isolated/tiny
node_in_clique  = {}   # node -> index of its largest assigned clique

# Only colour cliques of size >= 2
significant = [c for c in all_cliques_sorted if len(c) >= 2]
for ci, clique in enumerate(significant):
    color = cmap(ci % 20)
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(color[0]*255), int(color[1]*255), int(color[2]*255))
    for n in clique:
        if n not in node_in_clique:          # first (= largest) clique wins
            node_in_clique[n] = ci
            node_color_mac[n] = hex_color

node_colors_mac_list = [node_color_mac[n] for n in G.nodes()]
node_sizes_mac = [80 if n in node_in_clique else 20 for n in G.nodes()]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: graph coloured by maximal-clique membership
ax = axes[0]
ax.set_title(f"MAC — All Maximal Cliques\n"
             f"{len(significant)} cliques  (each colour = one clique)",
             fontsize=12, fontweight='bold')
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.10, edge_color='#aab7b8')
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors_mac_list,
                       node_size=node_sizes_mac)
ax.axis('off')

# Right: clique-size histogram
ax = axes[1]
sizes = [len(c) for c in all_cliques]
ax.hist(sizes, bins=range(1, max(sizes)+2), color='#5dade2',
        edgecolor='white', align='left')
ax.axvline(max(sizes), color='#e74c3c', linewidth=2,
           label=f'Maximum clique = {max(sizes)}')
ax.set_xlabel('Clique size', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Maximal Clique Sizes', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

fig.suptitle('MAC approach: every maximal clique is a candidate hypothesis',
             fontsize=12)
plt.tight_layout()
plt.savefig('graph_2_mac_cliques.png', dpi=150, bbox_inches='tight')
print("[Graph] Saved → graph_2_mac_cliques.png  (close window to continue)")
plt.show()  # blocks until closed

# ── Window 3: TEASER / ROBIN — maximum clique only ──────────────────
clique_edges = [(u, v) for u, v in G.edges()
                if u in max_clique_set and v in max_clique_set]
other_edges  = [(u, v) for u, v in G.edges()
                if not (u in max_clique_set and v in max_clique_set)]

node_colors_max = ['#e74c3c' if n in max_clique_set else '#7f8c8d'
                   for n in G.nodes()]
node_sizes_max  = [120 if n in max_clique_set else 25 for n in G.nodes()]

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f"TEASER++ / ROBIN — Maximum Clique\n"
             f"{len(max_clique)} inlier correspondences (red)",
             fontsize=13, fontweight='bold')
nx.draw_networkx_edges(G, pos, edgelist=other_edges,  ax=ax,
                       alpha=0.07, edge_color='#aab7b8')
nx.draw_networkx_edges(G, pos, edgelist=clique_edges, ax=ax,
                       alpha=0.85, edge_color='#e74c3c', width=2.5)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors_max,
                       node_size=node_sizes_max)
label_dict = {n: str(n) for n in max_clique}
nx.draw_networkx_labels(G, pos, labels=label_dict, ax=ax,
                        font_size=7, font_color='white')
ax.axis('off')
plt.tight_layout()
plt.savefig('graph_3_maximum_clique.png', dpi=150, bbox_inches='tight')
print("[Graph] Saved → graph_3_maximum_clique.png")
plt.show()
