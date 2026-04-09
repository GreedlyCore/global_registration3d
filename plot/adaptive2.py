import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Load data from JSON file

with open(os.path.expanduser("~/thesis/global_registration3d/results/adaptive/adaptive_params.json"), 'r') as f:
    data = json.load(f)

# Extract data separately for KITTI and MulRan
kitti_voxel = []
kitti_r_local = []
kitti_r_middle = []
kitti_r_global = []

mulran_voxel = []
mulran_r_local = []
mulran_r_middle = []
mulran_r_global = []

for dataset_name, dataset_content in data.items():
    if dataset_name == "OXFORD" or not dataset_content:
        continue
    
    for scene_name, scene_data in dataset_content.items():
        if dataset_name == "KITTI":
            kitti_voxel.append(scene_data['voxel_size_mean'])
            kitti_r_local.append(scene_data['r_local_mean'])
            kitti_r_middle.append(scene_data['r_middle_mean'])
            kitti_r_global.append(scene_data['r_global_mean'])
        elif dataset_name == "MulRan":
            mulran_voxel.append(scene_data['voxel_size_mean'])
            mulran_r_local.append(scene_data['r_local_mean'])
            mulran_r_middle.append(scene_data['r_middle_mean'])
            mulran_r_global.append(scene_data['r_global_mean'])

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Parameter Distribution: KITTI vs MulRan', fontsize=14, fontweight='bold')

# 1. Voxel Size
data_voxel = [kitti_voxel, mulran_voxel]
bp1 = axes[0,0].boxplot(data_voxel, labels=['KITTI', 'MulRan'], 
                         patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('lightblue')
bp1['boxes'][1].set_facecolor('lightcoral')
axes[0,0].set_ylabel('Voxel Size (m)', fontsize=11)
axes[0,0].set_title('(a) $\\nu$ (Voxel Size)', fontsize=12)
axes[0,0].grid(True, alpha=0.3, axis='y')

# Add individual points
for i, data in enumerate([kitti_voxel, mulran_voxel]):
    x = np.random.normal(i+1, 0.04, size=len(data))
    axes[0,0].scatter(x, data, alpha=0.6, color='black', s=30)

# 2. r_local
data_r_local = [kitti_r_local, mulran_r_local]
bp2 = axes[0,1].boxplot(data_r_local, labels=['KITTI', 'MulRan'], 
                         patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor('lightblue')
bp2['boxes'][1].set_facecolor('lightcoral')
axes[0,1].set_ylabel('Radius (m)', fontsize=11)
axes[0,1].set_title('(b) $r_{local}$', fontsize=12)
axes[0,1].grid(True, alpha=0.3, axis='y')

for i, data in enumerate([kitti_r_local, mulran_r_local]):
    x = np.random.normal(i+1, 0.04, size=len(data))
    axes[0,1].scatter(x, data, alpha=0.6, color='black', s=30)

# 3. r_middle
data_r_middle = [kitti_r_middle, mulran_r_middle]
bp3 = axes[1,0].boxplot(data_r_middle, labels=['KITTI', 'MulRan'], 
                         patch_artist=True, widths=0.6)
bp3['boxes'][0].set_facecolor('lightblue')
bp3['boxes'][1].set_facecolor('lightcoral')
axes[1,0].set_ylabel('Radius (m)', fontsize=11)
axes[1,0].set_title('(c) $r_{middle}$', fontsize=12)
axes[1,0].grid(True, alpha=0.3, axis='y')

for i, data in enumerate([kitti_r_middle, mulran_r_middle]):
    x = np.random.normal(i+1, 0.04, size=len(data))
    axes[1,0].scatter(x, data, alpha=0.6, color='black', s=30)

# 4. r_global
data_r_global = [kitti_r_global, mulran_r_global]
bp4 = axes[1,1].boxplot(data_r_global, labels=['KITTI', 'MulRan'], 
                         patch_artist=True, widths=0.6)
bp4['boxes'][0].set_facecolor('lightblue')
bp4['boxes'][1].set_facecolor('lightcoral')
axes[1,1].set_ylabel('Radius (m)', fontsize=11)
axes[1,1].set_title('(d) $r_{global}$', fontsize=12)
axes[1,1].grid(True, alpha=0.3, axis='y')

for i, data in enumerate([kitti_r_global, mulran_r_global]):
    x = np.random.normal(i+1, 0.04, size=len(data))
    axes[1,1].scatter(x, data, alpha=0.6, color='black', s=30)

plt.tight_layout()
plt.savefig('boxplots_kitti_vs_mulran.pdf', dpi=300, bbox_inches='tight')
plt.show()
