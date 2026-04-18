import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# Load data from JSON file
with open('/home/sonieth3/thesis/global_registration3d/results/adaptive/adaptive_params.json', 'r') as f:
    data = json.load(f)

# Extract scenes and parameters
scenes = []
voxel_means = []
r_local_means = []
r_middle_means = []
r_global_means = []

for dataset_name, dataset_content in data.items():
    if dataset_name == "OXFORD" and not dataset_content:
        continue  # Skip empty Oxford
    for scene_name, scene_data in dataset_content.items():
        scenes.append(f"{dataset_name}{scene_name}")
        voxel_means.append(scene_data['voxel_size_mean'])
        r_local_means.append(scene_data['r_local_mean'])
        r_middle_means.append(scene_data['r_middle_mean'])
        r_global_means.append(scene_data['r_global_mean'])

# Normalize data for radar chart (min-max scaling)
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

voxel_norm = normalize(np.array(voxel_means))
r_local_norm = normalize(np.array(r_local_means))
r_middle_norm = normalize(np.array(r_middle_means))
r_global_norm = normalize(np.array(r_global_means))

# Prepare data for radar
metrics = ['$\\nu$ ', '$r_{loc}$', '$r_{mid}$', '$r_{global}$']
num_vars = len(metrics)

# Create angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Create figure
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

# Colors for different scenes
colors = plt.cm.tab20(np.linspace(0, 1, len(scenes)))

# Plot each scene
for idx, scene in enumerate(scenes):
    values = [voxel_norm[idx], r_local_norm[idx], r_middle_norm[idx], r_global_norm[idx]]
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, 'o-', linewidth=1.5, 
            color=colors[idx], label=scene, alpha=0.7)
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

# Set labels with LaTeX
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=10)
# ax.set_ylabel('Normalized Value', fontsize=10, labelpad=20)
# ax.set_title('Parameter Comparison Across Scenes', 
#              fontsize=14, fontweight='bold', pad=20)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
          fontsize=9, framealpha=0.9)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('radar_chart_params.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Scenes analyzed: {len(scenes)}")
print("\nVoxel size ($\\nu$) range: {:.3f} - {:.3f} m".format(
    min(voxel_means), max(voxel_means)))
print("$r_{loc}$ range: {:.3f} - {:.3f} m".format(
    min(r_local_means), max(r_local_means)))
print("$r_{mid}$ range: {:.3f} - {:.3f} m".format(
    min(r_middle_means), max(r_middle_means)))
print("$r_{global}$ range: {:.3f} - {:.3f} m".format(
    min(r_global_means), max(r_global_means)))