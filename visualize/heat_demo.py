import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

VOXEL_SIZES = [0.1, 0.3, 0.5, 0.7]
ALPHA_VALS = [2.0, 2.5, 3.0, 3.5, 4.0]   # normal radius / voxel size
BETA_VALS = [4.0, 5.0, 6.0, 7.0, 8.0]    # FPFH radius / voxel size


def parameter_grid():
  """Return the canonical (voxel, alpha, beta) search grid used in Fig.7-style studies."""
  return VOXEL_SIZES.copy(), ALPHA_VALS.copy(), BETA_VALS.copy()


def draw_heatmap(ax, data, title, vmin, vmax, cbar_label, fmt,
    alpha_vals=None, beta_vals=None, highlight_best=False,
    cbar_ticks=None, cbar_tick_fmt=None, add_cbar=True):
  """Draw one heatmap with optional best-cell highlighting."""
  if alpha_vals is None:
    alpha_vals = ALPHA_VALS
  if beta_vals is None:
    beta_vals = BETA_VALS

  im = ax.imshow(data, vmin=vmin, vmax=vmax)

  ax.set_title(title, fontsize=11)
  ax.set_xticks(range(len(beta_vals)))
  ax.set_yticks(range(len(alpha_vals)))
  ax.set_xticklabels(beta_vals)
  ax.set_yticklabels(alpha_vals)

  ax.set_xlabel("Ratio of FPFH radius to $\\nu$")
  ax.set_ylabel("Ratio of normal radius to $\\nu$")

  # Annotate cells.
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8)

  if highlight_best:
    best_idx = np.unravel_index(np.nanargmax(data), data.shape)
    rect = Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
             fill=False, edgecolor='limegreen', linewidth=2.2, linestyle='--')
    ax.add_patch(rect)

  if add_cbar:
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_ticks is not None:
      cbar.set_ticks(cbar_ticks)
    if cbar_tick_fmt is not None:
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_tick_fmt))
    cbar.set_label(cbar_label)


def demo_random_plot():
  """
  Standalone demo plot using synthetic data
  usage, just run it by: python3 heat_overview_plot.py
  """
  np.random.seed(42)

  # Success rate: [0, 100] %.
  sr_data = [np.random.uniform(0, 100, (5, 5)) for _ in VOXEL_SIZES]
  # Runtime: [0.04, 0.15] sec.
  time_data = [np.random.uniform(0.04, 0.15, (5, 5)) for _ in VOXEL_SIZES]

  ncols = len(VOXEL_SIZES)
  fig, axes = plt.subplots(2, ncols, figsize=(4.6 * ncols, 8), squeeze=False)
  plt.subplots_adjust(wspace=0.5, hspace=0.2)

  for col, nu in enumerate(VOXEL_SIZES):
    draw_heatmap(
      ax=axes[0, col],
      data=sr_data[col],
      title=f"nu = {nu} m",
      vmin=0,
      vmax=100,
      cbar_label="Success rate [%]",
      fmt=".1f",
      highlight_best=True,
    )

  for col, nu in enumerate(VOXEL_SIZES):
    draw_heatmap(
      ax=axes[1, col],
      data=time_data[col],
      title=f"nu = {nu} m",
      vmin=0.04,
      vmax=0.15,
      cbar_label="Time [sec]",
      fmt=".3f",
    )

  plt.show()


if __name__ == "__main__":
  demo_random_plot()