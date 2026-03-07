import argparse
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Selected episodes from datasets:

python visualize/preview_gt.py --dataset nya_01 --start-id 1250 --end-id 2000
python visualize/preview_gt.py --dataset eee_03 --start-id 1250 --end-id 1500

python visualize/preview_gt.py --dataset nclt
# python visualize/preview_gt.py --dataset tum --start-id 100 --end-id 400
python visualize/preview_gt.py --dataset tum --start-id 100 --end-id 200
"""

# ---------------------------------------------------------------------------
# VIRAL datasets
# ---------------------------------------------------------------------------
_VIRAL_GT_BASE = os.path.expanduser('~/thesis/data/VIRAL/ntuviral_gt')
GT_PATHS = {
    "eee_03": os.path.join(_VIRAL_GT_BASE, "eee_03/ground_truth.csv"),
    "nya_01": os.path.join(_VIRAL_GT_BASE, "nya_01/ground_truth.csv"),
}

# ---------------------------------------------------------------------------
# NCLT dataset
# ---------------------------------------------------------------------------
_NCLT_BASE = os.path.expanduser('~/thesis/data/2013-01-10_vel/2013-01-10')
NCLT_GT_FILE  = os.path.join(_NCLT_BASE, 'groundtruth_2013-01-10.csv')
NCLT_COV_FILE = os.path.join(_NCLT_BASE, 'cov_2013-01-10.csv')

# ---------------------------------------------------------------------------
# TUM dataset
# ---------------------------------------------------------------------------
_TUM_BASE = os.path.expanduser(
    '~/thesis/sogmm_registration/data/rgbd_dataset_freiburg3_long_office_household'
)
TUM_ODOM_FILE = os.path.join(_TUM_BASE, 'odometry.pkl')

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLORS = {
    "eee_03": "royalblue",
    "nya_01": "tomato",
    "nclt":   "seagreen",
    "tum":    "darkorchid",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    choices=["eee_03", "nya_01", "nclt", "tum"],
                    required=True)
parser.add_argument("--start-id", type=int, default=0,
                    help="Start trajectory at this index (inclusive) — VIRAL only")
parser.add_argument("--end-id", type=int, default=None,
                    help="End trajectory at this index (inclusive) — VIRAL only")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
color = COLORS[args.dataset]

if args.dataset in ("eee_03", "nya_01"):
    # ---- VIRAL ----
    df = pd.read_csv(GT_PATHS[args.dataset])
    x = df["field.pose.position.x"].to_numpy()
    y = df["field.pose.position.y"].to_numpy()
    z = df["field.pose.position.z"].to_numpy()

    end = args.end_id + 1 if args.end_id is not None else len(x)
    x, y, z = x[args.start_id:end], y[args.start_id:end], z[args.start_id:end]

    print(f"{args.dataset}: {len(x)} poses  [{args.start_id} : {end - 1}]  "
          f"(total in file: {len(df)})")

    title = f"VIRAL GT — {args.dataset}  [{args.start_id} : {end - 1}]"

elif args.dataset == "nclt":
    # ---- NCLT ----
    # GT columns: timestamp, x(North), y(East), z(Down), roll, pitch, heading
    import scipy.interpolate
    gt  = np.loadtxt(NCLT_GT_FILE,  delimiter=',')
    cov = np.loadtxt(NCLT_COV_FILE, delimiter=',')

    t_cov = cov[:, 0]  # SLAM graph node timestamps

    # Interpolate GT to SLAM-graph timestamps (node-level precision)
    interp  = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:],
                                         kind='nearest', axis=0,
                                         bounds_error=False,
                                         fill_value='extrapolate')
    pose_gt = interp(t_cov)

    # NED: x=North, y=East, z=Down (z points down → negate for plotting)
    x = pose_gt[:, 0]   # North
    y = pose_gt[:, 1]   # East
    z = -pose_gt[:, 2]  # Up (negated)

    end = args.end_id + 1 if args.end_id is not None else len(x)
    x, y, z = x[args.start_id:end], y[args.start_id:end], z[args.start_id:end]

    print(f"nclt: {len(x)} SLAM-graph nodes  [{args.start_id} : {end - 1}]")
    title = f"NCLT GT — 2013-01-10  [{args.start_id} : {end - 1}]"

elif args.dataset == "tum":
    # ---- TUM (from odometry.pkl) ----
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    K, Tbc, odometry, pointclouds = pickle.load(open(TUM_ODOM_FILE, 'rb'))

    # positions: (3, N)  — already in world frame
    x = odometry.positions[0, :]
    y = odometry.positions[1, :]
    z = odometry.positions[2, :]

    end = args.end_id + 1 if args.end_id is not None else len(x)
    x, y, z = x[args.start_id:end], y[args.start_id:end], z[args.start_id:end]

    print(f"tum: {len(x)} odometry poses  [{args.start_id} : {end - 1}]")
    title = f"TUM GT — freiburg3_long_office  [{args.start_id} : {end - 1}]"

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, color=color, linewidth=1.2)
ax.scatter(x[0],  y[0],  z[0],  color=color, marker="o", s=60, zorder=5, label="start")
ax.scatter(x[-1], y[-1], z[-1], color=color, marker="^", s=60, zorder=5, label="end")

ax.set_title(title + "\n● start  ▲ end")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
plt.tight_layout()
plt.show()
