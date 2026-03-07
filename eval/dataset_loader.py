#!/usr/bin/env python
"""
Datasets (KITTI/NCLT) loading utilities.
"""
import os
import numpy as np
import open3d as o3d
import scipy.interpolate

# SETUP according your data destination folder or create symlinks. #TODO how to refactor and handle this properly ???
KITTI_DIR = "/home/sonieth3/thesis/investigate/data/KITTI"
NCLT_DIR  = "/home/sonieth3/thesis/investigate/data/NCLT"


def load_kitti_velodyne(filepath):
    """
    Load KITTI velodyne point cloud from .bin file.

    Args:
        filepath: Path to .bin file

    Returns:
        Nx4 array (x, y, z, reflectance)
    """
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)


def load_kitti_velodyne_pcd(filepath):
    """
    Load KITTI velodyne .bin file and return an open3d PointCloud (xyz only).
    """
    pts = load_kitti_velodyne(filepath)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def load_kitti_poses(pose_file):

    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T_full = np.eye(4)
            T_full[0:3, :] = T
            poses.append(T_full)
    return poses


def load_kitti_calib(calib_file):

    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.split(':', 1)
                calib[key] = np.fromstring(value, dtype=np.float64, sep=' ')

    # Tr is the transformation from velodyne to camera 0
    Tr = calib['Tr'].reshape(3, 4)
    Tr_full = np.eye(4)
    Tr_full[0:3, :] = Tr

    return Tr_full


def load_kitti_ground_truth(sequence, kitti_dir, start_idx=0, end_idx=None):
    """
    Load KITTI ground truth poses in velodyne frame.

    Args:
        sequence: KITTI sequence number (e.g., '04')
        kitti_dir: Root directory of KITTI dataset
        start_idx: First scan index
        end_idx: Last scan index (exclusive)

    Returns:
        poses: List of 4x4 numpy arrays (velodyne frame poses)
        Tr: Velodyne to camera calibration matrix
    """
    kitti_dir = os.path.expanduser(kitti_dir)

    pose_file = os.path.join(kitti_dir, 'poses', f'{sequence}.txt')
    calib_file = os.path.join(kitti_dir, 'sequences', sequence, 'calib.txt')

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f'Pose file not found: {pose_file}')
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f'Calibration file not found: {calib_file}')

    # Load camera poses and calibration
    poses_cam = load_kitti_poses(pose_file)
    Tr = load_kitti_calib(calib_file)

    # Slice poses if needed
    if end_idx is not None:
        poses_cam = poses_cam[start_idx:end_idx + 1]
    else:
        poses_cam = poses_cam[start_idx:]

    print(f'Loaded {len(poses_cam)} KITTI poses')

    return poses_cam, Tr


def load_kitti_dataset(seq):
    """
    High-level loader for a KITTI sequence.
    Returns (scan_files, poses_cam, Tr) — same interface as load_nclt_dataset.
    """
    velodyne_dir = os.path.join(KITTI_DIR, 'sequences', seq, 'velodyne')
    if not os.path.isdir(velodyne_dir):
        raise FileNotFoundError(f'Velodyne dir not found: {velodyne_dir}')

    scan_files = sorted(
        [os.path.join(velodyne_dir, f)
         for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    if not scan_files:
        raise RuntimeError(f'No .bin files in {velodyne_dir}')

    poses_cam, Tr = load_kitti_ground_truth(seq, KITTI_DIR)
    assert len(poses_cam) == len(scan_files), (
        f'Pose count {len(poses_cam)} != scan count {len(scan_files)}')

    return scan_files, poses_cam, Tr


# ──────────────────────────────── NCLT ────────────────────────────────────── #

_NCLT_DTYPE  = np.dtype([('x', np.uint16), ('y', np.uint16), ('z', np.uint16),
                          ('i', np.uint8),  ('l', np.uint8)])
_NCLT_SCALE  = 0.005     # metres per uint16 unit
_NCLT_OFFSET = 32768.0   # unsigned zero-point


def load_nclt_velodyne(filepath):
    """Load one NCLT velodyne_sync/*.bin scan. Returns (N,3) float32, metres, sensor frame."""
    raw = np.fromfile(filepath, dtype=_NCLT_DTYPE)
    xyz = np.stack([
        (raw['x'].astype(np.float32) - _NCLT_OFFSET) * _NCLT_SCALE,
        (raw['y'].astype(np.float32) - _NCLT_OFFSET) * _NCLT_SCALE,
        (raw['z'].astype(np.float32) - _NCLT_OFFSET) * _NCLT_SCALE,
    ], axis=1)
    return xyz


def load_nclt_velodyne_pcd(filepath):
    """Load one NCLT velodyne_sync/*.bin scan as an open3d PointCloud."""
    pts = load_nclt_velodyne(filepath)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def _euler_to_rot(roll, pitch, heading):
    """NED Euler angles → 3×3 rotation matrix (body → world). R = Rz(h) Ry(p) Rx(r)."""
    cr, sr = np.cos(roll),    np.sin(roll)
    cp, sp = np.cos(pitch),   np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    Rx = np.array([[1,  0,   0 ], [0,  cr, -sr], [0,  sr,  cr]])
    Ry = np.array([[cp, 0,  sp ], [0,   1,   0 ], [-sp, 0,  cp]])
    Rz = np.array([[ch, -sh, 0 ], [sh,  ch,  0 ], [0,   0,   1]])
    return Rz @ Ry @ Rx


def load_nclt_ground_truth(seq, nclt_dir):
    """
    Load NCLT GT poses, one per velodyne_sync scan (matched by timestamp).

    Args:
        seq:      sequence name, e.g. '2013-01-10'
        nclt_dir: root NCLT dir containing '<seq>_vel/<seq>/'

    Returns:
        poses      : list of 4×4 arrays (NED world frame, body→world)
        scan_files : list of .bin paths sorted by timestamp, same length
    """
    seq_dir  = os.path.join(os.path.expanduser(nclt_dir), f'{seq}_vel', seq)
    gt_file  = os.path.join(seq_dir, f'groundtruth_{seq}.csv')
    sync_dir = os.path.join(seq_dir, 'velodyne_sync')

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f'GT file not found: {gt_file}')
    if not os.path.isdir(sync_dir):
        raise FileNotFoundError(f'velodyne_sync dir not found: {sync_dir}')

    # GT columns: timestamp, x(North), y(East), z(Down), roll, pitch, heading
    gt = np.loadtxt(gt_file, delimiter=',')
    interp = scipy.interpolate.interp1d(
        gt[:, 0], gt[:, 1:], kind='nearest', axis=0,
        bounds_error=False, fill_value='extrapolate')

    # Scan files sorted by timestamp embedded in filename
    scan_files = sorted(
        [os.path.join(sync_dir, f) for f in os.listdir(sync_dir) if f.endswith('.bin')])
    scan_ts = np.array(
        [int(os.path.splitext(os.path.basename(f))[0]) for f in scan_files],
        dtype=np.float64)

    pose_vals = interp(scan_ts)  # (N, 6)
    poses = []
    for x, y, z, roll, pitch, heading in pose_vals:
        T = np.eye(4)
        T[:3, :3] = _euler_to_rot(roll, pitch, heading)
        T[:3,  3] = [x, y, z]
        poses.append(T)

    print(f'Loaded {len(poses)} NCLT poses for seq {seq}')
    return poses, scan_files


def load_nclt_dataset(seq):
    """
    High-level loader for an NCLT sequence.
    Returns (scan_files, poses, eye(4)) — Tr=eye(4) since poses are already in sensor frame.
    """
    poses, scan_files = load_nclt_ground_truth(seq, NCLT_DIR)
    return scan_files, poses, np.eye(4)
