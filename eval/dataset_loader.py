#!/usr/bin/env python
"""
Datasets (KITTI/NCLT/MulRan) loading utilities.
"""
import os
import numpy as np
import open3d as o3d
import scipy.interpolate

# TODO: SETUP this according your datasets destination folder or create symlinks
KITTI_DIR = os.path.expanduser("~/thesis/global_registration3d/data/KITTI")
NCLT_DIR  = os.path.expanduser("~/thesis/global_registration3d/data/NCLT")
MULRAN_DIR = os.path.expanduser("~/thesis/global_registration3d/data/MulRan")

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


# ─────────────────────────────── MulRan ───────────────────────────────────── #

def load_mulran_ouster(filepath):
    """
    Load one MulRan Ouster scan from a KITTI-style .bin file.

    Returns:
        Nx4 float32 array (x, y, z, intensity)
    """
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)


def load_mulran_ouster_pcd(filepath):
    """Load one MulRan Ouster .bin file as an open3d PointCloud."""
    pts = load_mulran_ouster(filepath)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def _mulran_base_to_ouster_transform():
    """
    Static transform from MulRan base_link to Ouster.

    Translation is in metres. Rotation is from roll/pitch/yaw in degrees.
    """
    tx, ty, tz = 1.7042, -0.021, 1.8047
    roll_deg, pitch_deg, yaw_deg = 0.0001, 0.0003, 179.6654

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [tx, ty, tz]
    return T


def _load_mulran_global_poses(global_pose_file):
    """
    Load MulRan global_pose.csv.

    Returns:
        pose_ts: (N,) int64 timestamps
        poses_world_base: list of 4x4 world->base transforms
    """
    pose_rows = np.loadtxt(global_pose_file, delimiter=',', dtype=np.float64)
    if pose_rows.ndim == 1:
        pose_rows = pose_rows[None, :]
    if pose_rows.shape[1] != 13:
        raise ValueError(
            f'Expected 13 columns in global_pose.csv, got {pose_rows.shape[1]}')

    pose_ts = pose_rows[:, 0].astype(np.int64)
    poses_world_base = []
    for row in pose_rows:
        T = np.eye(4)
        T[:3, :] = row[1:].reshape(3, 4)
        poses_world_base.append(T)
    return pose_ts, poses_world_base


def _load_mulran_ouster_timestamps(data_stamp_file):
    """Load Ouster timestamps from data_stamp.csv."""
    ouster_ts = []
    with open(data_stamp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_str, sensor = line.split(',', 1)
            if sensor == 'ouster':
                ouster_ts.append(int(ts_str))
    return np.array(ouster_ts, dtype=np.int64)


def load_mulran_ground_truth(seq, mulran_dir):
    """
    Load MulRan GT poses, one per Ouster scan.

    global_pose.csv contains timestamp + 3x4 world->base pose. We convert that
    to world->ouster using the static base_link->ouster transform.

    Args:
        seq: sequence name, e.g. 'DCC02'
        mulran_dir: root MulRan dir containing '<seq>/'

    Returns:
        poses: list of 4x4 world->ouster transforms
        scan_files: list of Ouster .bin paths sorted by timestamp
    """
    seq_dir = os.path.join(os.path.expanduser(mulran_dir), seq)
    global_pose_file = os.path.join(seq_dir, 'global_pose.csv')
    data_stamp_file = os.path.join(seq_dir, 'data_stamp.csv')
    ouster_dir = os.path.join(seq_dir, 'Ouster')

    if not os.path.exists(global_pose_file):
        raise FileNotFoundError(f'global_pose.csv not found: {global_pose_file}')
    if not os.path.exists(data_stamp_file):
        raise FileNotFoundError(f'data_stamp.csv not found: {data_stamp_file}')
    if not os.path.isdir(ouster_dir):
        raise FileNotFoundError(f'Ouster dir not found: {ouster_dir}')

    pose_ts, poses_world_base = _load_mulran_global_poses(global_pose_file)
    ouster_ts_from_csv = set(_load_mulran_ouster_timestamps(data_stamp_file).tolist())

    scan_files = sorted(
        [os.path.join(ouster_dir, f) for f in os.listdir(ouster_dir) if f.endswith('.bin')]
    )
    scan_ts = np.array(
        [int(os.path.splitext(os.path.basename(f))[0]) for f in scan_files],
        dtype=np.int64)

    if ouster_ts_from_csv:
        keep_mask = np.array([ts in ouster_ts_from_csv for ts in scan_ts], dtype=bool)
        scan_ts = scan_ts[keep_mask]
        scan_files = [f for f, keep in zip(scan_files, keep_mask) if keep]

    if len(scan_files) == 0:
        raise RuntimeError(f'No MulRan Ouster scans found for seq {seq}')

    T_base_ouster = _mulran_base_to_ouster_transform()
    poses_world_ouster = []
    pose_ts_arr = np.asarray(pose_ts)
    for ts in scan_ts:
        nearest_idx = int(np.argmin(np.abs(pose_ts_arr - ts)))
        poses_world_ouster.append(poses_world_base[nearest_idx] @ T_base_ouster)

    print(f'Loaded {len(poses_world_ouster)} MulRan poses for seq {seq}')
    return poses_world_ouster, scan_files


def load_mulran_dataset(seq):
    """
    High-level loader for a MulRan sequence.

    Returns (scan_files, poses, eye(4)) — Tr=eye(4) because poses are converted
    to the Ouster sensor frame.
    """
    poses, scan_files = load_mulran_ground_truth(seq, MULRAN_DIR)
    return scan_files, poses, np.eye(4)
