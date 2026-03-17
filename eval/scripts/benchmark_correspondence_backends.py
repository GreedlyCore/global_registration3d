#!/usr/bin/env python3
"""Benchmark correspondence creation backends on random KITTI-04 scans.

Compares:
- kiss_matcher pybind find_correspondences (FLANN KD-tree)
- scipy.spatial.cKDTree mutual nearest-neighbor
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_kitti_scan_xyz(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32)
    pts = pts.reshape(-1, 4)[:, :3]
    return pts


def extract_fpfh_feats(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return np.asarray(fpfh.data, dtype=np.float32).T


def scipy_mutual_correspondences(feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batch query, all CPU cores (workers=-1)."""
    nns01 = cKDTree(feats1).query(feats0, k=1, workers=-1)[1]
    idx0 = np.arange(len(nns01), dtype=np.int64)
    nns10 = cKDTree(feats0).query(feats1, k=1, workers=-1)[1]
    mutual = (nns10[nns01] == idx0)
    return idx0[mutual], nns01[mutual].astype(np.int64)


def scipy_mutual_correspondences_single(feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batch query, single-threaded (workers=1) — isolates threading vs batching."""
    nns01 = cKDTree(feats1).query(feats0, k=1, workers=1)[1]
    idx0 = np.arange(len(nns01), dtype=np.int64)
    nns10 = cKDTree(feats0).query(feats1, k=1, workers=1)[1]
    mutual = (nns10[nns01] == idx0)
    return idx0[mutual], nns01[mutual].astype(np.int64)


def scipy_mutual_correspondences_pointbypoint(feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One kd-tree query per point — mirrors pybind's sequential access pattern.
    Uses a small random subset (500 pts) to keep runtime manageable."""
    rng = np.random.default_rng(0)
    sub0 = feats0[rng.choice(len(feats0), min(500, len(feats0)), replace=False)]
    sub1 = feats1[rng.choice(len(feats1), min(500, len(feats1)), replace=False)]
    tree1 = cKDTree(sub1)
    tree0 = cKDTree(sub0)
    nns01 = np.empty(len(sub0), dtype=np.int64)
    for i in range(len(sub0)):
        nns01[i] = tree1.query(sub0[i], k=1, workers=1)[1]
    nns10 = np.empty(len(sub1), dtype=np.int64)
    for j in range(len(sub1)):
        nns10[j] = tree0.query(sub1[j], k=1, workers=1)[1]
    idx0 = np.arange(len(nns01), dtype=np.int64)
    mutual = (nns10[nns01] == idx0)
    return idx0[mutual], nns01[mutual].astype(np.int64)


def get_pybind_correspondences_fn() -> Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Original pybind: sequential point-by-point FLANN queries, cores=1."""
    try:
        from kiss_matcher import find_correspondences as kiss_find_correspondences
    except Exception:
        return None

    def _run(feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx0, idx1 = kiss_find_correspondences(
            np.ascontiguousarray(feats0, dtype=np.float32),
            np.ascontiguousarray(feats1, dtype=np.float32),
            True,
            False,
            0,
        )
        return np.asarray(idx0, dtype=np.int64), np.asarray(idx1, dtype=np.int64)

    return _run


def get_pybind_parallel_fn() -> Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Parallelised pybind: fire the two independent directional NN queries on
    separate Python threads.

    The mutual-filter mutual-NN check requires two independent searches:
      (A) for every point in feats0, find its nearest neighbour in feats1
      (B) for every point in feats1, find its nearest neighbour in feats0
    (A) and (B) have zero data dependency on each other, so they can run in
    parallel.  pybind11 releases the GIL during C++ execution, so Python
    threads yield genuine concurrency here — no GIL contention.
    """
    try:
        from kiss_matcher import find_correspondences as _kiss_fn
    except Exception:
        return None

    from concurrent.futures import ThreadPoolExecutor

    def _nn_one_way(src_f: np.ndarray, tgt_f: np.ndarray) -> np.ndarray:
        """Return nns[i] = index of nearest tgt point for each src[i]."""
        f0c = np.ascontiguousarray(src_f, dtype=np.float32)
        f1c = np.ascontiguousarray(tgt_f, dtype=np.float32)
        idx0, idx1 = _kiss_fn(f0c, f1c, False, False, 0)  # mutual_filter=False
        # non-mutual: idx0 == arange(N), idx1 == nearest neighbour indices
        nns = np.empty(len(src_f), dtype=np.int64)
        nns[np.asarray(idx0, dtype=np.int64)] = np.asarray(idx1, dtype=np.int64)
        return nns

    def _run(feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_01 = ex.submit(_nn_one_way, feats0, feats1)  # (A)
            fut_10 = ex.submit(_nn_one_way, feats1, feats0)  # (B)
        nns01 = fut_01.result()
        nns10 = fut_10.result()
        idx0 = np.arange(len(feats0), dtype=np.int64)
        mutual = nns10[nns01] == idx0
        return idx0[mutual], nns01[mutual]

    return _run


def time_backend(
    fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    feats0: np.ndarray,
    feats1: np.ndarray,
    repeats: int,
    warmup: int,
) -> Tuple[float, float, int]:
    for _ in range(warmup):
        fn(feats0, feats1)

    times = []
    out_count = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        i0, i1 = fn(feats0, feats1)
        dt = time.perf_counter() - t0
        times.append(dt)
        out_count = min(len(i0), len(i1))

    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), int(out_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=Path, default=Path("data/KITTI/sequences/04/velodyne"))
    parser.add_argument("--voxel_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--src_scan", type=str, default="")
    parser.add_argument("--tgt_scan", type=str, default="")
    parser.add_argument("--backend", choices=["all", "scipy", "scipy_single", "scipy_pointbypoint", "pybind", "pybind_parallel"], default="all")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def pick_scan_pair(scan_paths: list[Path], seed: int, src_name: str, tgt_name: str) -> Tuple[Path, Path]:
    name_to_path = {p.name: p for p in scan_paths}
    if src_name and tgt_name:
        if src_name not in name_to_path or tgt_name not in name_to_path:
            raise RuntimeError("Provided --src_scan/--tgt_scan not found in kitti_root")
        return name_to_path[src_name], name_to_path[tgt_name]

    rng = random.Random(seed)
    return tuple(rng.sample(scan_paths, 2))


def build_features(args: argparse.Namespace) -> Tuple[Path, Path, np.ndarray, np.ndarray]:
    scan_paths = sorted(args.kitti_root.glob("*.bin"))
    if len(scan_paths) < 2:
        raise RuntimeError(f"Need at least two scans in {args.kitti_root}")

    src_path, tgt_path = pick_scan_pair(scan_paths, args.seed, args.src_scan, args.tgt_scan)

    src_xyz = load_kitti_scan_xyz(src_path)
    tgt_xyz = load_kitti_scan_xyz(tgt_path)

    feats0 = extract_fpfh_feats(src_xyz, args.voxel_size)
    feats1 = extract_fpfh_feats(tgt_xyz, args.voxel_size)
    return src_path, tgt_path, feats0, feats1


_SCIPY_FNS = {
    "scipy": scipy_mutual_correspondences,
    "scipy_single": scipy_mutual_correspondences_single,
    "scipy_pointbypoint": scipy_mutual_correspondences_pointbypoint,
}


_PYBIND_FN_GETTERS = {
    "pybind": get_pybind_correspondences_fn,
    "pybind_parallel": get_pybind_parallel_fn,
}


def run_backend(
    args: argparse.Namespace,
    backend: str,
    feats0: Optional[np.ndarray] = None,
    feats1: Optional[np.ndarray] = None,
    src_path: Optional[Path] = None,
    tgt_path: Optional[Path] = None,
    repeats_override: Optional[int] = None,
    warmup_override: Optional[int] = None,
) -> dict:
    if feats0 is None:
        src_path, tgt_path, feats0, feats1 = build_features(args)

    reps = repeats_override if repeats_override is not None else args.repeats
    wup  = warmup_override  if warmup_override  is not None else args.warmup

    out = {
        "src_scan": src_path.name,
        "tgt_scan": tgt_path.name,
        "n_src_feats": int(len(feats0)),
        "n_tgt_feats": int(len(feats1)),
        "backend": backend,
    }

    if backend in _SCIPY_FNS:
        mean_s, std_s, corr_count = time_backend(
            _SCIPY_FNS[backend], feats0, feats1, repeats=reps, warmup=wup
        )
        out.update({"ok": 1, "mean_s": mean_s, "std_s": std_s, "corr_count": corr_count})
        return out

    getter = _PYBIND_FN_GETTERS.get(backend)
    if getter is None:
        out.update({"ok": 0, "error": f"unknown_backend:{backend}"})
        return out

    fn = getter()
    if fn is None:
        out.update({"ok": 0, "error": "pybind_unavailable"})
        return out

    mean_s, std_s, corr_count = time_backend(fn, feats0, feats1, repeats=reps, warmup=wup)
    out.update({"ok": 1, "mean_s": mean_s, "std_s": std_s, "corr_count": corr_count})
    return out


def print_plain_result(prefix: str, result: dict) -> None:
    print(f"src_scan={result['src_scan']}")
    print(f"tgt_scan={result['tgt_scan']}")
    print(f"n_src_feats={result['n_src_feats']}")
    print(f"n_tgt_feats={result['n_tgt_feats']}")
    print(f"{prefix}_ok={int(result.get('ok', 0))}")
    if int(result.get("ok", 0)):
        print(f"{prefix}_mean_s={float(result['mean_s']):.6f}")
        print(f"{prefix}_std_s={float(result['std_s']):.6f}")
        print(f"{prefix}_corr_count={int(result['corr_count'])}")
    else:
        print(f"{prefix}_error={result.get('error', 'unknown')}")


def _run_subprocess_backend(backend: str, args: argparse.Namespace, src_name: str, tgt_name: str) -> dict:
    """Run a pybind backend in a fresh subprocess (avoids open3d/kiss import conflicts)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend", backend,
        "--json",
        "--kitti_root", str(args.kitti_root),
        "--voxel_size", str(args.voxel_size),
        "--seed", str(args.seed),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--src_scan", src_name,
        "--tgt_scan", tgt_name,
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"ok": 0, "error": f"exit={completed.returncode}", "stderr": completed.stderr[-400:]}
    payload = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        return json.loads(payload)
    except Exception:
        return {"ok": 0, "error": "invalid_json", "raw": payload[:200]}


def main() -> int:
    args = parse_args()

    if args.backend != "all":
        result = run_backend(args, args.backend)
        if args.json:
            print(json.dumps(result))
        else:
            print_plain_result(args.backend, result)
        return 0

    # --- "all" mode: build features once for scipy variants, then subprocess pybind ---
    src_path, tgt_path, feats0, feats1 = build_features(args)
    print(f"src_scan={src_path.name}  tgt_scan={tgt_path.name}")
    print(f"n_src_feats={len(feats0)}  n_tgt_feats={len(feats1)}")
    print()

    # scipy_single (batch, 1 thread) is ~8x slower than parallelised in 33D;
    # cap at 2 reps so we still get a data point without waiting forever.
    scipy_variants = [
        ("scipy  [batch, workers=-1]",  "scipy",        args.repeats, args.warmup),
        ("scipy  [batch, workers=1]",   "scipy_single", 2,            0),
    ]

    all_rows: list[Tuple[str, dict]] = []
    for label, key, reps, wup in scipy_variants:
        print(f"  running {label} ...", flush=True)
        r = run_backend(
            args, key,
            feats0=feats0, feats1=feats1,
            src_path=src_path, tgt_path=tgt_path,
            repeats_override=reps, warmup_override=wup,
        )
        all_rows.append((label, r))
        if r.get("ok"):
            print(f"  {label:38s}  mean={r['mean_s']:.4f}s  std={r['std_s']:.4f}s  corr={r['corr_count']}")
        else:
            print(f"  {label:38s}  ERROR: {r.get('error')}")

    # pybind variants run in subprocesses — avoids open3d/kiss_matcher import conflict
    pybind_variants = [
        ("pybind [FLANN, cores=1, sequential]",  "pybind"),
        ("pybind [FLANN, cores=1, 2 threads]",   "pybind_parallel"),
    ]
    for label, key in pybind_variants:
        print(f"  running {label} ...", flush=True)
        r = _run_subprocess_backend(key, args, src_path.name, tgt_path.name)
        all_rows.append((label, r))
        if r.get("ok"):
            print(f"  {label:38s}  mean={r['mean_s']:.4f}s  std={r['std_s']:.4f}s  corr={r['corr_count']}")
        else:
            print(f"  {label:38s}  ERROR: {r.get('error')}")

    # --- Summary table ---
    print()
    print("=" * 76)
    print("DIAGNOSIS SUMMARY")
    print("=" * 76)
    ref: Optional[float] = None
    for label, r in all_rows:
        if not r.get("ok"):
            print(f"  {label:45s}  FAILED")
            continue
        if ref is None:
            ref = r["mean_s"]
        speedup = ref / r["mean_s"]
        print(f"  {label:45s}  {r['mean_s']*1000:8.1f} ms   x{speedup:.2f} vs baseline")
    print()
    print("WHAT EACH COMPARISON ISOLATES:")
    print("  scipy[workers=-1]  vs  scipy[workers=1]     →  threading gain (scipy)")
    print("  scipy[workers=1]   vs  pybind[sequential]   →  library quality (FLANN vs cKDTree)")
    print("  pybind[sequential] vs  pybind[2-threads]    →  threading gain (pybind/FLANN)")
    print("  scipy[workers=-1]  vs  pybind[2-threads]    →  remaining gap (batching + N-cores)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
