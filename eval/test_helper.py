#!/usr/bin/env python3
"""Helper utilities for sweep/eval orchestration."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _build_scan2map_entries_from_pairs(
    pairs: List[List[int]],
    map_prev_scans: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src_idx, tgt_idx in pairs:
        start = max(0, int(src_idx) - int(map_prev_scans))
        src_map = list(range(start, int(src_idx) + 1))
        out.append(
            {
                'src': int(src_idx),
                'tgt': int(tgt_idx),
                'src_map': src_map,
            }
        )
    return out


def build_eval_config(
    base_cfg_path: str,
    generated_json_path: str,
    dataset: str,
    scene: str,
    dist_tag: str,
    out_cfg_path: str,
    mode: str = 'scan2scan',
    map_prev_scans: int = 5,
) -> str:
    """Build one config file with pre-generated eval lists."""
    with open(base_cfg_path) as f:
        cfg: Dict[str, Any] = json.load(f)

    with open(generated_json_path) as f:
        generated: Dict[str, Any] = json.load(f)

    try:
        pairs: List[List[int]] = generated['pairs'][dataset][scene][dist_tag]
    except Exception as exc:
        raise ValueError(
            f'[error] Missing generated pairs for dataset={dataset} scene={scene} dist_tag={dist_tag}: {exc}'
        ) from exc

    if mode not in ('scan2scan', 'scan2map'):
        raise ValueError(f'[error] Unknown mode: {mode}')

    cfg['test_type'] = mode
    cfg['test_scans'] = pairs
    cfg['test_count'] = len(pairs)

    if mode == 'scan2map':
        scan2map_entries = None
        try:
            scan2map_entries = generated['scan2map'][dataset][scene][dist_tag]
        except Exception:
            scan2map_entries = None

        if scan2map_entries is None:
            scan2map_entries = _build_scan2map_entries_from_pairs(
                pairs=pairs,
                map_prev_scans=map_prev_scans,
            )

        cfg['map_prev_scans'] = int(map_prev_scans)
        cfg['test_scan2map'] = scan2map_entries

    out_dir = os.path.dirname(out_cfg_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    return out_cfg_path


def build_scan2scan_config(
    base_cfg_path: str,
    generated_json_path: str,
    dataset: str,
    scene: str,
    dist_tag: str,
    out_cfg_path: str,
) -> str:
    """Backward-compatible wrapper for scan2scan config generation."""
    return build_eval_config(
        base_cfg_path=base_cfg_path,
        generated_json_path=generated_json_path,
        dataset=dataset,
        scene=scene,
        dist_tag=dist_tag,
        out_cfg_path=out_cfg_path,
        mode='scan2scan',
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', required=True)
    parser.add_argument('--generated_json', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--dist_tag', required=True)
    parser.add_argument('--out_cfg', required=True)
    parser.add_argument('--mode', default='scan2scan', choices=('scan2scan', 'scan2map'))
    parser.add_argument('--map_prev_scans', type=int, default=5)
    args = parser.parse_args()

    out = build_eval_config(
        base_cfg_path=args.base_cfg,
        generated_json_path=args.generated_json,
        dataset=args.dataset,
        scene=args.scene,
        dist_tag=args.dist_tag,
        out_cfg_path=args.out_cfg,
        mode=args.mode,
        map_prev_scans=args.map_prev_scans,
    )
    print(f'[info] wrote {args.mode} config: {out}')


if __name__ == '__main__':
    _main()
