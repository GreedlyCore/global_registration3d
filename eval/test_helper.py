#!/usr/bin/env python3
"""Helper utilities for sweep/eval orchestration."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def build_scan2scan_config(
    base_cfg_path: str,
    generated_json_path: str,
    dataset: str,
    scene: str,
    dist_tag: str,
    out_cfg_path: str,
) -> str:
    """Build one config file with pre-generated scan2scan pairs."""
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

    cfg['test_type'] = 'scan2scan'
    cfg['test_scans'] = pairs
    cfg['test_count'] = len(pairs)

    out_dir = os.path.dirname(out_cfg_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    return out_cfg_path


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', required=True)
    parser.add_argument('--generated_json', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--dist_tag', required=True)
    parser.add_argument('--out_cfg', required=True)
    args = parser.parse_args()

    out = build_scan2scan_config(
        base_cfg_path=args.base_cfg,
        generated_json_path=args.generated_json,
        dataset=args.dataset,
        scene=args.scene,
        dist_tag=args.dist_tag,
        out_cfg_path=args.out_cfg,
    )
    print(f'[info] wrote scan2scan config: {out}')


if __name__ == '__main__':
    _main()
