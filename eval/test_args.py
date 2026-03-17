"""Argument parsing helpers for `eval/test.py`."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence, Tuple

from helpers import resolve_feature_cfg


FEAT_METHOD_CHOICES = ("FPFH", "FPFH_PCL", "FasterPFH", "SHOT_PCL")
REG_METHOD_CHOICES = ("teaser", "mac", "quatro", "kiss")
TEST_TYPE_CHOICES = ("random", "scan2scan")


def load_config_from_argv(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Read the JSON config path in a minimal first pass."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    pre_args, _ = pre.parse_known_args(argv)

    with open(pre_args.config) as f:
        return json.load(f)


def build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the full parser using JSON values as defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=cfg.get("dataset"))
    parser.add_argument("--seq", default=str(cfg.get("seq", "")))
    parser.add_argument("--dist_min", type=float, default=cfg.get("dist_min"))
    parser.add_argument("--dist_max", type=float, default=cfg.get("dist_max"))
    parser.add_argument("--test_count", type=int, default=cfg.get("test_count", 100))
    parser.add_argument(
        "--feat",
        default=cfg.get("feat", "FPFH"),
        choices=FEAT_METHOD_CHOICES,
    )
    parser.add_argument(
        "--reg",
        default=cfg.get("reg", "teaser"),
        choices=REG_METHOD_CHOICES,
    )
    parser.add_argument("--voxel_size", type=float, default=cfg.get("voxel_size", 0.5))
    parser.add_argument("--rnormal", type=float, default=None)
    parser.add_argument("--rFPFH", type=float, default=None)
    parser.add_argument("--re_thre", type=float, default=cfg.get("re_thre", 5.0))
    parser.add_argument("--te_thre", type=float, default=cfg.get("te_thre", 2.0))
    parser.add_argument("--out_dir", default=cfg.get("out_dir", "results"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument(
        "--test_type",
        default=cfg.get("test_type", "random"),
        choices=TEST_TYPE_CHOICES,
    )
    return parser


def finalize_args(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Attach nested config blocks and validate cross-argument requirements."""
    args.teaser = cfg.get("teaser", {})
    args.mac = cfg.get("mac", {})
    args.quatro = cfg.get("quatro", {})
    args.feat_cfg = resolve_feature_cfg(cfg, args.feat)

    if args.rnormal is not None:
        args.feat_cfg["rnormal"] = float(args.rnormal)
    if args.rFPFH is not None:
        args.feat_cfg["rFPFH"] = float(args.rFPFH)

    args.test_scans = cfg.get("test_scans", [])

    if args.test_type == "random" and (args.dist_min is None or args.dist_max is None):
        parser.error("random mode requires both --dist_min and --dist_max (in meters)")

    return args


def parse_test_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, Dict[str, Any], argparse.ArgumentParser]:
    """Parse eval CLI args with JSON defaults and CLI overrides."""
    cfg = load_config_from_argv(argv)
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    return finalize_args(args, cfg, parser), cfg, parser