#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import Dict, Iterable

import numpy as np
import pandas as pd


METHOD_LABEL = {
    'quatro': 'QUATRO',
    'mac': 'MAC',
    'macpp': 'MAC++',
    'kiss': 'KISS-Matcher',
    'teaser': 'TEASER++',
    'trde': 'TR-DE',
    'gmor': 'GMOR',
}

METHOD_ORDER = ['trde', 'gmor', 'quatro', 'mac', 'macpp', 'kiss', 'teaser']


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def read_csv_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def canonical_method_name(method: object) -> str:
    raw = str(method).strip().lower()
    aliases = {
        'quatro': 'quatro',
        'mac': 'mac',
        'mac++': 'macpp',
        'macpp': 'macpp',
        'kiss': 'kiss',
        'kiss-matcher': 'kiss',
        'teaser': 'teaser',
        'teaser++': 'teaser',
        'tr-de': 'trde',
        'trde': 'trde',
        'gmor': 'gmor',
    }
    return aliases.get(raw, raw)


def method_display_name(method: object) -> str:
    canonical = canonical_method_name(method)
    return METHOD_LABEL.get(canonical, str(method).strip())


def method_feat_label(method: object, feat: object) -> str:
    method_txt = method_display_name(method)
    feat_txt = str(feat).strip()
    if feat_txt:
        return f'{method_txt}+{feat_txt}'
    return method_txt


def dist_tag_from_row(row: pd.Series) -> str:
    dist_tag = str(row.get('dist_tag', '')).strip()
    if dist_tag and dist_tag.lower() != 'nan':
        return normalize_dist_tag(dist_tag)

    dmin = safe_float(row.get('dist_min', np.nan))
    dmax = safe_float(row.get('dist_max', np.nan))
    if np.isfinite(dmin) and np.isfinite(dmax):
        return normalize_dist_tag(f'{int(dmin)}_{int(dmax)}')

    return ''


def normalize_dist_tag(tag: object) -> str:
    cleaned = str(tag).strip().strip('[]').replace('-', '_').replace('~', '_')
    m = re.match(r'^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$', cleaned)
    if not m:
        return str(tag).strip()
    lo = m.group(1).rstrip('0').rstrip('.') if '.' in m.group(1) else m.group(1)
    hi = m.group(2).rstrip('0').rstrip('.') if '.' in m.group(2) else m.group(2)
    return f'{lo}_{hi}'


def dist_label(tag: object) -> str:
    norm = normalize_dist_tag(tag)
    parts = norm.split('_', 1)
    if len(parts) == 2:
        return f'{parts[0]}~{parts[1]}m'
    return str(tag)


def dist_sort_key(tag: object):
    norm = normalize_dist_tag(tag)
    parts = norm.split('_', 1)
    if len(parts) != 2:
        return (float('inf'), float('inf'), str(tag))
    try:
        return (float(parts[0]), float(parts[1]), norm)
    except ValueError:
        return (float('inf'), float('inf'), norm)


def scene_sort_key(scene: object):
    txt = str(scene).strip()
    if txt.isdigit():
        return (0, f'{int(txt):04d}')
    return (1, txt)


def default_output_dir(input_csv: str) -> str:
    return os.path.join(os.path.dirname(input_csv) or '.', 'plots')


def output_path(output_dir: str, stem: str, suffix: str = '.txt') -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'{stem}{suffix}')


def analyze_run_csv(run_csv: str) -> Dict[str, float]:
    stats = {
        'rre_deg': float('nan'),
        'rte_m': float('nan'),
        'time_s': float('nan'),
        'n_corr_init': float('nan'),
        'n_inliers': float('nan'),
        'n_outliers': float('nan'),
    }
    if not run_csv or not os.path.isfile(run_csv):
        return stats

    df = pd.read_csv(run_csv)
    if df.empty:
        return stats

    pair_df = df[df.get('pair_id', '').astype(str) != 'SUMMARY'].copy() if 'pair_id' in df.columns else df.copy()
    if pair_df.empty:
        return stats

    succ_col = None
    for key in ('success', 'is_success', 'succ'):
        if key in pair_df.columns:
            succ_col = key
            break

    if succ_col is not None:
        succ_df = pair_df[pd.to_numeric(pair_df[succ_col], errors='coerce') > 0]
    else:
        succ_df = pair_df

    if not succ_df.empty:
        if 'RE_deg' in succ_df.columns:
            vals = pd.to_numeric(succ_df['RE_deg'], errors='coerce').dropna()
            if not vals.empty:
                stats['rre_deg'] = float(vals.mean())
        if 'TE_m' in succ_df.columns:
            vals = pd.to_numeric(succ_df['TE_m'], errors='coerce').dropna()
            if not vals.empty:
                stats['rte_m'] = float(vals.mean())

    if 'total_time_s' in pair_df.columns:
        vals = pd.to_numeric(pair_df['total_time_s'], errors='coerce').dropna()
        if not vals.empty:
            stats['time_s'] = float(vals.mean())

    # Prefer SUMMARY row for correspondence counts when present.
    summary_df = pd.DataFrame()
    if 'pair_id' in df.columns:
        summary_df = df[df['pair_id'].astype(str) == 'SUMMARY']

    for key in ('n_corr_init', 'n_inliers', 'n_outliers'):
        if key in pair_df.columns:
            vals = pd.to_numeric(pair_df[key], errors='coerce').dropna()
            if not vals.empty:
                stats[key] = float(vals.mean())
        if (not summary_df.empty) and (key in summary_df.columns):
            svals = pd.to_numeric(summary_df[key], errors='coerce').dropna()
            if not svals.empty:
                stats[key] = float(svals.iloc[0])

    return stats


def build_run_cache(csv_paths: Iterable[str]) -> Dict[str, Dict[str, float]]:
    cache: Dict[str, Dict[str, float]] = {}
    for path in csv_paths:
        clean = str(path).strip()
        if not clean or clean in cache:
            continue
        cache[clean] = analyze_run_csv(clean)
    return cache
