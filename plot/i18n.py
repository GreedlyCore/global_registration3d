import json
import os
from typing import Dict

DEFAULT_I18N_EN = {
    'title_success_rate': 'Success Rate',
    'title_average_inliers': 'Average Inlier Correspondences',
    'title_average_outliers': 'Average Outlier Correspondences',
    'title_ratio': 'Inlier / Outlier Correspondence Ratio',
    'title_average_rte': 'Average RTE',
    'title_average_rre': 'Average RRE',
    'title_pipeline_time': 'Average Pipeline Time by Method',
    'title_registration_time': 'Average Registration Time by Method',
    'xlabel_gt_bin_m': 'GT Distance Bin (m)',
    'ylabel_success_rate_percent': 'Success Rate (%)',
    'ylabel_average_inliers': 'Correspondences',
    'ylabel_average_outliers': 'False Correspondences',
    'ylabel_ratio': 'Correspondences / False Correspondences',
    'ylabel_rte_m': 'Average RTE (m)',
    'ylabel_rre_deg': 'Average RRE (deg)',
    'xlabel_time_s': 'Average Time (s)',
    'xlabel_reg_time_s': 'Average Registration Time (s)',
    'scene_prefix': 'Scene',
    'legend_downsampling': 'Point Cloud Resolution Reduction',
    'legend_features_creation': 'Features Creation',
    'legend_corr_creation': 'Correspondence Creation',
    'legend_registration': 'Registration',
}


def normalize_lang(lang: str) -> str:
    normalized = (lang or 'ENG').strip().upper()
    aliases = {
        'EN': 'ENG',
        'ENG': 'ENG',
        'RU': 'RUS',
        'RUS': 'RUS',
    }
    return aliases.get(normalized, 'ENG')


def default_i18n_path(base_dir: str) -> str:
    return os.path.join(base_dir, 'i18n_plot_labels.json')


def load_i18n_labels(i18n_path: str, lang: str) -> Dict[str, str]:
    labels = dict(DEFAULT_I18N_EN)
    if not i18n_path or not os.path.isfile(i18n_path):
        print(f'[warn] i18n file not found: {i18n_path}, using built-in ENG labels')
        return labels

    try:
        with open(i18n_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[warn] failed to load i18n file {i18n_path}: {exc}, using built-in ENG labels')
        return labels

    eng = data.get('ENG', {})
    if isinstance(eng, dict):
        labels.update({k: str(v) for k, v in eng.items()})

    lang_key = normalize_lang(lang)
    selected = data.get(lang_key, {})
    if isinstance(selected, dict):
        labels.update({k: str(v) for k, v in selected.items()})

    return labels


def tr(labels: Dict[str, str], key: str) -> str:
    return labels.get(key, DEFAULT_I18N_EN.get(key, key))
