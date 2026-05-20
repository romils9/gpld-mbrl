#!/usr/bin/env python3
"""
Compare Lipschitz constants between two experiment configurations (e.g., baseline vs Lipschitz-regularized).

This script:
1. Reads Lipschitz summary data from two experiment folders
2. Plots comparison of mean Lipschitz constants across training
3. Supports both individual seed comparisons and aggregation across seeds

Usage:
    # Compare two specific experiments
    python compare_lipschitz.py --baseline <baseline_folder> --lipschitz <lipschitz_folder>

    # Compare all experiments for a game (aggregated across seeds)
    python compare_lipschitz.py --game walker_walk --base_dir /path/to/experiments

    # Compare with specific output directory
    python compare_lipschitz.py --baseline <folder1> --lipschitz <folder2> --output_dir ./comparison_plots
"""

import os
import sys
import argparse
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re
import math

# ============================================================================
# STYLE SETTINGS (matching plot_2_sir.py)
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
TARGET_PIXELS = 2400
SAVE_DPI = 300
FIGSIZE = (TARGET_PIXELS / SAVE_DPI, TARGET_PIXELS / SAVE_DPI)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': FIGSIZE,
    'figure.dpi': SAVE_DPI,
})

# Color scheme matching plot_2_sir.py
COLOR_BASELINE = '#000000'  # Black
COLOR_LIPSCHITZ = '#1f77b4'  # Blue

# Labels matching plot_2_sir.py
LABEL_BASELINE = "DreamerV3"
LABEL_LIPSCHITZ = "GPLD"
X_LABEL = "Environment steps"
PROPRIO_XMAX = 500032

METRICS = [
    ('lipschitz_prior_mean', 'prior_mean', 'Prior KL mean'),
    ('lipschitz_prior_max', 'prior_max', 'Prior KL max'),
    ('lipschitz_post_e_mean', 'post_e_mean', 'Posterior KL mean'),
    ('lipschitz_dynamics_mean', 'dynamics_mean', 'Dynamics L2 mean'),
]


def parse_formats(formats):
    if isinstance(formats, str):
        return [fmt.strip().lower() for fmt in formats.split(',') if fmt.strip()]
    return list(formats)


def save_figure(fig, output_path, formats=('png', 'pdf'), png_dpi=SAVE_DPI):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    saved = []
    for fmt in parse_formats(formats):
        path = output_path.with_suffix(f'.{fmt}')
        dpi = png_dpi if fmt == 'png' else SAVE_DPI
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        saved.append(path)
    return saved


def format_game_name(game_name):
    return game_name.replace('_', '-')


def format_steps_axis(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1_000_000:.1f}M' if abs(x) >= 1_000_000 else f'{x/1000:.0f}K'
    ))


def style_axis(ax, ylabel, show_xlabel=True, show_ylabel=True, xmax=None):
    ax.set_xlabel(X_LABEL if show_xlabel else '')
    ax.set_ylabel(ylabel if show_ylabel else '')
    if xmax is None:
        xmax = PROPRIO_XMAX
    if xmax:
        ax.set_xlim(0, xmax)
    format_steps_axis(ax)
    ax.grid(True, alpha=0.3)


def plot_curve(
    ax,
    steps,
    mean,
    std,
    color,
    label,
    linestyle='-',
    marker=None,
    show_std=True,
    linewidth=2.0,
):
    steps = np.asarray(steps, dtype=float)
    mean = np.asarray(mean, dtype=float)
    valid = np.isfinite(steps) & np.isfinite(mean)
    if not np.any(valid):
        return
    steps = steps[valid]
    mean = mean[valid]
    markevery = max(1, len(steps) // 8)
    ax.plot(
        steps,
        mean,
        linestyle=linestyle,
        color=color,
        label=label,
        linewidth=linewidth,
        marker=marker,
        markersize=5,
        markevery=markevery,
        alpha=0.95,
    )
    if show_std and std is not None:
        std = np.asarray(std, dtype=float)[valid]
        lower = mean - std
        upper = mean + std
        ax.fill_between(steps, lower, upper, color=color, alpha=0.12, linewidth=0)


def aggregate_runs(run_data, metric_key, common_steps=None):
    valid_runs = [
        data for data in run_data
        if len(data.get('steps', [])) and len(data.get(metric_key, []))
    ]
    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    if common_steps is None:
        common_steps = sorted({int(step) for data in valid_runs for step in data['steps']})
    common_steps = np.asarray(common_steps, dtype=float)

    values = []
    for data in valid_runs:
        steps = np.asarray(data['steps'], dtype=float)
        metric_values = np.asarray(data[metric_key], dtype=float)
        mask = np.isfinite(steps) & np.isfinite(metric_values)
        if np.count_nonzero(mask) < 1:
            continue
        interp = np.interp(common_steps, steps[mask], metric_values[mask])
        outside = (common_steps < np.nanmin(steps[mask])) | (common_steps > np.nanmax(steps[mask]))
        interp[outside] = np.nan
        values.append(interp)

    if not values:
        return common_steps, np.array([]), np.array([])

    values = np.asarray(values, dtype=float)
    return common_steps, np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def _as_direct_dict(obj):
    """Unwrap numpy object arrays saved by np.savez."""
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return obj.item()
    return obj


def _current_schema_values(direct, eps_idx, mean_key, per_row_key=None, value_key=None):
    """Read metrics saved by analyze_smoothness_2.py."""
    if mean_key not in direct:
        return np.nan, np.nan

    mean_values = np.asarray(direct[mean_key])
    mean = float(mean_values[eps_idx])

    max_value = np.nan
    if per_row_key and per_row_key in direct:
        per_row = np.asarray(direct[per_row_key])[eps_idx]
        # KL per-row arrays are [batch, stoch]; old "max" used total KL per sample.
        values = np.sum(per_row, axis=-1) if per_row.ndim >= 2 else per_row
        max_value = float(np.nanmax(values))
    elif value_key and value_key in direct:
        values = np.asarray(direct[value_key])[eps_idx]
        max_value = float(np.nanmax(values))

    return mean, max_value


def _old_schema_values(direct, eps_idx, key):
    """Read metrics saved by older analyze_smoothness.py variants."""
    values = direct.get(key, [])
    if values and len(values) > 0:
        return values[eps_idx].get('mean', np.nan), values[eps_idx].get('max', np.nan)
    return np.nan, np.nan


def extract_metrics_from_direct(direct, eps_idx=-1):
    """Extract comparison metrics from either supported smoothness result schema."""
    direct = _as_direct_dict(direct)
    if not isinstance(direct, dict):
        return {
            'prior_mean': np.nan,
            'prior_max': np.nan,
            'post_h_mean': np.nan,
            'post_h_max': np.nan,
            'post_e_mean': np.nan,
            'post_e_max': np.nan,
            'dynamics_mean': np.nan,
            'dynamics_max': np.nan,
        }

    prior_mean, prior_max = _old_schema_values(direct, eps_idx, 'lipschitz_prior_h_kl')
    post_h_mean, post_h_max = _old_schema_values(direct, eps_idx, 'lipschitz_post_h_kl')
    post_e_mean, post_e_max = _old_schema_values(direct, eps_idx, 'lipschitz_post_e_kl')
    dynamics_mean, dynamics_max = _old_schema_values(direct, eps_idx, 'lipschitz_dynamics_l2')

    if np.isnan(prior_mean):
        prior_mean, prior_max = _current_schema_values(
            direct, eps_idx, 'prior_h_kl_mean', per_row_key='prior_h_kl_per_row_all')
    if np.isnan(post_h_mean):
        post_h_mean, post_h_max = _current_schema_values(
            direct, eps_idx, 'post_h_kl_mean', per_row_key='post_h_kl_per_row_all')
    if np.isnan(post_e_mean):
        post_e_mean, post_e_max = _current_schema_values(
            direct, eps_idx, 'post_e_kl_mean', per_row_key='post_e_kl_per_row_all')
    if np.isnan(dynamics_mean):
        dynamics_mean, dynamics_max = _current_schema_values(
            direct, eps_idx, 'dynamics_h_perturbation_l2_mean')

    return {
        'prior_mean': prior_mean,
        'prior_max': prior_max,
        'post_h_mean': post_h_mean,
        'post_h_max': post_h_max,
        'post_e_mean': post_e_mean,
        'post_e_max': post_e_max,
        'dynamics_mean': dynamics_mean,
        'dynamics_max': dynamics_max,
    }


def load_lipschitz_data(exp_dir):
    """Load Lipschitz data from an experiment directory.

    Tries to load from:
    1. smoothness_all_checkpoints.npz (if --all_checkpoints was used)
    2. Individual smoothness_step_*.npz files
    3. smoothness_results.npz (single checkpoint)

    Returns:
        dict with keys: steps, lipschitz_prior_mean, lipschitz_prior_max,
                        lipschitz_post_h_mean, lipschitz_post_e_mean,
                        lipschitz_dynamics_mean, epsilon_fractions
    """
    exp_dir = Path(exp_dir)

    data = {
        'steps': [],
        'lipschitz_prior_mean': [],
        'lipschitz_prior_max': [],
        'lipschitz_post_h_mean': [],
        'lipschitz_post_h_max': [],
        'lipschitz_post_e_mean': [],
        'lipschitz_post_e_max': [],
        'lipschitz_dynamics_mean': [],
        'lipschitz_dynamics_max': [],
        'epsilon_fractions': None,
        'task_name': None,
    }

    # Try loading combined results first
    combined_path = exp_dir / 'smoothness_all_checkpoints.npz'
    if combined_path.exists():
        print(f"  Loading from combined file: {combined_path}")
        npz = np.load(combined_path, allow_pickle=True)

        steps = npz['steps']
        direct_results_list = npz['direct_perturbation_results']
        data['epsilon_fractions'] = npz['epsilon_fractions']
        data['task_name'] = str(npz.get('task_name', ''))

        # Use largest epsilon (last index)
        eps_idx = -1

        for i, step in enumerate(steps):
            direct = direct_results_list[i]
            if direct is None:
                continue

            data['steps'].append(step)

            metrics = extract_metrics_from_direct(direct, eps_idx)
            data['lipschitz_prior_mean'].append(metrics['prior_mean'])
            data['lipschitz_prior_max'].append(metrics['prior_max'])
            data['lipschitz_post_h_mean'].append(metrics['post_h_mean'])
            data['lipschitz_post_h_max'].append(metrics['post_h_max'])
            data['lipschitz_post_e_mean'].append(metrics['post_e_mean'])
            data['lipschitz_post_e_max'].append(metrics['post_e_max'])
            data['lipschitz_dynamics_mean'].append(metrics['dynamics_mean'])
            data['lipschitz_dynamics_max'].append(metrics['dynamics_max'])

        return data

    # Try loading individual step files
    analysis_dir = exp_dir / 'smoothness_analysis_all_ckpts'
    if analysis_dir.exists():
        step_files = sorted(analysis_dir.glob('smoothness_step_*.npz'))
        if step_files:
            print(f"  Loading from {len(step_files)} individual step files")

            for step_file in step_files:
                # Extract step from filename
                match = re.search(r'smoothness_step_(\d+)\.npz', step_file.name)
                if not match:
                    continue
                step = int(match.group(1))

                npz = np.load(step_file, allow_pickle=True)
                direct = npz.get('direct_perturbation_results')

                if direct is None:
                    continue

                if data['epsilon_fractions'] is None:
                    data['epsilon_fractions'] = npz.get('epsilon_fractions')
                    data['task_name'] = str(npz.get('task_name', ''))

                eps_idx = -1

                data['steps'].append(step)

                metrics = extract_metrics_from_direct(direct, eps_idx)
                data['lipschitz_prior_mean'].append(metrics['prior_mean'])
                data['lipschitz_prior_max'].append(metrics['prior_max'])
                data['lipschitz_post_h_mean'].append(metrics['post_h_mean'])
                data['lipschitz_post_h_max'].append(metrics['post_h_max'])
                data['lipschitz_post_e_mean'].append(metrics['post_e_mean'])
                data['lipschitz_post_e_max'].append(metrics['post_e_max'])
                data['lipschitz_dynamics_mean'].append(metrics['dynamics_mean'])
                data['lipschitz_dynamics_max'].append(metrics['dynamics_max'])

            return data

    # Try loading single checkpoint results
    single_path = exp_dir / 'smoothness_results.npz'
    if single_path.exists():
        print(f"  Loading from single checkpoint file: {single_path}")
        npz = np.load(single_path, allow_pickle=True)

        direct = npz.get('direct_perturbation_results')
        step = npz.get('training_step', 0)

        if direct is not None:
            data['epsilon_fractions'] = npz.get('epsilon_fractions')
            data['task_name'] = str(npz.get('task_name', ''))

            eps_idx = -1
            data['steps'].append(step)

            metrics = extract_metrics_from_direct(direct, eps_idx)
            data['lipschitz_prior_mean'].append(metrics['prior_mean'])
            data['lipschitz_prior_max'].append(metrics['prior_max'])
            data['lipschitz_post_h_mean'].append(metrics['post_h_mean'])
            data['lipschitz_post_h_max'].append(metrics['post_h_max'])
            data['lipschitz_post_e_mean'].append(metrics['post_e_mean'])
            data['lipschitz_post_e_max'].append(metrics['post_e_max'])
            data['lipschitz_dynamics_mean'].append(metrics['dynamics_mean'])
            data['lipschitz_dynamics_max'].append(metrics['dynamics_max'])

    return data


def extract_seed_from_folder(folder_name):
    """Extract seed number from folder name."""
    match = re.search(r'seed(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_game_from_folder(folder_name):
    """Extract exact DMC game name from an experiment folder name."""
    parts = folder_name.split('_')
    try:
        dmc_idx = parts.index('dmc')
    except ValueError:
        return None

    game_parts = []
    stop_words = {
        'baseline', 'lipschitz', 'gp', 'seed', 'tr', 'h', 'prior', 'post',
        'rows32', 'sqrt', 'linear', 'exponential', 'cosine', 'step', 'none',
    }
    for part in parts[dmc_idx + 1:]:
        low = part.lower()
        if part.startswith('seed'):
            break
        if low in stop_words:
            break
        if low.startswith(('prior', 'post')):
            break
        if low.startswith('h') and len(low) > 1 and low[1].isdigit():
            break
        game_parts.append(part)

    return '_'.join(game_parts) if game_parts else None


def extract_config_type(folder_name):
    """Determine if folder is baseline or lipschitz config."""
    folder_lower = folder_name.lower()
    if 'baseline' in folder_lower:
        return 'baseline'
    elif 'lipschitz' in folder_lower or 'gp_prior' in folder_lower or 'gp_post' in folder_lower:
        return 'lipschitz'
    return 'unknown'


def plot_comparison(baseline_data, lipschitz_data, output_path, title_suffix=''):
    """Plot comparison of Lipschitz constants between baseline and Lipschitz models.

    Uses color scheme and labels from plot_2_sir.py.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Convert to arrays
    base_steps = np.array(baseline_data['steps'])
    lip_steps = np.array(lipschitz_data['steps'])

    # 1. Prior Lipschitz Mean
    ax = axes[0, 0]
    if baseline_data['lipschitz_prior_mean']:
        plot_curve(ax, base_steps, baseline_data['lipschitz_prior_mean'], None,
                   COLOR_BASELINE, LABEL_BASELINE, show_std=False, linewidth=2.1)
    if lipschitz_data['lipschitz_prior_mean']:
        plot_curve(ax, lip_steps, lipschitz_data['lipschitz_prior_mean'], None,
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D', show_std=False)
    style_axis(ax, 'Prior KL mean')
    ax.legend(loc='best', framealpha=0.9)

    # 2. Posterior (e) Lipschitz Mean
    ax = axes[0, 1]
    if baseline_data['lipschitz_post_e_mean']:
        plot_curve(ax, base_steps, baseline_data['lipschitz_post_e_mean'], None,
                   COLOR_BASELINE, LABEL_BASELINE, show_std=False, linewidth=2.1)
    if lipschitz_data['lipschitz_post_e_mean']:
        plot_curve(ax, lip_steps, lipschitz_data['lipschitz_post_e_mean'], None,
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D', show_std=False)
    style_axis(ax, 'Posterior KL mean')
    ax.legend(loc='best', framealpha=0.9)

    # 3. Dynamics Lipschitz Mean
    ax = axes[1, 0]
    if baseline_data['lipschitz_dynamics_mean']:
        plot_curve(ax, base_steps, baseline_data['lipschitz_dynamics_mean'], None,
                   COLOR_BASELINE, LABEL_BASELINE, show_std=False, linewidth=2.1)
    if lipschitz_data['lipschitz_dynamics_mean']:
        plot_curve(ax, lip_steps, lipschitz_data['lipschitz_dynamics_mean'], None,
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D', show_std=False)
    style_axis(ax, 'Dynamics L2 mean')
    ax.legend(loc='best', framealpha=0.9)

    # 4. Prior Lipschitz Max
    ax = axes[1, 1]
    if baseline_data['lipschitz_prior_max']:
        plot_curve(ax, base_steps, baseline_data['lipschitz_prior_max'], None,
                   COLOR_BASELINE, LABEL_BASELINE, show_std=False, linewidth=2.1)
    if lipschitz_data['lipschitz_prior_max']:
        plot_curve(ax, lip_steps, lipschitz_data['lipschitz_prior_max'], None,
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D', show_std=False)
    style_axis(ax, 'Prior KL max')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight')
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf') if hasattr(output_path, 'with_suffix') else str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to: {output_path}")
    print(f"Saved comparison plot to: {pdf_path}")


def plot_individual_metric(baseline_data, lipschitz_data, metric_key, metric_name,
                           output_path, ylabel, title_suffix=''):
    """Plot a single metric comparison using plot_2_sir.py style."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    base_steps = np.array(baseline_data['steps'])
    lip_steps = np.array(lipschitz_data['steps'])

    if baseline_data[metric_key]:
        plot_curve(ax, base_steps, baseline_data[metric_key], None,
                   COLOR_BASELINE, LABEL_BASELINE, show_std=False, linewidth=2.1)
    if lipschitz_data[metric_key]:
        plot_curve(ax, lip_steps, lipschitz_data[metric_key], None,
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D', show_std=False)

    style_axis(ax, ylabel)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight')
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf') if hasattr(output_path, 'with_suffix') else str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()


def plot_aggregated_comparison(all_baseline_data, all_lipschitz_data, output_dir, game_name):
    """Plot aggregated comparison across multiple seeds.

    Only plots mean ± std (no individual seed curves).
    Uses color scheme and labels from plot_2_sir.py.

    Args:
        all_baseline_data: List of data dicts from baseline experiments
        all_lipschitz_data: List of data dicts from lipschitz experiments
        output_dir: Directory to save plots
        game_name: Game name for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    metrics = METRICS

    for metric_key, short_name, ylabel in metrics:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Use union of all steps
        all_steps = set()
        for data in all_baseline_data + all_lipschitz_data:
            all_steps.update(data['steps'])
        common_steps = sorted(all_steps)

        if common_steps:
            # Plot baseline mean ± std
            common_steps, base_mean, base_std = aggregate_runs(
                all_baseline_data, metric_key, common_steps)
            if base_mean.size:
                plot_curve(ax, common_steps, base_mean, base_std, COLOR_BASELINE,
                           LABEL_BASELINE, linewidth=2.1)

            # Plot lipschitz mean ± std
            common_steps, lip_mean, lip_std = aggregate_runs(
                all_lipschitz_data, metric_key, common_steps)
            if lip_mean.size:
                plot_curve(ax, common_steps, lip_mean, lip_std, COLOR_LIPSCHITZ,
                           LABEL_LIPSCHITZ, linestyle='-', marker='D')

        style_axis(ax, ylabel)
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        output_path = output_dir / f'aggregated_{game_name}_{short_name}.png'
        plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight')
        # Also save as PDF
        pdf_path = output_dir / f'aggregated_{game_name}_{short_name}.pdf'
        plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()

        print(f"Saved aggregated plot: {output_path}")
        print(f"Saved aggregated plot: {pdf_path}")


def find_experiment_folders(base_dir, game_name):
    """Find all experiment folders for a specific game.

    Returns:
        baseline_folders: List of baseline experiment folders
        lipschitz_folders: List of lipschitz experiment folders
    """
    base_dir = Path(base_dir)

    baseline_folders = []
    lipschitz_folders = []

    # Search patterns
    patterns = [
        f'*{game_name}*baseline*',
        f'*{game_name}*lipschitz*',
        f'*{game_name}*gp_prior*',
        f'*{game_name}*gp_post*',
    ]

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # Exact match to avoid mixing variants like acrobot_swingup and
        # acrobot_swingup_sparse.
        if extract_game_from_folder(item.name) != game_name:
            continue

        config_type = extract_config_type(item.name)

        if config_type == 'baseline':
            baseline_folders.append(item)
        elif config_type == 'lipschitz':
            lipschitz_folders.append(item)

    # Sort by seed
    baseline_folders.sort(key=lambda x: extract_seed_from_folder(x.name) or 0)
    lipschitz_folders.sort(key=lambda x: extract_seed_from_folder(x.name) or 0)

    return baseline_folders, lipschitz_folders


def discover_games(base_dir):
    """Return games with both baseline and lipschitz experiment folders."""
    base_dir = Path(base_dir)
    games = {}
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        game = extract_game_from_folder(item.name)
        config = extract_config_type(item.name)
        if not game or config not in {'baseline', 'lipschitz'}:
            continue
        games.setdefault(game, set()).add(config)
    return sorted(game for game, configs in games.items() if {'baseline', 'lipschitz'} <= configs)


def load_game_data(base_dir, game_name):
    baseline_folders, lipschitz_folders = find_experiment_folders(base_dir, game_name)
    all_baseline_data = []
    all_lipschitz_data = []

    for folder in baseline_folders:
        data = load_lipschitz_data(folder)
        if data['steps']:
            data['seed'] = extract_seed_from_folder(folder.name)
            all_baseline_data.append(data)

    for folder in lipschitz_folders:
        data = load_lipschitz_data(folder)
        if data['steps']:
            data['seed'] = extract_seed_from_folder(folder.name)
            all_lipschitz_data.append(data)

    return all_baseline_data, all_lipschitz_data


def compute_game_metric(all_baseline_data, all_lipschitz_data, metric_key):
    all_steps = sorted({
        int(step)
        for data in all_baseline_data + all_lipschitz_data
        for step in data.get('steps', [])
    })
    if not all_steps:
        return None
    steps, base_mean, base_std = aggregate_runs(all_baseline_data, metric_key, all_steps)
    _, lip_mean, lip_std = aggregate_runs(all_lipschitz_data, metric_key, all_steps)
    if not base_mean.size and not lip_mean.size:
        return None
    return {
        'steps': steps,
        'baseline_mean': base_mean,
        'baseline_std': base_std,
        'lipschitz_mean': lip_mean,
        'lipschitz_std': lip_std,
    }


def plot_task_metric(game_name, metric_result, ylabel, output_path, formats=('png', 'pdf'), png_dpi=SAVE_DPI):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_curve(ax, metric_result['steps'], metric_result['baseline_mean'],
               metric_result['baseline_std'], COLOR_BASELINE, LABEL_BASELINE, linewidth=2.1)
    plot_curve(ax, metric_result['steps'], metric_result['lipschitz_mean'],
               metric_result['lipschitz_std'], COLOR_LIPSCHITZ, LABEL_LIPSCHITZ,
               linestyle='-', marker='D')
    style_axis(ax, ylabel)
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    saved = save_figure(fig, output_path, formats, png_dpi)
    plt.close(fig)
    return saved


def plot_all_task_aggregate(metric_by_game, metric_key, short_name, ylabel, output_dir,
                            formats=('png', 'pdf'), png_dpi=SAVE_DPI):
    output_dir = Path(output_dir)
    valid = {
        game: result for game, result in metric_by_game.items()
        if result is not None
        and len(result['steps'])
        and len(result['baseline_mean'])
        and len(result['lipschitz_mean'])
    }
    if not valid:
        return []

    common_steps = sorted({int(step) for result in valid.values() for step in result['steps']})
    baseline_values = []
    lipschitz_values = []
    for result in valid.values():
        steps = np.asarray(result['steps'], dtype=float)
        base = np.asarray(result['baseline_mean'], dtype=float)
        lip = np.asarray(result['lipschitz_mean'], dtype=float)
        baseline_values.append(np.interp(common_steps, steps, base))
        lipschitz_values.append(np.interp(common_steps, steps, lip))

    common_steps = np.asarray(common_steps, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    lipschitz_values = np.asarray(lipschitz_values, dtype=float)
    base_mean = np.nanmean(baseline_values, axis=0)
    base_std = np.nanstd(baseline_values, axis=0)
    lip_mean = np.nanmean(lipschitz_values, axis=0)
    lip_std = np.nanstd(lipschitz_values, axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_curve(ax, common_steps, base_mean, base_std, COLOR_BASELINE, LABEL_BASELINE, linewidth=2.1)
    plot_curve(ax, common_steps, lip_mean, lip_std, COLOR_LIPSCHITZ, LABEL_LIPSCHITZ,
               linestyle='-', marker='D')
    style_axis(ax, ylabel)
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    output_path = output_dir / f'aggregate_all_tasks_{short_name}'
    saved = save_figure(fig, output_path, formats, png_dpi)
    plt.close(fig)

    np.savez(
        output_dir / f'aggregate_all_tasks_{short_name}.npz',
        steps=common_steps,
        baseline_mean=base_mean,
        baseline_std=base_std,
        lipschitz_mean=lip_mean,
        lipschitz_std=lip_std,
        games=np.asarray(sorted(valid)),
        metric_key=metric_key,
        ylabel=ylabel,
    )
    return saved


def plot_metric_grid(metric_by_game, short_name, ylabel, output_dir, ncols=4,
                     formats=('png', 'pdf'), png_dpi=SAVE_DPI):
    games = sorted(game for game, result in metric_by_game.items() if result is not None)
    if not games:
        return []

    # Reserve one empty cell for the legend when the grid shape has spare space.
    total_cells = len(games)
    nrows = math.ceil(total_cells / ncols)
    if total_cells % ncols == 0:
        nrows += 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    handles = labels = None

    for idx, game in enumerate(games):
        ax = axes_flat[idx]
        result = metric_by_game[game]
        plot_curve(ax, result['steps'], result['baseline_mean'], result['baseline_std'],
                   COLOR_BASELINE, LABEL_BASELINE, linewidth=2.1)
        plot_curve(ax, result['steps'], result['lipschitz_mean'], result['lipschitz_std'],
                   COLOR_LIPSCHITZ, LABEL_LIPSCHITZ, linestyle='-', marker='D')
        ax.set_title(format_game_name(game), fontsize=20)
        row = idx // ncols
        col = idx % ncols
        style_axis(
            ax,
            ylabel,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(games):]:
        ax.axis('off')

    if handles:
        if len(games) < len(axes_flat):
            legend_ax = axes_flat[len(games)]
            legend_ax.legend(handles, labels, loc='center', frameon=False, fontsize=12)
        else:
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=12)

    fig.tight_layout()
    output_path = Path(output_dir) / f'grid_all_tasks_{short_name}'
    saved = save_figure(fig, output_path, formats, png_dpi)
    plt.close(fig)
    return saved


def run_all_games(base_dir, output_dir, metrics, formats=('png', 'pdf'), png_dpi=SAVE_DPI, ncols=4):
    output_dir = Path(output_dir)
    individual_dir = output_dir / 'individual'
    aggregate_dir = output_dir / 'aggregate'
    grid_dir = output_dir / 'grid'
    for path in (individual_dir, aggregate_dir, grid_dir):
        path.mkdir(exist_ok=True, parents=True)

    games = discover_games(base_dir)
    print(f'Found {len(games)} games with baseline and GPLD smoothness data')
    metric_lookup = {short_name: (metric_key, short_name, ylabel) for metric_key, short_name, ylabel in METRICS}
    selected_metrics = [metric_lookup[name] for name in metrics]

    all_metric_results = {short_name: {} for _, short_name, _ in selected_metrics}
    manifest_lines = [
        '# Lipschitz comparison plots',
        '',
        f'Base directory: `{base_dir}`',
        f'Games: {len(games)}',
        f'X label: `{X_LABEL}`',
        '',
    ]

    for game in games:
        print(f'Loading {game}')
        all_baseline_data, all_lipschitz_data = load_game_data(base_dir, game)
        manifest_lines.append(
            f'- {game}: baseline seeds={len(all_baseline_data)}, GPLD seeds={len(all_lipschitz_data)}'
        )
        if not all_baseline_data or not all_lipschitz_data:
            continue
        game_dir = individual_dir / game
        game_dir.mkdir(exist_ok=True, parents=True)
        for metric_key, short_name, ylabel in selected_metrics:
            result = compute_game_metric(all_baseline_data, all_lipschitz_data, metric_key)
            all_metric_results[short_name][game] = result
            if result is None:
                continue
            saved = plot_task_metric(
                game, result, ylabel, game_dir / f'{game}_{short_name}', formats, png_dpi)
            print(f'  Saved individual {short_name}: {saved[0]}')

    for metric_key, short_name, ylabel in selected_metrics:
        saved = plot_all_task_aggregate(
            all_metric_results[short_name], metric_key, short_name, ylabel,
            aggregate_dir, formats, png_dpi)
        if saved:
            print(f'Saved aggregate {short_name}: {saved[0]}')
        saved = plot_metric_grid(
            all_metric_results[short_name], short_name, ylabel, grid_dir,
            ncols=ncols, formats=formats, png_dpi=png_dpi)
        if saved:
            print(f'Saved grid {short_name}: {saved[0]}')

    (output_dir / 'manifest.md').write_text('\n'.join(manifest_lines) + '\n')
    (output_dir / 'games_used.txt').write_text('\n'.join(games) + '\n')
    return games


def main():
    global PROPRIO_XMAX
    parser = argparse.ArgumentParser(description='Compare Lipschitz constants between experiments')

    # Mode 1: Direct comparison of two folders
    parser.add_argument('--baseline', '-b', type=str, default=None,
                        help='Path to baseline experiment folder')
    parser.add_argument('--lipschitz', '-l', type=str, default=None,
                        help='Path to Lipschitz experiment folder')

    # Mode 2: Game-based comparison
    parser.add_argument('--game', '-g', type=str, default=None,
                        help='Game name to search for (e.g., walker_walk)')
    parser.add_argument('--base_dir', '-d', type=str, default=None,
                        help='Base directory containing experiment folders')
    parser.add_argument('--all_games', action='store_true',
                        help='Generate per-task, aggregate-over-task, and grid plots for all discovered games')

    # Output options
    parser.add_argument('--output_dir', '-o', type=str, default='./lipschitz_comparison_plots',
                        help='Output directory for plots')
    parser.add_argument('--individual_plots', action='store_true',
                        help='Generate individual metric plots (in addition to combined)')
    parser.add_argument('--metrics', type=str, default='prior_mean,prior_max,post_e_mean,dynamics_mean',
                        help='Comma-separated metric short names to plot')
    parser.add_argument('--formats', type=str, default='png,pdf',
                        help='Comma-separated output formats')
    parser.add_argument('--png_dpi', type=int, default=SAVE_DPI,
                        help='DPI for PNG outputs')
    parser.add_argument('--xmax', type=int, default=PROPRIO_XMAX,
                        help='Maximum x-axis environment step')
    parser.add_argument('--ncols', type=int, default=4,
                        help='Number of grid columns in --all_games mode')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    PROPRIO_XMAX = args.xmax
    formats = parse_formats(args.formats)
    metric_names = parse_formats(args.metrics)
    valid_metric_names = {short_name for _, short_name, _ in METRICS}
    unknown_metrics = sorted(set(metric_names) - valid_metric_names)
    if unknown_metrics:
        print(f"ERROR: Unknown metrics: {unknown_metrics}. Valid metrics: {sorted(valid_metric_names)}")
        return

    if args.all_games and args.base_dir:
        print("="*80)
        print("LIPSCHITZ COMPARISON FOR ALL GAMES")
        print("="*80)
        games = run_all_games(
            args.base_dir,
            output_dir,
            metric_names,
            formats=formats,
            png_dpi=args.png_dpi,
            ncols=args.ncols,
        )
        print(f"\nGenerated all-game plots for {len(games)} games")
        print(f"All plots saved to: {output_dir}")
    elif args.baseline and args.lipschitz:
        # Mode 1: Direct comparison
        print("="*80)
        print("LIPSCHITZ CONSTANT COMPARISON")
        print("="*80)

        print(f"\nBaseline: {args.baseline}")
        baseline_data = load_lipschitz_data(args.baseline)
        print(f"  Loaded {len(baseline_data['steps'])} checkpoints")

        print(f"\nLipschitz: {args.lipschitz}")
        lipschitz_data = load_lipschitz_data(args.lipschitz)
        print(f"  Loaded {len(lipschitz_data['steps'])} checkpoints")

        if not baseline_data['steps'] or not lipschitz_data['steps']:
            print("\nERROR: No Lipschitz data found. Run analyze_smoothness.py first with --all_checkpoints")
            return

        # Generate comparison plot
        task_name = baseline_data['task_name'] or lipschitz_data['task_name'] or 'unknown'
        output_path = output_dir / f'lipschitz_comparison_{task_name}.png'
        plot_comparison(baseline_data, lipschitz_data, output_path, f' - {task_name}')

        # Individual metric plots
        if args.individual_plots:
            metrics = [
                ('lipschitz_prior_mean', 'prior_mean', 'Mean Lipschitz Constant (Prior)'),
                ('lipschitz_prior_max', 'prior_max', 'Max Lipschitz Constant (Prior)'),
                ('lipschitz_post_e_mean', 'post_e_mean', 'Mean Lipschitz Constant (Post e)'),
                ('lipschitz_dynamics_mean', 'dynamics_mean', 'Mean Lipschitz Constant (Dynamics)'),
            ]

            for metric_key, short_name, ylabel in metrics:
                output_path = output_dir / f'lipschitz_{short_name}_{task_name}.png'
                plot_individual_metric(baseline_data, lipschitz_data, metric_key,
                                      short_name, output_path, ylabel, f' - {task_name}')

        print(f"\nPlots saved to: {output_dir}")

    elif args.game and args.base_dir:
        # Mode 2: Game-based comparison
        print("="*80)
        print(f"LIPSCHITZ COMPARISON FOR GAME: {args.game}")
        print("="*80)

        baseline_folders, lipschitz_folders = find_experiment_folders(args.base_dir, args.game)

        print(f"\nFound {len(baseline_folders)} baseline experiments:")
        for f in baseline_folders:
            seed = extract_seed_from_folder(f.name)
            print(f"  Seed {seed}: {f.name}")

        print(f"\nFound {len(lipschitz_folders)} Lipschitz experiments:")
        for f in lipschitz_folders:
            seed = extract_seed_from_folder(f.name)
            print(f"  Seed {seed}: {f.name}")

        if not baseline_folders and not lipschitz_folders:
            print("\nNo experiment folders found!")
            return

        # Load all data
        all_baseline_data = []
        all_lipschitz_data = []

        for folder in baseline_folders:
            print(f"\nLoading baseline: {folder.name}")
            data = load_lipschitz_data(folder)
            if data['steps']:
                data['seed'] = extract_seed_from_folder(folder.name)
                all_baseline_data.append(data)

        for folder in lipschitz_folders:
            print(f"\nLoading lipschitz: {folder.name}")
            data = load_lipschitz_data(folder)
            if data['steps']:
                data['seed'] = extract_seed_from_folder(folder.name)
                all_lipschitz_data.append(data)

        # Create per-seed comparison plots
        seed_output_dir = output_dir / f'{args.game}_seed_comparisons'
        seed_output_dir.mkdir(exist_ok=True, parents=True)

        # Match by seed
        baseline_by_seed = {d.get('seed'): d for d in all_baseline_data}
        lipschitz_by_seed = {d.get('seed'): d for d in all_lipschitz_data}

        all_seeds = set(baseline_by_seed.keys()) | set(lipschitz_by_seed.keys())

        for seed in sorted(all_seeds):
            if seed in baseline_by_seed and seed in lipschitz_by_seed:
                output_path = seed_output_dir / f'comparison_seed{seed}.png'
                plot_comparison(baseline_by_seed[seed], lipschitz_by_seed[seed],
                              output_path, f' - {args.game} Seed {seed}')

        # Create aggregated plot
        if all_baseline_data or all_lipschitz_data:
            plot_aggregated_comparison(all_baseline_data, all_lipschitz_data,
                                      output_dir, args.game)

        print(f"\nAll plots saved to: {output_dir}")

    else:
        print("ERROR: Must specify either (--baseline and --lipschitz) or (--game and --base_dir)")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
