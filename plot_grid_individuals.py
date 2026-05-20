#!/usr/bin/env python3
"""
plot_neurips_grid_individuals.py

Generate a NeurIPS-style baseline-vs-GPLD grid and matching individual per-game
plots from processed aggregate NPZ files.

Defaults match the proprio final setting:
- Baseline: {prefix}_prior0.0_post0.0_rows32_none_h25_size12m_tr512.npz
- GPLD:     {prefix}_prior0.0_post0.5_rows32_sqrt_time_gp_start_0_h25_size12m_tr512.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plotting_neurips import BASELINE_COLOR, DEFAULT_SPEC, apply_neurips_style


apply_neurips_style(DEFAULT_SPEC)
AXIS_FONTSIZE = 14
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 20
GP_COLOR = "#1f77b4"
GP_MARKER = "D"


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0 or data.size == 0:
        return data.copy()
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def load_stat_from_npz(npz_path: Path, stat: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    if stat == "mean":
        arr = data["mean_scores"]
        std_arr = data["std_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = std_arr[:, 1]
    elif stat == "median":
        arr = data["median_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = np.zeros_like(scores)
    elif stat == "iqm":
        arr = data["iqm_scores"]
        std_arr = data["iqm_std_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = std_arr[:, 1]
    else:
        raise ValueError(stat)

    meta = {
        "prior_lam": float(data["prior_lam"]) if "prior_lam" in data else 0.0,
        "post_lam": float(data["post_lam"]) if "post_lam" in data else 0.0,
        "decay_type": str(data["decay_type"]) if "decay_type" in data else "none",
        "h": int(data["h"]) if "h" in data else 25,
        "size": str(data["size"]) if "size" in data else "12m",
        "tr": int(data["tr"]) if "tr" in data else 512,
        "gp_rows": int(data["gp_rows"]) if "gp_rows" in data else None,
    }
    return steps, scores, std, meta


def _load_seed_curve(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if "scores" in data:
        arr = data["scores"].astype(np.float64)
        return arr[:, 0], arr[:, 1]
    if "step" in data and "score" in data:
        return data["step"].astype(np.float64), data["score"].astype(np.float64)
    raise ValueError(f"No seed curve arrays found in {npz_path}")


def _aggregate_path_to_seed_glob(npz_path: Path, stat: str) -> str:
    stem = npz_path.stem
    prefix = f"{stat.upper()}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.startswith("SEEDS") and "_" in stem:
        stem = stem.split("_", 1)[1]
    return f"{stem}_seed*.npz"


def median_ci_radius_from_seed_files(npz_path: Path, stat: str, target_steps: np.ndarray) -> np.ndarray:
    seed_paths = sorted(npz_path.parent.glob(_aggregate_path_to_seed_glob(npz_path, stat)))
    if len(seed_paths) < 2 or target_steps.size == 0:
        return np.zeros_like(target_steps, dtype=np.float64)

    seed_values: List[np.ndarray] = []
    for seed_path in seed_paths:
        try:
            steps, scores = _load_seed_curve(seed_path)
        except ValueError:
            continue
        if steps.size == 0 or scores.size == 0:
            continue
        seed_values.append(np.interp(target_steps, steps, scores))

    if len(seed_values) < 2:
        return np.zeros_like(target_steps, dtype=np.float64)

    values = np.stack(seed_values, axis=0)
    rng = np.random.default_rng(0)
    sample_idx = rng.integers(0, values.shape[0], size=(2000, values.shape[0]))
    boot_medians = np.median(values[sample_idx], axis=1)
    lower = np.percentile(boot_medians, 2.5, axis=0)
    upper = np.percentile(boot_medians, 97.5, axis=0)
    median = np.median(values, axis=0)
    return np.maximum(median - lower, upper - median)


def step_formatter(x: float, _: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def clip_curve(
    steps: np.ndarray,
    scores: np.ndarray,
    std: np.ndarray,
    xmax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xmax is None:
        return steps, scores, std
    mask = steps <= float(xmax)
    return steps[mask], scores[mask], std[mask]


def format_template(template: str, stat: str) -> str:
    prefix = stat.upper()
    return template.format(prefix=prefix, stat=stat, stat_upper=prefix)


def game_title(game: str) -> str:
    return game.replace("dmc_", "").replace("_", " ").title()


def xmax_tag(xmax: Optional[float]) -> str:
    if xmax is None:
        return "none"
    value = float(xmax)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, formats: List[str], png_dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=png_dpi if fmt.lower() == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")


def draw_pair(
    ax: plt.Axes,
    item: Dict[str, Any],
    show_std: bool,
    legend: bool,
    title: bool,
    xmax: Optional[float],
) -> Tuple[Any, Any]:
    line1, = ax.plot(
        item["b_steps"],
        item["b_scores"],
        color=BASELINE_COLOR,
        linestyle="-",
        linewidth=2.1,
        label="Baseline",
        zorder=10,
    )
    if show_std and np.any(item["b_std"] > 0):
        ax.fill_between(
            item["b_steps"],
            item["b_scores"] - item["b_std"],
            item["b_scores"] + item["b_std"],
            color=BASELINE_COLOR,
            alpha=0.12,
        )

    line2, = ax.plot(
        item["g_steps"],
        item["g_scores"],
        color=GP_COLOR,
        linestyle="-",
        linewidth=2.0,
        marker=GP_MARKER,
        markevery=max(1, len(item["g_steps"]) // 10),
        label="GPLD",
        zorder=11,
    )
    if show_std and np.any(item["g_std"] > 0):
        ax.fill_between(
            item["g_steps"],
            item["g_scores"] - item["g_std"],
            item["g_scores"] + item["g_std"],
            color=GP_COLOR,
            alpha=0.12,
        )

    ax.set_title(game_title(str(item["game"])) if title else "", fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(step_formatter))
    if xmax is not None:
        ax.set_xlim(left=0.0, right=float(xmax))
    if legend:
        ax.legend(loc="lower right", framealpha=0.92, fontsize=12)
    return line1, line2


def collect_plot_data(
    processed_root: Path,
    baseline_root: Path,
    gp_root: Path,
    stat: str,
    ema: float,
    xmax: Optional[float],
    baseline_template: str,
    gp_template: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    games = sorted(
        {
            d.name
            for root in (baseline_root, gp_root)
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("dmc_") and not d.name.startswith(".")
        }
    )
    if not games:
        raise SystemExit(f"No dmc_* game folders found under: {baseline_root} or {gp_root}")

    base_name = format_template(baseline_template, stat)
    gp_name = format_template(gp_template, stat)
    out: List[Dict[str, Any]] = []
    missing: List[str] = []

    def smooth(y: np.ndarray) -> np.ndarray:
        return exponential_moving_average(y, ema) if ema < 1.0 else y

    for game in games:
        base_path = baseline_root / game / base_name
        gp_path = gp_root / game / gp_name
        if not base_path.is_file() or not gp_path.is_file():
            missing.append(game)
            print(f"[skip] {game}: missing {base_name} or {gp_name}")
            continue

        b_steps, b_scores, b_std, _ = load_stat_from_npz(base_path, stat)
        g_steps, g_scores, g_std, _ = load_stat_from_npz(gp_path, stat)
        if stat == "median":
            b_std = median_ci_radius_from_seed_files(base_path, stat, b_steps)
            g_std = median_ci_radius_from_seed_files(gp_path, stat, g_steps)
        b_steps, b_scores, b_std = clip_curve(b_steps, b_scores, b_std, xmax)
        g_steps, g_scores, g_std = clip_curve(g_steps, g_scores, g_std, xmax)

        if b_steps.size == 0 or g_steps.size == 0:
            missing.append(game)
            print(f"[skip] {game}: no points remain after applying xmax={xmax}")
            continue

        out.append(
            {
                "game": game,
                "b_steps": b_steps,
                "b_scores": smooth(b_scores),
                "b_std": smooth(b_std) if ema < 1.0 else b_std,
                "g_steps": g_steps,
                "g_scores": smooth(g_scores),
                "g_std": smooth(g_std) if ema < 1.0 else g_std,
            }
        )
    return out, missing


def save_individual_plots(
    plot_data: List[Dict[str, Any]],
    out_dir: Path,
    stat: str,
    ema: float,
    xmax: Optional[float],
    formats: List[str],
    png_dpi: int,
) -> None:
    show_std = True
    xmax_label = xmax_tag(xmax)
    for item in plot_data:
        fig, ax = plt.subplots(figsize=DEFAULT_SPEC.figsize)
        draw_pair(ax, item, show_std=show_std, legend=True, title=False, xmax=xmax)
        ax.set_xlabel("Environment steps", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("Episodic return", fontsize=AXIS_FONTSIZE)
        plt.tight_layout()
        stem = f"{item['game']}_{stat}_ema{ema:g}_xmax{xmax_label}"
        save_fig(fig, out_dir, stem, formats, png_dpi)
        plt.close(fig)


def save_grid(
    plot_data: List[Dict[str, Any]],
    out_dir: Path,
    stat: str,
    ema: float,
    xmax: Optional[float],
    ncols: int,
    formats: List[str],
    png_dpi: int,
) -> None:
    if not plot_data:
        raise SystemExit("No games available for grid plot.")

    show_std = True
    n_games = len(plot_data)
    nrows = math.ceil(n_games / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = ["Baseline", "GPLD"]
    for ax, item in zip(axes_flat, plot_data):
        handles = draw_pair(ax, item, show_std=show_std, legend=False, title=True, xmax=xmax)
        if legend_handles is None:
            legend_handles = list(handles)

    for idx, ax in enumerate(axes_flat[:n_games]):
        row = idx // ncols
        col = idx % ncols
        ax.set_ylabel("Episodic return" if col == 0 else "", fontsize=AXIS_FONTSIZE)
        ax.set_xlabel("Environment steps" if row == nrows - 1 else "", fontsize=AXIS_FONTSIZE)

    empty_axes = list(axes_flat[n_games:])
    if empty_axes:
        legend_ax = empty_axes[0]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False, fontsize=LEGEND_FONTSIZE)
        for ax in empty_axes[1:]:
            ax.axis("off")
        tight_rect = [0, 0, 1, 1]
    else:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=LEGEND_FONTSIZE,
        )
        tight_rect = [0, 0.06, 1, 1]

    fig.tight_layout(rect=tight_rect)
    save_fig(fig, out_dir, f"grid_all_games_{stat}_ema{ema:g}_xmax{xmax_tag(xmax)}", formats, png_dpi)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate baseline-vs-GPLD grid plus individual per-game plots with CLI xmax."
    )
    parser.add_argument(
        "--processed-root",
        "--processed_root",
        dest="processed_root",
        type=str,
        default=str(root / "processed_scores"),
        help="Root folder containing per-game processed score files.",
    )
    parser.add_argument("--output", "-o", type=str, default=str(root / "plots_neurips_grid_individuals"))
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=None,
        help="Optional root for baseline per-game folders. Defaults to --processed-root.",
    )
    parser.add_argument(
        "--gp-root",
        type=str,
        default=None,
        help="Optional root for GPLD per-game folders. Defaults to --processed-root.",
    )
    parser.add_argument("--stat", choices=["mean", "median", "iqm"], default="mean")
    parser.add_argument("--ema", type=float, default=0.1)
    parser.add_argument("--xmax", type=float, default=500_032.0)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--png-dpi", type=int, default=150)
    parser.add_argument("--individual-only", action="store_true", help="Only generate individual per-game plots.")
    parser.add_argument("--grid-only", action="store_true", help="Only generate the grid plot.")
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="{prefix}_prior0.0_post0.0_rows32_none_h25_size12m_tr512.npz",
        help="Per-game baseline filename template. Supports {prefix}, {stat}, {stat_upper}.",
    )
    parser.add_argument(
        "--gp-template",
        type=str,
        default="{prefix}_prior0.0_post0.5_rows32_sqrt_time_gp_start_0_h25_size12m_tr512.npz",
        help="Per-game GPLD filename template. Supports {prefix}, {stat}, {stat_upper}.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser()
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")
    baseline_root = Path(args.baseline_root).expanduser() if args.baseline_root else processed_root
    gp_root = Path(args.gp_root).expanduser() if args.gp_root else processed_root
    if not baseline_root.is_dir():
        raise SystemExit(f"Baseline root not found: {baseline_root}")
    if not gp_root.is_dir():
        raise SystemExit(f"GPLD root not found: {gp_root}")
    if int(args.ncols) <= 0:
        raise SystemExit("--ncols must be positive.")
    if bool(args.individual_only) and bool(args.grid_only):
        raise SystemExit("--individual-only and --grid-only are mutually exclusive.")

    formats = [str(fmt).strip() for fmt in args.formats if str(fmt).strip()]
    if not formats:
        raise SystemExit("No output formats specified.")

    output = Path(args.output).expanduser()
    grid_dir = output / "grid"
    individual_dir = output / "individual"
    plot_data, missing = collect_plot_data(
        processed_root=processed_root,
        baseline_root=baseline_root,
        gp_root=gp_root,
        stat=str(args.stat),
        ema=float(args.ema),
        xmax=float(args.xmax) if args.xmax is not None else None,
        baseline_template=str(args.baseline_template),
        gp_template=str(args.gp_template),
    )
    if not plot_data:
        raise SystemExit("No games available to plot. Check processed root and filename templates.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "games_used.txt").write_text("\n".join(str(item["game"]) for item in plot_data) + "\n", encoding="utf-8")
    if missing:
        (output / "games_missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")

    if not bool(args.grid_only):
        save_individual_plots(
            plot_data=plot_data,
            out_dir=individual_dir,
            stat=str(args.stat),
            ema=float(args.ema),
            xmax=float(args.xmax) if args.xmax is not None else None,
            formats=formats,
            png_dpi=int(args.png_dpi),
        )
    if not bool(args.individual_only):
        save_grid(
            plot_data=plot_data,
            out_dir=grid_dir,
            stat=str(args.stat),
            ema=float(args.ema),
            xmax=float(args.xmax) if args.xmax is not None else None,
            ncols=int(args.ncols),
            formats=formats,
            png_dpi=int(args.png_dpi),
        )


if __name__ == "__main__":
    main()
