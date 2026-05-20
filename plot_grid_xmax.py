#!/usr/bin/env python3
"""
plot_neurips_grid_xmax.py

Generate a NeurIPS-style grid of per-game learning curves comparing:
- Baseline: prior=0.0, post=0.0, rows32, decay=none
- GPLD:     prior=0.0, post=0.5, rows32, decay=sqrt_time, gp_start=0

This is a focused variant of plot_neurips_grid.py with:
- --xmax exposed as a CLI argument
- --stat support for mean/median/IQM aggregate NPZ files
- grid-label and legend behavior aligned with plot_summary.md
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
TITLE_FONTSIZE = 20
AXIS_FONTSIZE = 14
LEGEND_FONTSIZE = 20
MAIN_VARIANT_COLOR = "#1f77b4"
MAIN_VARIANT_MARKER = "D"


def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if alpha >= 1.0 or data.size == 0:
        return data.copy()
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _load_stat_from_npz(npz_path: Path, stat_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    if stat_type == "mean":
        arr = data["mean_scores"]
        std_arr = data["std_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = std_arr[:, 1]
    elif stat_type == "median":
        arr = data["median_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = np.zeros_like(scores)
    elif stat_type == "iqm":
        arr = data["iqm_scores"]
        std_arr = data["iqm_std_scores"]
        steps = arr[:, 0]
        scores = arr[:, 1]
        std = std_arr[:, 1]
    else:
        raise ValueError(stat_type)

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


def _format_steps_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"
        )
    )


def _clip_curve(
    steps: np.ndarray,
    scores: np.ndarray,
    std: np.ndarray,
    xmax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xmax is None:
        return steps, scores, std
    mask = steps <= float(xmax)
    return steps[mask], scores[mask], std[mask]


def _filename_from_template(template: str, prefix: str, stat_type: str) -> str:
    return template.format(prefix=prefix, stat=stat_type, stat_upper=prefix)


def _game_title(game: str) -> str:
    return game.replace("dmc_", "").replace("_", " ").title()


def _xmax_tag(xmax: Optional[float]) -> str:
    if xmax is None:
        return "none"
    value = float(xmax)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def plot_final_grid(
    processed_root: Path,
    output_dir: Path,
    stat_type: str,
    ema_alpha: float,
    xmax: Optional[float],
    png_dpi: int,
    ncols: int,
    save_formats: List[str],
    baseline_template: str,
    variant_template: str,
) -> None:
    games = sorted(
        d.name
        for d in processed_root.iterdir()
        if d.is_dir() and d.name.startswith("dmc_") and not d.name.startswith(".")
    )
    if not games:
        raise SystemExit(f"No dmc_* game folders found under: {processed_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = stat_type.upper()
    base_name = _filename_from_template(baseline_template, prefix, stat_type)
    var_name = _filename_from_template(variant_template, prefix, stat_type)
    show_std = stat_type != "median"

    def smooth(y: np.ndarray) -> np.ndarray:
        return exponential_moving_average(y, ema_alpha) if ema_alpha < 1.0 else y

    grid_plot_data: List[Dict[str, Any]] = []
    missing: List[str] = []

    for game in games:
        game_dir = processed_root / game
        base_path = game_dir / base_name
        var_path = game_dir / var_name

        if not base_path.is_file() or not var_path.is_file():
            missing.append(game)
            print(f"[skip] {game}: missing {base_name} or {var_name}")
            continue

        b_steps, b_scores, b_std, _ = _load_stat_from_npz(base_path, stat_type)
        v_steps, v_scores, v_std, _ = _load_stat_from_npz(var_path, stat_type)
        b_steps, b_scores, b_std = _clip_curve(b_steps, b_scores, b_std, xmax)
        v_steps, v_scores, v_std = _clip_curve(v_steps, v_scores, v_std, xmax)

        if b_steps.size == 0 or v_steps.size == 0:
            missing.append(game)
            print(f"[skip] {game}: no points remain after applying xmax={xmax}")
            continue

        grid_plot_data.append(
            {
                "game": game,
                "b_steps": b_steps,
                "b_scores": smooth(b_scores),
                "b_std": smooth(b_std) if ema_alpha < 1.0 else b_std,
                "v_steps": v_steps,
                "v_scores": smooth(v_scores),
                "v_std": smooth(v_std) if ema_alpha < 1.0 else v_std,
            }
        )

    if not grid_plot_data:
        raise SystemExit("No games available for grid plot. Check processed_root, templates, and xmax.")

    games_used = [item["game"] for item in grid_plot_data]
    (output_dir / "games_used.txt").write_text("\n".join(games_used) + "\n", encoding="utf-8")
    if missing:
        (output_dir / "games_missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")

    n_games = len(grid_plot_data)
    nrows = math.ceil(n_games / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = ["Baseline", "GPLD"]

    for ax, item in zip(axes_flat, grid_plot_data):
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
            item["v_steps"],
            item["v_scores"],
            color=MAIN_VARIANT_COLOR,
            linestyle="-",
            linewidth=2.0,
            marker=MAIN_VARIANT_MARKER,
            markevery=max(1, len(item["v_steps"]) // 10),
            label="GPLD",
            zorder=11,
        )
        if show_std and np.any(item["v_std"] > 0):
            ax.fill_between(
                item["v_steps"],
                item["v_scores"] - item["v_std"],
                item["v_scores"] + item["v_std"],
                color=MAIN_VARIANT_COLOR,
                alpha=0.12,
            )

        ax.set_title(_game_title(item["game"]), fontsize=TITLE_FONTSIZE)
        ax.grid(True, alpha=0.3)
        _format_steps_axis(ax)
        if xmax is not None:
            ax.set_xlim(left=0.0, right=float(xmax))
        if legend_handles is None:
            legend_handles = [line1, line2]

    for idx, ax in enumerate(axes_flat[:n_games]):
        row = idx // ncols
        col = idx % ncols
        ax.set_ylabel("Episodic return" if col == 0 else "", fontsize=AXIS_FONTSIZE)
        ax.set_xlabel("Environment steps" if row == nrows - 1 else "", fontsize=AXIS_FONTSIZE)

    empty_axes = list(axes_flat[n_games:])
    legend_in_empty_slot = bool(empty_axes) and (n_games // ncols == nrows - 1)
    if legend_in_empty_slot:
        legend_ax = empty_axes[0]
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )
        for ax in empty_axes[1:]:
            ax.axis("off")
        tight_rect = [0, 0, 1, 1]
    else:
        for ax in empty_axes:
            ax.axis("off")
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

    stem = f"grid_all_games_{stat_type}_ema{ema_alpha:g}_xmax{_xmax_tag(xmax)}"
    for fmt in save_formats:
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=png_dpi if fmt.lower() == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate a NeurIPS-style baseline-vs-GPLD grid with CLI xmax.")
    parser.add_argument(
        "--processed-root",
        "--processed_root",
        dest="processed_root",
        type=str,
        default=str(root / "processed_scores"),
        help="Root folder containing per-game processed score files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(root / "plots_neurips_grid_xmax"),
        help="Directory to save grid figures.",
    )
    parser.add_argument("--stat", type=str, default="mean", choices=["mean", "median", "iqm"])
    parser.add_argument("--ema", type=float, default=0.1, help="EMA smoothing alpha (1.0 disables smoothing).")
    parser.add_argument(
        "--xmax",
        type=float,
        default=500_032.0,
        help="Right x-axis limit in environment steps. Use 500032 for proprio, 1000032 for pixel.",
    )
    parser.add_argument("--png-dpi", type=int, default=150, help="PNG dpi when exporting PNGs.")
    parser.add_argument("--ncols", type=int, default=4, help="Number of columns in the subplot grid.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure formats to save.")
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="{prefix}_prior0.0_post0.0_rows32_none_h25_size12m_tr512.npz",
        help="Per-game baseline filename template. Supports {prefix}, {stat}, {stat_upper}.",
    )
    parser.add_argument(
        "--variant-template",
        type=str,
        default="{prefix}_prior0.0_post0.5_rows32_sqrt_time_gp_start_0_h25_size12m_tr512.npz",
        help="Per-game GPLD filename template. Supports {prefix}, {stat}, {stat_upper}.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser()
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")

    if int(args.ncols) <= 0:
        raise SystemExit("--ncols must be positive.")

    save_formats = [str(fmt).strip() for fmt in args.formats if str(fmt).strip()]
    if not save_formats:
        raise SystemExit("No output formats specified.")

    plot_final_grid(
        processed_root=processed_root,
        output_dir=Path(args.output).expanduser(),
        stat_type=str(args.stat).lower(),
        ema_alpha=float(args.ema),
        xmax=float(args.xmax) if args.xmax is not None else None,
        png_dpi=int(args.png_dpi),
        ncols=int(args.ncols),
        save_formats=save_formats,
        baseline_template=str(args.baseline_template),
        variant_template=str(args.variant_template),
    )


if __name__ == "__main__":
    main()
