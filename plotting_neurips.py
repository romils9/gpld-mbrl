"""
Shared matplotlib styling and helpers for NeurIPS-style figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class NeuripsFigureSpec:
    width_in: float = 3.25
    height_in: float = 2.4
    font_size: int = 9
    axes_label_size: int = 9
    axes_title_size: int = 9
    tick_label_size: int = 8
    legend_font_size: int = 8
    line_width: float = 1.6
    marker_size: float = 5.0

    @property
    def figsize(self) -> Tuple[float, float]:
        return (self.width_in, self.height_in)


DEFAULT_SPEC = NeuripsFigureSpec()


def apply_neurips_style(spec: NeuripsFigureSpec = DEFAULT_SPEC) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": spec.font_size,
            "axes.titlesize": spec.axes_title_size,
            "axes.labelsize": spec.axes_label_size,
            "xtick.labelsize": spec.tick_label_size,
            "ytick.labelsize": spec.tick_label_size,
            "legend.fontsize": spec.legend_font_size,
            "figure.figsize": spec.figsize,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": spec.line_width,
            "lines.markersize": spec.marker_size,
            "savefig.bbox": "tight",
        }
    )


BASELINE_COLOR = "#000000"
DEFAULT_MARKERS: Tuple[str, ...] = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h")


def marker_cycle(n: int, markers: Sequence[str] = DEFAULT_MARKERS) -> List[str]:
    if n <= 0:
        return []
    return [markers[i % len(markers)] for i in range(n)]
