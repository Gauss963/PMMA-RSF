from __future__ import annotations

from typing import Any


INK = "#202124"
MUTED = "#5F6368"
GRID = "#D7DADD"
NAVY = "#24557A"
TEAL = "#00838F"
ORANGE = "#D55E00"
GOLD = "#B8860B"
RED = "#A33A2B"
BLUE = "#4477AA"
CYAN = "#66CCEE"
GREEN = "#228833"
YELLOW = "#CCBB44"
MAGENTA = "#AA3377"
GREY = "#777777"
JOURNAL_COLORS = (BLUE, ORANGE, GREEN, MAGENTA, CYAN, GOLD)


def configure_journal_style() -> None:
    """Apply the compact two-column style used by PMMA analysis figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "axes.titleweight": "normal",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: Any, *, grid: bool = True, twin: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    if not twin:
        axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(True, which="major", axis="both")
    axis.set_axisbelow(True)


def panel_label(axis: Any, label: str) -> None:
    axis.text(
        0.0,
        1.02,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontweight="bold",
        fontsize=9.0,
    )
