"""Shared chart styling.

One palette, one set of rcParams, so every figure in the project reads as part
of the same report. Colours are assigned in a fixed order and never cycled, and
no chart in this repo uses two y-axes.

The palette is a validated categorical set. Slots 1 to 3 are safe for scatter
plots where every pair of colours can appear side by side; past three series,
charts here fall back to facets or direct labels rather than adding hues.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# categorical slots, fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
     "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

SERIES = {
    "base only": C[3],
    "plain GP": C[1],
    "base+residual": C[0],
    "transfer": C[0],
    "qNEHVI": C[0],
    "random": MUTED,
    "greedy potency": C[1],
    "qNEHVI no transfer": C[2],
}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,     # grid behind the marks, never through them
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def label_last(ax, x, y, text, color, dx=0.01, fontsize=9):
    """Direct label at the end of a line, so identity is never colour alone."""
    ax.annotate(text, xy=(x[-1], y[-1]),
                xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=fontsize, color=color)


def label_series_ends(ax, items, x_pad=6, min_gap_frac=0.06, fontsize=9):
    """Direct-label several lines at their right-hand end without collisions.

    `items` is a list of (x_last, y_last, text, colour). Labels that would sit
    on top of each other get pushed apart vertically, so two series ending at
    the same value stay readable.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])
    placed: list[float] = []
    for x, y, text, colour in items:
        target = y if not placed else max(y, placed[-1] + gap)
        placed.append(target)
        ax.annotate(text, xy=(x, target), xytext=(x_pad, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=fontsize, color=colour, annotation_clip=False)
    ax.set_ylim(lo, max(hi, placed[-1] + gap * 0.6))


def parity_plot(ax, y_true, y_pred, color=C[0], label=None, lims=None):
    ax.scatter(y_true, y_pred, s=14, alpha=0.5, color=color,
               edgecolors=SURFACE, linewidths=0.5, label=label)
    lo = min(np.min(y_true), np.min(y_pred)) if lims is None else lims[0]
    hi = max(np.max(y_true), np.max(y_pred)) if lims is None else lims[1]
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    return ax


def save(fig, name, outdir="../outputs/figures"):
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{outdir}/{name}.png")
    print(f"  saved {outdir}/{name}.png")
