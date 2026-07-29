"""Shared visual theme for the workshop notebooks.

One import gives both tutorials the same look: an Okabe-Ito colorblind-safe
palette, consistent marker conventions, and small helpers. Conventions used
in EVERY figure of both tutorials:

    truth            filled ▼ in vermilion  (C["truth"])
    prediction/slot  open  ○ in blue        (C["pred"])
    ordered model    open  ○ in grey        (C["ordered"])
    inactive slot    small ○ in light grey  (C["inactive"])
    unmatched/alert  ◆ in reddish purple    (C["alert"])
    rescued/new      + in bluish green      (C["rescue"])
    uncertainty      band/errorbar in sky   (C["band"])
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito palette (colorblind safe)
C = {
    "truth": "#D55E00",     # vermilion
    "pred": "#0072B2",      # blue
    "ordered": "#7F7F7F",   # grey
    "inactive": "#C7C7C7",  # light grey
    "alert": "#CC79A7",     # reddish purple
    "rescue": "#009E73",    # bluish green
    "band": "#56B4E9",      # sky blue
    "gold": "#E69F00",      # orange (secondary series)
    "ink": "#333333",
}

SERIES = [C["pred"], C["truth"], C["rescue"], C["gold"], C["alert"],
          "#56B4E9", "#F0E442", "#000000", "#999999", "#8C613C"]


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "font.size": 10,
    })


def mark_truth(ax, x, y, label=None, ms=9):
    ax.plot(x, y, "v", color=C["truth"], ms=ms, mec="white", mew=0.5,
            label=label, zorder=5)


def mark_pred(ax, x, y, label=None, ms=9, color=None, alpha=1.0):
    ax.plot(x, y, "o", mfc="none", color=color or C["pred"], ms=ms, mew=1.7,
            label=label, alpha=alpha, zorder=6)


def title2(ax, conclusion, detail=None):
    """Conclusion-first title with an optional smaller detail line."""
    # pad lifts the title clear of the detail line drawn at axes-top
    ax.set_title(conclusion, loc="left", pad=16 if detail else 6)
    if detail:
        ax.text(0.0, 1.015, detail, transform=ax.transAxes, fontsize=8,
                color="#666666", va="bottom")


def mhz(x):
    """Hz -> mHz (for display; everything internal stays in Hz)."""
    return 1e3 * x


def fmt_mhz(x, nd=2):
    return f"{1e3 * x:.{nd}f} mHz"


use_style()
