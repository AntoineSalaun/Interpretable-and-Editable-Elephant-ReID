"""Consistent printed typography for figures included at different LaTeX widths."""
from pathlib import Path

import matplotlib.pyplot as plt

from figures.style import set_style

# Slightly smaller labels than the 22/20 pt dataset histograms; ticks unchanged.
X_LABEL_SIZE = 20
Y_LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 15


def set_journal_style():
    set_style(use_tex=False)
    plt.rcParams["mathtext.fontset"] = "dejavuserif"


def journal_figsize(linewidth_fraction, height=5.6):
    # Reference: a 6-inch histogram included at 0.49\linewidth.
    return (6 * linewidth_fraction / 0.49, height)


def format_journal_axes(ax, *, grid=True):
    ax.xaxis.label.set_size(X_LABEL_SIZE)
    ax.yaxis.label.set_size(Y_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=6)
    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 9
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7, alpha=1.0)


def save_journal(fig, path):
    """Keep the full PDF canvas so LaTeX preserves the intended font scale."""
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(path, facecolor="white")
        fig.savefig(path.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved → {path}")
