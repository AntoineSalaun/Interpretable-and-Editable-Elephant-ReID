from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import GROUP_NAMES, LEVELS, METRICS_DIR, SCRIPT_DIR, ensure_dirs

sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
from figures.style import get_color, savefig, set_style  # noqa: E402


set_style()


DISPLAY_NAMES = {
    "sex": "Sex",
    "age": "Age",
    "tusks": "Tusks",
    "tear_1": "Tear 1",
    "tear_2": "Tear 2",
    "hole_1": "Hole 1",
    "hole_2": "Hole 2",
    "extreme_special": "Extreme &\nspecial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    return parser.parse_args()


def render_grouped_bars(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    out_path: Path,
    levels: list[int],
    *,
    x_axis_at_zero: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.85, 3.0))

    labels = [DISPLAY_NAMES.get(name, name.replace("_", " ").title()) for name in GROUP_NAMES]
    x = np.arange(len(GROUP_NAMES))
    width = min(0.18, 0.78 / max(len(levels), 1))
    offsets = (np.arange(len(levels)) - (len(levels) - 1) / 2) * width

    vmax = df[value_col].max()
    vmin = df[value_col].min()
    if np.isfinite(vmin) and vmin < 0:
        pad = max(abs(vmin), abs(vmax)) * 0.1
        ylim = (vmin - pad, vmax + pad)
    else:
        ylim = (0, vmax * 1.12 if np.isfinite(vmax) and vmax > 0 else 1)

    for offset, level_idx in zip(offsets, range(len(levels))):
        level = levels[level_idx]
        sub = df[df["level"] == level].set_index("concept").reindex(GROUP_NAMES)
        ax.bar(
            x + offset,
            sub[value_col].to_numpy(),
            color=get_color(level_idx),
            width=width,
            label=f"{level}%",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if x_axis_at_zero:
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["bottom"].set_linewidth(0.8)
    else:
        ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(title="Correction", ncol=len(levels), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    savefig(fig, out_path)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    geometry = pd.read_csv(args.metrics_dir / "geometry.csv")
    single = pd.read_csv(args.metrics_dir / "single_concept_identity.csv")
    leave_one = pd.read_csv(args.metrics_dir / "leave_one_out.csv")

    render_grouped_bars(
        geometry,
        "within_cosine_distance",
        "Mean within-group cosine distance",
        SCRIPT_DIR / "geometry" / "concept_geometry.pdf",
        args.levels,
    )
    render_grouped_bars(
        single,
        "identity_accuracy_pct",
        "Identity accuracy (%)",
        SCRIPT_DIR / "single-concept-identity" / "single_concept_identity.pdf",
        args.levels,
    )
    render_grouped_bars(
        leave_one,
        "delta_pct",
        "Combinatorial accuracy gain",
        SCRIPT_DIR / "leave-one-out" / "leave_one_out.pdf",
        args.levels,
        x_axis_at_zero=True,
    )


if __name__ == "__main__":
    main()
