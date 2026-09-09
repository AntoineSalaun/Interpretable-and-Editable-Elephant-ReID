#!/usr/bin/env python3
"""Plot Recall@1 against query-sighting correction budget."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from figures.style import get_color, savefig, set_style
except Exception:  # pragma: no cover - fallback for minimal environments
    get_color = None

    def set_style() -> None:
        return None

    def savefig(fig, path) -> None:
        fig.savefig(path, bbox_inches="tight")


POLICY_ORDER = [
    "Random",
    "Expected entropy reduction",
    "Margin-ratio",
    "Oracle",
]


def plot_results(csv_path: Path, pdf_path: Path, png_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Results CSV is empty: {csv_path}")

    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for idx, policy in enumerate(POLICY_ORDER):
        policy_df = df[df["policy"] == policy]
        if policy_df.empty:
            continue
        grouped = (
            policy_df.groupby("budget", as_index=False)
            .agg(
                recall_at_1=("recall_at_1", "mean"),
                recall_at_1_std=("recall_at_1", "std"),
            )
            .sort_values("budget")
        )
        color = get_color(idx) if get_color is not None else None
        ax.plot(
            grouped["budget"],
            grouped["recall_at_1"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=policy,
            color=color,
        )
        if policy == "Random" and grouped["recall_at_1_std"].notna().any():
            std = grouped["recall_at_1_std"].fillna(0.0)
            ax.fill_between(
                grouped["budget"],
                grouped["recall_at_1"] - std,
                grouped["recall_at_1"] + std,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax.set_xlabel("Corrected SEEK attributes per query sighting")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks(sorted(df["budget"].unique()))
    ax.set_xlim(df["budget"].min(), df["budget"].max())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, pdf_path)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "prioritization_results.csv",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "intervention_prioritization.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "intervention_prioritization.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_results(args.results, args.pdf, args.png)


if __name__ == "__main__":
    main()

