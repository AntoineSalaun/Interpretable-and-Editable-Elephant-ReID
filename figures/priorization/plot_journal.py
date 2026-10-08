"""Render the journal figure from the already exported confidence intervals."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from figures.journal_style import (
    set_journal_style, journal_figsize, format_journal_axes, save_journal, LEGEND_SIZE,
)
from experiments.seek_intervention_top20.plot_results import POLICIES, FINAL_POLICIES


def main():
    folder = Path(__file__).resolve().parent
    intervals = pd.read_csv(folder / "confidence_intervals.csv")
    set_journal_style()
    fig, ax = plt.subplots(figsize=journal_figsize(0.75))
    for key in FINAL_POLICIES:
        subset = intervals[intervals.policy_key == key].sort_values("budget")
        if subset.empty:
            raise ValueError(f"Missing confidence intervals for {key}")
        name, color = POLICIES[key]
        budgets = subset.budget.to_numpy()
        low = subset.ci95_low.to_numpy()
        high = subset.ci95_high.to_numpy()
        ax.plot(budgets, subset.recall_at_1.to_numpy(), marker="o", markersize=6.5,
                linewidth=1.9, color=color, label="Random" if key == "random" else name)
        ax.fill_between(budgets, low, high, color=color, alpha=0.10)
        ax.vlines(budgets, low, high, color=color, linewidth=0.9)
        ax.hlines(low, budgets - 0.08, budgets + 0.08, color=color, linewidth=0.9)
        ax.hlines(high, budgets - 0.08, budgets + 0.08, color=color, linewidth=0.9)
    ax.set_xlabel("# corrected attributes per query")
    ax.set_ylabel("Recall@1 (%)")
    ax.set(xticks=sorted(intervals.budget.unique())[::2], ylim=(0, 100))
    format_journal_axes(ax)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left", handlelength=2)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.155, top=0.965)
    save_journal(fig, folder / "intervention_prioritization_journal.pdf")


if __name__ == "__main__":
    main()
