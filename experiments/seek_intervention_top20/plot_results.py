"""Image-weighted Recall@1 with identity-cluster and random-seed bootstrap CIs."""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICIES = {
    "random": ("Random (20 seeds)", "#777777"),
    "entropy_full": ("Entropy: all identities", "#bd4770"),
    "entropy": ("Entropy: fixed initial top-20", "#1976b9"),
    "margin": ("Margin-ratio", "#008477"),
    "oracle": ("Oracle", "#9b7317"),
}
FINAL_POLICIES = ("random", "margin", "oracle")


def cluster_totals(hits, labels):
    _, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    totals = np.zeros((len(hits), len(sizes), hits.shape[2]), dtype=float)
    for run, values in enumerate(hits):
        np.add.at(totals[run], inverse, values)
    return totals, sizes


def bootstrap_curves(hits, labels, draws=4000, seed=2026):
    """Resample whole identities and, independently, random-policy rollouts."""
    totals, sizes = cluster_totals(hits, labels)
    clusters = len(sizes)
    rng_clusters = np.random.default_rng(seed)
    rng_runs = np.random.default_rng(seed + 1)
    curves = np.empty((draws, hits.shape[2]))
    seed_only = np.empty_like(curves)
    per_run = hits.mean(axis=1) * 100
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        weights = rng_clusters.multinomial(clusters, np.full(clusters, 1 / clusters), size=count)
        run_weights = rng_runs.multinomial(len(hits), np.full(len(hits), 1 / len(hits)), size=count) / len(hits)
        sampled_totals = np.einsum("dr,rcb->dcb", run_weights, totals)
        curves[start:start+count] = 100 * np.einsum("dc,dcb->db", weights, sampled_totals) / (weights @ sizes)[:, None]
        seed_only[start:start+count] = run_weights @ per_run
    return np.percentile(curves, [2.5, 97.5], axis=0), np.percentile(seed_only, [2.5, 97.5], axis=0)


def plot_results(csv_path, pdf_path, png_path, *, include_entropy=False):
    csv_path, pdf_path, png_path = map(Path, (csv_path, pdf_path, png_path))
    folder = csv_path.parent
    frame = pd.read_csv(csv_path)
    with np.load(folder / "query_metadata.npz", allow_pickle=False) as metadata:
        labels = metadata["labels"]
    with (folder / "prioritization_results.json").open() as handle:
        config = json.load(handle)["config"]
    output = []
    fig, ax = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    for key, (name, color) in POLICIES.items():
        subset = frame[frame.policy_key == key]
        if subset.empty:
            continue
        seeds = sorted(subset.seed.unique())
        arrays = []
        for run_seed in seeds:
            with np.load(folder / f"trace_{key}_{run_seed}.npz", allow_pickle=False) as trace:
                arrays.append(trace["hits"])
        hits = np.stack(arrays).astype(float)
        estimates = hits.mean(axis=(0, 1)) * 100
        budgets = np.sort(subset.budget.unique())
        assert hits.shape[1] == len(labels)
        assert len(budgets) == hits.shape[2]
        assert np.allclose(estimates, subset.groupby("budget").recall_at_1.mean(), atol=1e-5)
        interval, seed_interval = bootstrap_curves(hits, labels, config.get("bootstrap_draws", 4000), config.get("bootstrap_seed", 2026))
        for j, budget in enumerate(budgets):
            output.append({"policy_key": key, "budget": budget, "recall_at_1": estimates[j],
                           "ci95_low": interval[0, j], "ci95_high": interval[1, j],
                           "seed_only_ci95_low": seed_interval[0, j], "seed_only_ci95_high": seed_interval[1, j],
                           "n_seeds": len(seeds), "n_identities": len(np.unique(labels)), "n_images": len(labels)})
        if not include_entropy and key not in FINAL_POLICIES:
            continue
        label = "Random" if key == "random" else name
        ax.plot(budgets, estimates, marker="o", markersize=3.5, linewidth=1.9, color=color, label=label)
        ax.fill_between(budgets, interval[0], interval[1], color=color, alpha=.10)
        # Draw interval bounds directly; percentile intervals need not enclose the point estimate.
        ax.vlines(budgets, interval[0], interval[1], color=color, linewidth=.9)
        ax.hlines(interval[0], budgets-.08, budgets+.08, color=color, linewidth=.9)
        ax.hlines(interval[1], budgets-.08, budgets+.08, color=color, linewidth=.9)
    intervals = pd.DataFrame(output)
    intervals.to_csv(folder / "confidence_intervals.csv", index=False)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.parent.resolve() != folder.resolve():
        selected = intervals if include_entropy else intervals[intervals.policy_key.isin(FINAL_POLICIES)]
        selected.to_csv(pdf_path.parent / "confidence_intervals.csv", index=False)
    ax.set_xlabel("# corrected attributes per query", fontsize=16, labelpad=7)
    ax.set_ylabel("Recall@1 (%)", fontsize=16, labelpad=7)
    ax.set(xticks=sorted(frame.budget.unique())[::2], ylim=(0, 100))
    ax.tick_params(axis="both", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=13, loc="upper left", handlelength=1.8)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output-dir", type=Path, help="Export figure and displayed-policy intervals to this directory.")
    parser.add_argument("--include-entropy", action="store_true", help="Include the two historical entropy comparisons.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    plot_results(args.results_dir / "prioritization_results.csv", output_dir / "intervention_prioritization.pdf",
                 output_dir / "intervention_prioritization.png", include_entropy=args.include_entropy)
