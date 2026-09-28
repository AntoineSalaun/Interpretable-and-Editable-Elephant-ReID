"""Summarize saved intervention orders without rerunning or changing retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_mean_interval(values, rng, draws=4000):
    values = np.asarray(values, dtype=float)
    means = np.empty(draws)
    for start in range(0, draws, 100):
        indices = rng.integers(0, len(values), size=(min(100, draws - start), len(values)))
        means[start:start + len(indices)] = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    source = args.results_dir
    output = source / "concept_analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "prioritization_results.json").read_text())
    with (source / "per_sighting_interventions.json").open() as handle:
        runs = json.load(handle)
    margin_runs = [v for k, v in runs.items() if k.startswith("margin:")]
    if len(margin_runs) != 1:
        raise ValueError("Expected exactly one margin run")
    margin = margin_runs[0]
    random_runs = [v for k, v in runs.items() if k.startswith("random:")]
    sightings = list(margin)
    n = len(sightings)
    attributes = [r["attribute_name"] for r in sorted(next(iter(margin.values())), key=lambda r: r["attribute_index"])]
    rows = []
    reranked = 0
    uncertainty_agreement = 0
    uncertainty_range = []
    importance_range = []
    for sighting, records in margin.items():
        assert [r["budget"] for r in records] == list(range(1, 17))
        assert len({r["attribute_name"] for r in records}) == 16
        initial = records[0]
        initial_order = sorted(initial["scores_by_attribute"], key=lambda a: (-initial["scores_by_attribute"][a], attributes.index(a)))
        assert initial_order[0] == initial["attribute_name"]
        reranked += records[1]["attribute_name"] != initial_order[1]
        diagnostics = initial["policy_diagnostics"]["attributes"]
        most_uncertain = max(attributes, key=lambda a: diagnostics[a]["margin_concept_uncertainty"])
        uncertainty_agreement += initial["attribute_name"] == most_uncertain
        uncertainty_range.append(np.ptp([diagnostics[a]["margin_concept_uncertainty"] for a in attributes]))
        importance_range.append(np.ptp([initial["margin_lambda"] * diagnostics[a]["margin_importance"] for a in attributes]))
        for j, record in enumerate(records):
            rows.append({
                "sighting": sighting,
                "budget": record["budget"],
                "attribute": record["attribute_name"],
                "recall_at_1_pct": 100 * record["true_recall_at_1"],
                "gain_pp": np.nan if j == 0 else 100 * (record["true_recall_at_1"] - records[j-1]["true_recall_at_1"]),
                "concept_uncertainty": record["margin_concept_uncertainty"],
                "margin_importance": record["margin_importance"],
                "weighted_margin_importance": record["margin_lambda"] * record["margin_importance"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "margin_decisions.csv", index=False)
    counts = pd.crosstab(frame.attribute, frame.budget).reindex(index=attributes, columns=range(1, 17), fill_value=0)
    assert (counts.sum(axis=0) == n).all()
    assert (counts.sum(axis=1) == n).all()
    ranking = pd.DataFrame(index=attributes)
    for budget in [1, 2]:
        ranking[f"position_{budget}_count"] = counts[budget]
        ranking[f"position_{budget}_pct"] = 100 * counts[budget] / n
    for budget in [2, 4]:
        ranking[f"included_by_{budget}_pct"] = 100 * counts.loc[:, :budget].sum(axis=1) / n
    ranking["mean_position"] = frame.groupby("attribute").budget.mean()
    ranking = ranking.sort_values("position_1_count", ascending=False)
    ranking.index.name = "attribute"
    ranking.to_csv(output / "concept_ranking.csv")
    (100 * counts / n).to_csv(output / "position_frequencies_pct.csv")
    pair_rows = [{"first": r[0]["attribute_name"], "second": r[1]["attribute_name"]} for r in margin.values()]
    pairs = pd.DataFrame(pair_rows).value_counts().rename("count").reset_index()
    pairs["pct_all_sightings"] = 100 * pairs["count"] / n
    pairs["pct_given_first"] = [100 * row["count"] / counts.loc[row["first"], 1] for _, row in pairs.iterrows()]
    pairs.to_csv(output / "first_second_pairs.csv", index=False)
    gain_rows = []
    for (budget, attr), group in frame[frame.budget > 1].groupby(["budget", "attribute"]):
        gains = group.gain_pp.to_numpy()
        gain_rows.append({"budget": budget, "attribute": attr, "n_sightings": len(gains),
                          "mean_gain_pp": gains.mean(), "median_gain_pp": np.median(gains),
                          "improved_pct": 100 * np.mean(gains > 1e-7),
                          "unchanged_pct": 100 * np.mean(np.abs(gains) <= 1e-7),
                          "worsened_pct": 100 * np.mean(gains < -1e-7)})
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(output / "observed_gains_by_position.csv", index=False)
    rng = np.random.default_rng(42)
    oracle_runs = [v for k, v in runs.items() if k.startswith("oracle:")]
    if len(oracle_runs) != 1:
        raise ValueError("Expected one oracle run with first-step counterfactual outcomes")
    oracle = oracle_runs[0]
    # All first-step oracle candidates share the same uncorrected starting state.
    first_outcomes = np.array([
        [oracle[s][0]["policy_diagnostics"]["values"][a]["true_recall_at_1"] for a in attributes]
        for s in sightings
    ]) * 100
    for row_index, sighting in enumerate(sightings):
        for run in [margin, *random_runs]:
            record = run[sighting][0]
            expected = first_outcomes[row_index, attributes.index(record["attribute_name"])]
            assert np.isclose(100 * record["true_recall_at_1"], expected, rtol=0, atol=1e-5)
        assert np.isclose(100 * oracle[sighting][0]["true_recall_at_1"], first_outcomes[row_index].max(), rtol=0, atol=1e-5)
    uniform_random = first_outcomes.mean(axis=1)
    controlled_rows = []
    for j, attr in enumerate(attributes):
        differences = first_outcomes[:, j] - uniform_random
        lo, hi = bootstrap_mean_interval(differences, rng)
        controlled_rows.append({"attribute": attr, "n_sightings": n,
                                "mean_first_r1_pct": first_outcomes[:, j].mean(),
                                "gain_over_uniform_random_pp": differences.mean(),
                                "sighting_bootstrap_95_low": lo,
                                "sighting_bootstrap_95_high": hi,
                                "tied_best_first_pct": 100 * np.mean(np.isclose(first_outcomes[:, j], first_outcomes.max(axis=1), rtol=0, atol=1e-7))})
    controlled = pd.DataFrame(controlled_rows).sort_values("mean_first_r1_pct", ascending=False)
    controlled.to_csv(output / "controlled_first_concept_comparison.csv", index=False)
    pd.DataFrame(first_outcomes, index=sightings, columns=attributes).rename_axis("sighting").to_csv(output / "first_concept_counterfactual_r1_pct.csv")
    comparisons = []
    for budget in [1, 2, 4, 8, 16]:
        m = np.array([margin[s][budget-1]["true_recall_at_1"] for s in sightings]) * 100
        r = np.array([[run[s][budget-1]["true_recall_at_1"] for s in sightings] for run in random_runs]).mean(axis=0) * 100
        lo, hi = bootstrap_mean_interval(m-r, rng)
        comparisons.append({"budget": budget, "margin_sighting_mean_r1_pct": m.mean(),
                            "random_sighting_mean_r1_pct": r.mean(), "paired_difference_pp": (m-r).mean(),
                            "sighting_bootstrap_95_low": lo, "sighting_bootstrap_95_high": hi})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "paired_policy_comparison.csv", index=False)
    curves = pd.read_csv(source / "prioritization_results.csv")
    global_r1 = curves.groupby(["policy_key", "budget"]).recall_at_1.mean()
    second_gains = frame[frame.budget == 2].gain_pp.to_numpy()
    stats = {
        "n_sightings": n, "random_rollouts": len(random_runs),
        "margin_lambda": summary["margin_lambda"]["margin_lambda"],
        "temperature": summary["temperature"]["temperature"],
        "exact_uniform_random_first_sighting_r1_pct": float(uniform_random.mean()),
        "second_choice_changed_from_initial_runner_up_count": int(reranked),
        "second_choice_changed_from_initial_runner_up_pct": 100 * reranked / n,
        "first_choice_matches_max_concept_entropy_pct": 100 * uncertainty_agreement / n,
        "mean_initial_uncertainty_range": float(np.mean(uncertainty_range)),
        "mean_initial_weighted_importance_range": float(np.mean(importance_range)),
        "second_correction_sighting_mean_gain_pp": float(second_gains.mean()),
        "second_correction_improved_pct": float(100 * np.mean(second_gains > 1e-7)),
        "second_correction_unchanged_pct": float(100 * np.mean(np.abs(second_gains) <= 1e-7)),
        "second_correction_worsened_pct": float(100 * np.mean(second_gains < -1e-7)),
        "all_sequences_have_16_unique_concepts": True,
        "first_counterfactuals_match_margin_and_all_random_runs": True,
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")

    order = ranking.index.tolist()
    labels = [a.replace("_", " ") for a in order]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), layout="constrained")
    y = np.arange(16)
    axes[0].barh(y - .19, ranking.position_1_pct, height=.36, color="#087e8b", label="First correction")
    axes[0].barh(y + .19, ranking.position_2_pct, height=.36, color="#ba4164", label="Second correction")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Sightings (%)", title="Which concept is selected?")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=.2)
    matrix = 100 * counts.loc[order, 1:8].to_numpy() / n
    im = axes[1].imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    axes[1].set(xticks=range(8), xticklabels=range(1, 9), yticks=y, yticklabels=labels,
                xlabel="Correction position", title="Selection frequency by position")
    for row in range(16):
        for col in range(8):
            if matrix[row, col] >= 1:
                axes[1].text(col, row, f"{matrix[row,col]:.0f}", ha="center", va="center", fontsize=8,
                             color="white" if matrix[row,col] > matrix.max() * .55 else "black")
    fig.colorbar(im, ax=axes[1], location="bottom", shrink=.8, label="Sightings (%)")
    second = gains[(gains.budget == 2) & (gains.n_sightings >= 20)].sort_values("mean_gain_pp", ascending=False)
    axes[2].barh(np.arange(len(second)), second.mean_gain_pp, color="#587b3c")
    axes[2].set(yticks=np.arange(len(second)),
                yticklabels=[f"{r.attribute.replace('_', ' ')} (n={r.n_sightings})" for r in second.itertuples()],
                xlabel="Mean change in sighting Recall@1 (pp)", title="Observed gain when selected second")
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="black", linewidth=.7)
    axes[2].grid(axis="x", alpha=.2)
    fig.suptitle(f"Margin policy: {n:,} test sightings, lambda={stats['margin_lambda']:g}\n"
                 "Each sighting has equal weight; second-step gains are conditional on the selected sequence", fontsize=14)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"concept_choices.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    means = controlled.gain_over_uniform_random_pp.to_numpy()
    errors = np.vstack([means - controlled.sighting_bootstrap_95_low, controlled.sighting_bootstrap_95_high - means])
    ax.barh(np.arange(16), means, color=["#087e8b" if value >= 0 else "#ba4164" for value in means])
    ax.errorbar(means, np.arange(16), xerr=errors, fmt="none", ecolor="black", capsize=3)
    ax.set(yticks=np.arange(16), yticklabels=[a.replace("_", " ") for a in controlled.attribute],
           xlabel="First-correction Recall@1 minus uniform random (percentage points)",
           title=f"Which first correction actually helps most?\nSame {n:,} sightings for every concept; equal sighting weight")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=.8)
    ax.grid(axis="x", alpha=.2)
    fig.supxlabel("Saved oracle counterfactuals; exploratory 95% sighting-bootstrap intervals", fontsize=10)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"controlled_first_concept.{extension}", dpi=180)
    plt.close(fig)

    report = ["# Margin-policy concept analysis", "",
              f"Full saved test run: {n:,} sightings; {len(random_runs)} random rollouts; "
              f"validation-selected lambda = {stats['margin_lambda']}; retrieval temperature = {stats['temperature']:.8f}.", "",
              "## What stays fixed?", "",
              "Corrections are cumulative within each sighting. At budget 2 the first corrected concept and its expert value remain fixed, "
              f"and a distinct second concept is added. All {n:,} sequences were checked for 16 unique concepts in budgets 1..16. "
              "The runner clones the current SEEK vector, overwrites only the newly selected attribute, and masks corrected attributes out of selection. "
              "Different sightings can have different first concepts. This is a sequential greedy policy, not a separately optimized subset at each budget.", "",
              f"The actual second choice differs from the initial second-ranked score in {reranked}/{n} sightings "
              f"({100*reranked/n:.1f}%). This measures reranking after the first correction; it does not mean the first correction was undone.", "",
              "## Selection frequencies", "",
              "Percentages below count sightings, not images. First and second columns each sum to 100%; inclusion by budget 2 sums to 200%.", "",
              "| Concept | First: count (%) | Second: count (%) | Included by 2 | Included by 4 | Mean position |",
              "|---|---:|---:|---:|---:|---:|"]
    for a, row in ranking.iterrows():
        report.append(f"| {a} | {int(row.position_1_count)} ({row.position_1_pct:.1f}%) | "
                      f"{int(row.position_2_count)} ({row.position_2_pct:.1f}%) | {row.included_by_2_pct:.1f}% | "
                      f"{row.included_by_4_pct:.1f}% | {row.mean_position:.2f} |")
    report += ["", "R/L are the right/left ear attribute prefixes; the 1/2 suffix is the SEEK feature slot, not intervention order.", "",
               "## Common ordered pairs", "", "| First | Second | Count | All sightings | Given this first concept |",
               "|---|---|---:|---:|---:|"]
    for r in pairs.head(12).itertuples():
        report.append(f"| {r.first} | {r.second} | {r.count} | {r.pct_all_sightings:.1f}% | {r.pct_given_first:.1f}% |")
    report += ["", "## Controlled first-concept comparison", "",
               "The first-step oracle diagnostics contain actual expert-correction outcomes for ALL 16 concepts on EVERY sighting, "
               "before any correction is applied. This supports a matched comparison of first concepts on identical initial states. "
               "For concept i, average R@1(s, i) across sightings. The reference for each sighting is the exact mean over its 16 possible "
               "first corrections, which is the expected performance of a uniform random first choice. "
               "The paired effect is mean_s [R@1(s, i) - mean_j R@1(s, j)], in percentage points. "
               "This is a controlled effect of simulated expert correction within this fixed model and gallery. "
               "It does not measure human annotation accuracy or intrinsic concept importance outside this setup. "
               "The oracle labels are used only for this post-hoc analysis; they were not supplied to margin selection.", "",
               f"Exact uniform-random first-correction reference: {uniform_random.mean():.2f}% sighting-mean Recall@1. "
               "Intervals use 4,000 paired sighting-bootstrap samples and are pointwise, not adjusted for 16 comparisons; "
               "they are not clustered by identity. Small differences between concepts should not be overinterpreted.", "",
               "| First concept | Mean R@1 after correction | Difference vs uniform random (pp) | 95% interval |",
               "|---|---:|---:|---:|"]
    for r in controlled.itertuples():
        report.append(f"| {r.attribute} | {r.mean_first_r1_pct:.2f}% | {r.gain_over_uniform_random_pp:+.2f} | "
                      f"[{r.sighting_bootstrap_95_low:+.2f}, {r.sighting_bootstrap_95_high:+.2f}] |")
    report += ["", "## Retrieval benefit", "",
               "The original curve is query-image-weighted ReID Recall@1 using fused visual and SEEK embeddings against fixed G_S(20). "
               "The table below retains that original weighting. Differences are percentage points (pp).", "",
               "| Budget | Margin R@1 | Random mean R@1 | Difference |", "|---|---:|---:|---:|"]
    for b in [0, 1, 2, 4, 8, 16]:
        m, r = global_r1.loc["margin", b], global_r1.loc["random", b]
        report.append(f"| {b} | {m:.2f}% | {r:.2f}% | {m-r:+.2f} pp |")
    report += ["", "For concept-specific changes below, each sighting has equal weight. "
               "Gain is 100 * (sighting R@1 after correction 2 - after correction 1). "
               "These are observed gains among sightings where the policy selected that concept second. "
               "They do not compare concepts on identical sightings or first-correction histories and therefore are not a causal ranking.", "",
               "| Second concept | Sightings | Mean gain (pp) | Median gain (pp) | Improved | Unchanged | Worsened |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for r in gains[gains.budget == 2].sort_values("n_sightings", ascending=False).itertuples():
        report.append(f"| {r.attribute} | {r.n_sightings} | {r.mean_gain_pp:+.2f} | {r.median_gain_pp:+.2f} | "
                      f"{r.improved_pct:.1f}% | {r.unchanged_pct:.1f}% | {r.worsened_pct:.1f}% |")
    report += ["", f"Across all sightings, the second correction has mean gain {second_gains.mean():+.2f} pp; "
               f"{stats['second_correction_improved_pct']:.1f}% improve, "
               f"{stats['second_correction_unchanged_pct']:.1f}% stay unchanged, and "
               f"{stats['second_correction_worsened_pct']:.1f}% worsen.", "",
               "## Paired uncertainty estimates", "",
               f"For each sighting, compare margin to its mean outcome across {len(random_runs)} random rollouts; then average the paired differences. "
               "95% percentile intervals use 4,000 bootstrap resamples of whole sightings (seed 42). "
               "These use equal sighting weighting, so they differ from the original image-weighted curve. "
               "They treat sightings as independent and are conditional on the trained model, gallery, and saved random rollouts. "
               "Multiple sightings can share an identity, so these are exploratory sighting-level intervals, not identity-clustered or training-seed uncertainty estimates.", "",
               "| Budget | Margin minus random (pp) | 95% sighting-bootstrap interval |", "|---|---:|---:|"]
    for r in comparison.itertuples():
        report.append(f"| {r.budget} | {r.paired_difference_pp:+.2f} | [{r.sighting_bootstrap_95_low:+.2f}, {r.sighting_bootstrap_95_high:+.2f}] |")
    report += ["", "## Interpretation and limitations", "",
               f"The score is concept entropy + {stats['margin_lambda']:g} * expected retrieval-margin gain. "
               f"The first selected concept matches the maximum concept-entropy attribute in {stats['first_choice_matches_max_concept_entropy_pct']:.1f}% of sightings. "
               f"At budget 0, the mean across-candidate score range is {np.mean(uncertainty_range):.3f} for entropy "
               f"and {np.mean(importance_range):.3f} for weighted margin importance. These ranges describe scale, not a causal attribution of selection.", "",
               "All eight ear-hole/tear attributes have five categories, whereas other concepts have two or three. "
               "Raw entropy has maximum log(number of categories), so this policy has a larger possible uncertainty score for those ear concepts. "
               "Frequent selection reflects both uncertainty and expected importance; it must not be read as intrinsic biological or retrieval importance.", "",
               "The logs do not store per-sighting budget-0 true Recall@1, so first-step per-concept actual gains cannot be reconstructed exactly here. "
               "Specifically, gains relative to NO correction are unavailable per sighting; the controlled first-concept comparison above "
               "instead measures gains relative to an exactly averaged random first correction. "
               "The overall first-step gain can be computed from the aggregate curve. "
               "Controlled second-concept effects under the MARGIN first choice would require additional counterfactual outcomes with that first correction fixed. "
               "The saved oracle second-step candidates use the oracle's own first choice, so they are not interchangeable. This report analyzes the existing run only.", "",
               "## Reproduce", "", "```bash", "python experiments/seek_intervention_prioritization/analyze_concept_choices.py", "```", "",
               "Inputs: ../per_sighting_interventions.json, ../prioritization_results.csv, ../prioritization_results.json. "
               "Outputs include concept_choices.png/pdf, concept_ranking.csv, first_second_pairs.csv, "
               "position_frequencies_pct.csv, observed_gains_by_position.csv, paired_policy_comparison.csv, margin_decisions.csv, summary.json, "
               "controlled_first_concept.png/pdf, controlled_first_concept_comparison.csv, first_concept_counterfactual_r1_pct.csv.", ""]
    (output / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(stats, indent=2))
    print(ranking.head(8).to_string())
    print(pairs.head(8).to_string(index=False))
    print(controlled.to_string(index=False))
    print(f"Report and plots: {output}")


if __name__ == "__main__":
    main()
