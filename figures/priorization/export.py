"""Export the final margin figure and verify its ear-concept evidence."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.seek_intervention_top20.plot_results import plot_results


def main():
    source = ROOT / "experiments/seek_intervention_top20/results"
    output = Path(__file__).resolve().parent
    summary = json.loads((source / "prioritization_results.json").read_text())
    metadata = json.loads((source / "run_metadata.json").read_text())
    rescaled = ROOT / "experiments/seek_margin_rescaled/results"
    audit_path = rescaled / "rescaling_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    plot_source = source
    if audit is not None:
        if not audit["all_image_hits_identical"] or audit["raw_vs_original_choice_differences"] != 0:
            raise RuntimeError("Raw-margin trajectories differ; update the concept evidence before exporting the paper figure")
        plot_source = rescaled
        (output / "rescaling_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Exporting the three-policy figure and 95% intervals", flush=True)
    plot_results(plot_source / "prioritization_results.csv", output / "intervention_prioritization.pdf",
                 output / "intervention_prioritization.png")
    print("Checking ear-concept statistics against the completed full run", flush=True)
    with (source / "per_sighting_interventions.json").open() as handle:
        runs = json.load(handle)
    seed = summary["config"]["seed"]
    margin, oracle = runs[f"margin:{seed}"], runs[f"oracle:{seed}"]
    sightings = list(margin)
    ears = {f"{side}_{kind}_{slot}" for side in ("L", "R") for kind in ("hole", "tear") for slot in (1, 2)}
    attributes = list(oracle[sightings[0]][0]["policy_diagnostics"]["values"])
    assert len(attributes) == 16 and ears.issubset(attributes)
    first_five = []
    for sighting in sightings:
        records = margin[sighting]
        assert [r["budget"] for r in records] == list(range(1, 17))
        assert len({r["attribute_name"] for r in records}) == 16
        first_five.extend(r["attribute_name"] for r in records[:5])
    ear_count = sum(attr in ears for attr in first_five)
    outcomes = np.array([[oracle[s][0]["policy_diagnostics"]["values"][a]["true_recall_at_1"]
                          for a in attributes] for s in sightings]) * 100
    for j, sighting in enumerate(sightings):
        chosen = margin[sighting][0]
        assert np.isclose(outcomes[j, attributes.index(chosen["attribute_name"])],
                          100 * chosen["true_recall_at_1"], rtol=0, atol=1e-5)
    uniform_random = outcomes.mean(axis=1)
    controlled = pd.DataFrame({"attribute": attributes, "mean_first_r1_pct": outcomes.mean(axis=0),
                               "gain_vs_uniform_random_pp": (outcomes - uniform_random[:, None]).mean(axis=0)})
    controlled = controlled.sort_values("mean_first_r1_pct", ascending=False)
    controlled.to_csv(output / "first_concept_effects.csv", index=False)
    results = pd.read_csv(source / "prioritization_results.csv")
    early = results[results.budget.between(1, 4)].groupby("policy_key").recall_at_1.mean()
    evidence = {
        "source_results": str(source.relative_to(ROOT)),
        "plot_results_source": str(plot_source.relative_to(ROOT)),
        "rescaling_audit": audit,
        "slurm_job_id": metadata["slurm_job_id"], "wandb_url": metadata["wandb_url"],
        "margin_lambda": summary["margin_lambda"]["margin_lambda"],
        "temperature": summary["temperature"]["temperature"],
        "n_sightings": len(sightings), "first_five_ear_corrections": ear_count,
        "first_five_total_corrections": len(first_five),
        "first_five_ear_pct": 100 * ear_count / len(first_five),
        "mean_first_r1_uniform_random_sighting_weighted_pct": float(uniform_random.mean()),
        "top_eight_first_concepts_are_all_ear_holes_or_tears": set(controlled.head(8).attribute) == ears,
        "mean_r1_budgets_1_to_4_image_weighted_pct": early.to_dict(),
        "first_concept_effects": controlled.to_dict(orient="records"),
    }
    with np.load(source / "query_metadata.npz") as query:
        evidence.update(n_images=len(query["labels"]), n_identities=len(np.unique(query["labels"])))
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Ear holes/tears: {ear_count}/{len(first_five)} = {evidence['first_five_ear_pct']:.5f}%")
    print(controlled.head(4).to_string(index=False))
    print(f"Exported to {output}", flush=True)


if __name__ == "__main__":
    main()
