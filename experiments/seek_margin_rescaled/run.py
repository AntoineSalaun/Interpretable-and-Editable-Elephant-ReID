"""Rerun raw-margin selection, audit equivalence, and retain original baselines."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import wandb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.seek_intervention_top20 import run_prioritization as base
from experiments.seek_intervention_top20.plot_results import plot_results
from experiments.seek_margin_rescaled.policy import RawMarginPolicy, raw_margin, rescaled_weight
from experiments.seek_intervention_prioritization.policies import PolicyState, copy_attribute_from_expert
from experiments.seek_intervention_prioritization.retrieval_uncertainty import IdentityScorePooler, top_two_log_margin


def score_variants(projector, retrieval, visual, variants, gallery, pooler, tau, eps, audit):
    n = len(visual)
    seeks = torch.cat(variants)
    visuals = visual.repeat(len(variants), 1)
    raw, legacy = [], []
    with torch.inference_mode():
        for start in range(0, len(seeks), 256):
            # Batch independent hypothetical corrections; normalize features per image as in the original runner.
            v = visuals[start:start+256].to(projector.device)
            p = projector.layer(seeks[start:start+256].to(projector.device))
            if projector.layer_norm:
                v, p = F.layer_norm(v, v.shape[1:]), F.layer_norm(p, p.shape[1:])
            embedding = (1-base.alpha_value(projector))*v + base.alpha_value(projector)*p
            similarity = retrieval.similarity_matrix(embedding, gallery)
            _, scores = pooler.score_matrix(similarity)
            gaps = raw_margin(scores)
            raw.append(gaps.cpu())
            # Only the audit uses softmax/tau. These values never enter raw-policy selection.
            probabilities = torch.softmax(scores / tau, dim=1)
            log_gaps = top_two_log_margin(probabilities, eps)
            legacy.append(log_gaps.cpu())
            if probabilities.shape[1] > 1:
                top2 = probabilities.topk(2, dim=1).values[:, 1]
                audit["clipped_top2_rows"] += int((top2 < eps).sum().item())
                audit["min_top2_probability"] = min(audit["min_top2_probability"], top2.min().item())
            audit["counterfactual_image_rows"] += len(gaps)
            audit["max_per_image_margin_error"] = max(audit["max_per_image_margin_error"], (gaps/tau-log_gaps).abs().max().item())
    raw = torch.cat(raw).reshape(-1, n).mean(dim=1).tolist()
    legacy = torch.cat(legacy).reshape(-1, n).mean(dim=1).tolist()
    return [{"raw_margin": r, "legacy_margin": l} for r, l in zip(raw, legacy)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/seek_intervention_top20/results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--max-sightings", type=int)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "disabled", "offline"])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if source == output:
        raise ValueError("Use an isolated output directory")
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / "prioritization_results.json").read_text())
    config = copy.deepcopy(saved["config"])
    tau, old_lambda = saved["temperature"]["temperature"], saved["margin_lambda"]["margin_lambda"]
    beta = rescaled_weight(old_lambda, tau)
    config.update(experiment_name="seek_margin_rescaled", output_dir=str(output), margin_beta=beta,
                  margin_definition="raw top-two identity similarity gap", normalization="raw concept entropy",
                  max_sightings=args.max_sightings)
    started = time.time()
    audit = dict(original_lambda=old_lambda, original_temperature=tau, margin_beta=beta,
                 clipped_top2_rows=0, min_top2_probability=1., counterfactual_image_rows=0,
                 max_per_image_margin_error=0., max_policy_score_error=0.,
                 raw_vs_live_legacy_choice_differences=0)
    with wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"],
                    group="seek_margin_rescaled", name="raw_margin_equivalence", config=config, mode=args.wandb_mode):
        print(f"Raw-gap beta={beta:.12f}; audit tau={tau}; original lambda={old_lambda}", flush=True)
        dataset = base.EleHandler(subset=config["subset"], dataset_type=config["dataset"])
        train_indices, test_indices = base.load_indices(base.repo_path(config["train_indices_path"]), base.repo_path(config["test_indices_path"]))
        test_indices = base.select_indices_for_sightings(dataset, test_indices, args.max_sightings, config["seed"])
        checkpoint = base.resolve_checkpoint_dir(config, download=False)
        projector, concept_backbone, backbone, head = base.build_models(config, checkpoint, dataset)
        caches = []
        for name, indices in [("gallery", train_indices), ("query", test_indices)]:
            print(f"Collecting {name} features for {len(indices)} images", flush=True)
            loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=True)
            caches.append(base.collect_cache(loader, dataset, concept_backbone, backbone, head, projector.device))
        train, query = caches
        query = query.subset(base.select_query_positions(query, args.max_sightings, config["seed"]))
        gallery_fn = base.SEEK.get_correction_policy(config["gallery_correction_fn"], dataset, default_scope="sighting")
        gallery = base.embeddings_from_cache(projector, train, gallery_fn).to(projector.device)
        retrieval = base.Retrieval(experiment_code=config["experiment_name"])
        pooler = IdentityScorePooler.from_labels(train.labels, projector.device)
        groups = base.group_positions(query.encounter_ids)
        seeks = base.SEEK.closest_valid_one_hot(query.concept_logits.to(projector.device)).cpu()
        masks = {s: torch.zeros(16, dtype=torch.bool) for s in groups}
        policy = RawMarginPolicy(beta)
        rows, hits, decisions = [], [], []
        reference_meta = np.load(source / "query_metadata.npz")
        positions = {int(v): i for i, v in enumerate(reference_meta["indices"])}
        matched = np.array([positions[int(idx)] for idx in query.indices])
        np.testing.assert_array_equal(query.labels.numpy(), reference_meta["labels"][matched])
        reference_meta.close()
        reference_hits = np.load(source / f"trace_margin_{config['seed']}.npz")["hits"][matched]
        for budget in range(17):
            if budget:
                for sighting, pos in groups.items():
                    state = PolicyState(seeks[pos], query.concept_logits[pos], masks[sighting])
                    decision = policy.select(state, lambda variants: score_variants(
                        projector, retrieval, query.visual_embeddings[pos], variants, gallery, pooler, tau, config["epsilon"], audit))
                    legacy_scores = {}
                    diagnostics = decision.diagnostics
                    for name, a in diagnostics["attributes"].items():
                        importance = sum(p*m["legacy_margin"] for p, m in zip(a["probabilities"], a["values"])) - diagnostics["current"]["legacy_margin"]
                        legacy_scores[name] = a["concept_uncertainty"] + old_lambda * importance
                        audit["max_policy_score_error"] = max(audit["max_policy_score_error"], abs(legacy_scores[name]-a["combined_score"]))
                    legacy_choice = max(legacy_scores, key=lambda a: (legacy_scores[a], -base.SEEK.attribute_names.index(a)))
                    audit["raw_vs_live_legacy_choice_differences"] += decision.attribute_name != legacy_choice
                    # Reveal the expert value only after the raw-policy decision.
                    seeks[pos] = copy_attribute_from_expert(seeks[pos], query.sighting_seeks[pos], decision.attribute_index)
                    masks[sighting][decision.attribute_index] = True
                    decisions.append({"sighting": sighting, "budget": budget, "attribute_name": decision.attribute_name,
                                      "margin_beta": beta, "combined_score": decision.score,
                                      "concept_uncertainty": diagnostics["attributes"][decision.attribute_name]["concept_uncertainty"],
                                      "raw_importance": diagnostics["attributes"][decision.attribute_name]["raw_importance"],
                                      "live_legacy_choice": legacy_choice})
            metrics = base.evaluate_budget(projector, retrieval, query, seeks, gallery, train.labels, tau, config["epsilon"])
            budget_hits = metrics.pop("_hits")
            hits.append(budget_hits)
            rows.append({"policy_key": "margin", "policy": "Margin-ratio", "seed": config["seed"],
                         "budget": budget, "margin_beta": beta, **metrics})
            delta = int(np.count_nonzero(budget_hits != reference_hits[:, budget]))
            print(f"Budget {budget:2d}/16: Recall@1={metrics['recall_at_1']:.6f}%; changed image hits={delta}; elapsed={time.time()-started:.0f}s", flush=True)
            wandb.log({**rows[-1], "changed_image_hits": delta})
            pd.DataFrame(rows).to_csv(output / "raw_margin_results.csv", index=False)
        assert torch.equal(seeks, query.sighting_seeks)
        actual_hits = np.stack(hits, axis=1)
        audit["changed_hits_by_budget"] = np.count_nonzero(actual_hits != reference_hits, axis=0).tolist()
        audit["all_image_hits_identical"] = bool(np.array_equal(actual_hits, reference_hits))
        audit.update(n_query_images=len(query.labels), n_sightings=len(groups), n_decisions=len(decisions))
        print("Comparing full-run recorded intervention sequences", flush=True)
        with (source / "per_sighting_interventions.json").open() as handle:
            original_decisions = json.load(handle)[f"margin:{config['seed']}"]
        for record in decisions:
            record["original_choice"] = original_decisions[record["sighting"]][record["budget"]-1]["attribute_name"]
        audit["raw_vs_original_choice_differences"] = sum(r["attribute_name"] != r["original_choice"] for r in decisions)
        pd.DataFrame(decisions).to_csv(output / "margin_decisions.csv", index=False)
        np.savez_compressed(output / f"trace_margin_{config['seed']}.npz", hits=actual_hits)
        np.savez_compressed(output / "query_metadata.npz", labels=query.labels.numpy(), indices=query.indices.numpy(), encounter_ids=np.array(query.encounter_ids))
        # Reuse unchanged Random/Oracle traces on exactly the same query image subset.
        baseline_rows = []
        for path in source.glob("trace_*.npz"):
            _, key, seed_str = path.stem.split("_", 2)
            if key not in {"random", "oracle"}:
                continue
            with np.load(path) as trace:
                baseline = trace["hits"][matched]
            np.savez_compressed(output / path.name, hits=baseline)
            for b in range(17):
                baseline_rows.append({"policy_key": key, "policy": base.POLICY_DISPLAY[key], "seed": int(seed_str),
                                      "budget": b, "recall_at_1": 100*baseline[:, b].mean()})
        final_rows = baseline_rows + rows
        pd.DataFrame(final_rows).to_csv(output / "prioritization_results.csv", index=False)
        result = {"config": config, "rows": final_rows, "rescaling_audit": audit,
                  "source_results": str(source), "original_validation": saved["margin_lambda"]}
        (output / "prioritization_results.json").write_text(json.dumps(result, indent=2) + "\n")
        plot_results(output / "prioritization_results.csv", output / "intervention_prioritization.pdf", output / "intervention_prioritization.png")
        original_ci = pd.read_csv(source / "confidence_intervals.csv")
        new_ci = pd.read_csv(output / "confidence_intervals.csv")
        if args.max_sightings is None:
            before = original_ci[original_ci.policy_key.isin(["margin", "random", "oracle"])].sort_values(["policy_key", "budget"])
            after = new_ci.sort_values(["policy_key", "budget"])
            audit["max_ci_difference_pp"] = float(np.max(np.abs(before[["ci95_low", "ci95_high"]].to_numpy() - after[["ci95_low", "ci95_high"]].to_numpy())))
            target = ROOT / "figures/priorization"
            for name in ["intervention_prioritization.pdf", "intervention_prioritization.png", "confidence_intervals.csv"]:
                shutil.copy2(output / name, target / name)
        audit["elapsed_seconds"] = time.time()-started
        audit["wandb_url"] = wandb.run.url
        if args.max_sightings is None:
            (target / "rescaling_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (output / "rescaling_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        result["rescaling_audit"] = audit
        (output / "prioritization_results.json").write_text(json.dumps(result, indent=2) + "\n")
        wandb.summary.update(audit)
        wandb.log({"rescaled_margin_plot": wandb.Image(str(output / "intervention_prioritization.png"))})
        print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
