#!/usr/bin/env python3
"""Run prioritized per-concept interventions for SEEK-CBM retrieval."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import wandb


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "methods"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from backbone import Backbone  # noqa: E402
from concept_head_tunneled import ConceptHeadTunneled, ThreeHeadNN, categorical_CE_loss  # noqa: E402
from data_handler import EleHandler  # noqa: E402
from projector import Projector  # noqa: E402
from retrieval import Retrieval  # noqa: E402
from seek_code import SEEK  # noqa: E402

from experiments.seek_intervention_prioritization.policies import (  # noqa: E402
    ExpectedEntropyReductionPolicy,
    GreedyOraclePolicy,
    MarginRatioPolicy,
    PolicyDecision,
    PolicyState,
    RandomPolicy,
    copy_attribute_from_expert,
)
from experiments.seek_intervention_prioritization.retrieval_uncertainty import (  # noqa: E402
    IdentityScorePooler,
    fit_temperature_grid,
    retrieval_metrics,
    summarize_uncertainty,
    true_identity_ranking_metrics,
)


from experiments.seek_intervention_top20.topk_entropy import initial_candidates, fixed_candidate_entropy


POLICY_DISPLAY = {
    "random": "Random",
    "entropy": "Entropy: fixed initial top-20",
    "entropy_full": "Entropy: all identities",
    "margin": "Margin-ratio",
    "oracle": "Oracle",
}


@dataclass
class SplitCache:
    visual_embeddings: torch.Tensor
    concept_logits: torch.Tensor
    labels: torch.Tensor
    subject_seeks: torch.Tensor
    sighting_seeks: torch.Tensor
    indices: torch.Tensor
    encounter_ids: list[str]
    ele_ids: list[str]
    subject_ids: list[str]

    def subset(self, positions: list[int]) -> "SplitCache":
        pos = torch.as_tensor(positions, dtype=torch.long)
        return SplitCache(
            visual_embeddings=self.visual_embeddings[pos],
            concept_logits=self.concept_logits[pos],
            labels=self.labels[pos],
            subject_seeks=self.subject_seeks[pos],
            sighting_seeks=self.sighting_seeks[pos],
            indices=self.indices[pos],
            encounter_ids=[self.encounter_ids[i] for i in positions],
            ele_ids=[self.ele_ids[i] for i in positions],
            subject_ids=[self.subject_ids[i] for i in positions],
        )


def repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_indices(train_path: Path, test_path: Path) -> tuple[list[int], list[int]]:
    with train_path.open() as f:
        train_indices = [int(line.strip()) for line in f if line.strip()]
    with test_path.open() as f:
        test_indices = [int(line.strip()) for line in f if line.strip()]
    return train_indices, test_indices


def resolve_checkpoint_dir(config: dict[str, Any], *, download: bool) -> Path:
    ref = config["heatmap_reference"]
    required = ref["checkpoint_files"]
    candidates = [
        repo_path(ref["local_checkpoint_dir"]),
        repo_path(ref["download_checkpoint_dir"]),
    ]
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in required):
            return candidate

    if not download:
        missing = ", ".join(required)
        raise FileNotFoundError(
            "Could not find the target heatmap checkpoint locally. "
            f"Expected {missing} under one of: {candidates}. "
            "Rerun with --download-checkpoint to fetch the W&B run files."
        )

    target_dir = repo_path(ref["download_checkpoint_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    run = api.run(f"{config['wandb']['project_path']}/{ref['wandb_run_id']}")
    for filename in required + ["config.yaml", "wandb-summary.json"]:
        run.file(filename).download(root=str(target_dir), replace=False)
    return target_dir


def build_models(config: dict[str, Any], checkpoint_dir: Path, dataset: EleHandler):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The full prioritization runner requires CUDA because the existing "
            "Projector/ThreeHeadNN code places layers on 'cuda'."
        )

    cfg = config["heatmap_reference"]["config"]
    num_classes = len(dataset.ele_id_to_label)
    backbone_for_concepts = Backbone(
        model_name="MegaDescriptor",
        pretraining=cfg["backbone_for_concepts_pretraining"],
        experiment_code=config["heatmap_reference"]["wandb_run_name"],
        num_classes=num_classes,
    )
    backbone = Backbone(
        model_name="MegaDescriptor",
        pretraining=cfg["backbone_pretraining"],
        experiment_code=config["heatmap_reference"]["wandb_run_name"],
        num_classes=num_classes,
    )
    concept_head = ConceptHeadTunneled(
        loss=categorical_CE_loss,
        wd=cfg.get("concept_head_wd"),
        experiment_code=config["heatmap_reference"]["wandb_run_name"],
        layer=ThreeHeadNN(),
        reset_weights=False,
        pretraining=cfg["concept_head_pretraining"],
    )
    projector = Projector(
        loss_type="ArcFace",
        lr=cfg["projector_lr"],
        scale=64,
        margin=0.5,
        experiment_code=config["heatmap_reference"]["wandb_run_name"],
        intervention_fn=None,
        alpha=cfg["alpha"],
        reset_weights=True,
        network_type=cfg["projector_architecture"],
        num_classes=num_classes,
        wd=cfg["projector_wd"],
        layer_norm=cfg["layer_norm"],
        alpha_learnable=cfg["alpha_learnable"],
    )

    projector.layer.load_state_dict(torch.load(checkpoint_dir / "projector.pt", map_location=projector.device))
    backbone.layer.load_state_dict(torch.load(checkpoint_dir / "backbone_w.pt", map_location=backbone.device))
    alpha_path = checkpoint_dir / "alpha.pt"
    if alpha_path.exists():
        alpha_data = torch.load(alpha_path, map_location=projector.device)
        alpha = float(alpha_data["alpha"] if isinstance(alpha_data, dict) else alpha_data)
        if projector.alpha_learnable:
            projector.alpha.data.fill_(alpha)
        else:
            projector.alpha = alpha

    backbone_for_concepts.freeze()
    backbone.freeze()
    concept_head.freeze()
    projector.freeze()
    return projector, backbone_for_concepts, backbone, concept_head


def collect_cache(
    loader: DataLoader,
    dataset: EleHandler,
    backbone_for_concepts: Backbone,
    backbone: Backbone,
    concept_head: ConceptHeadTunneled,
    device: str,
) -> SplitCache:
    visual_embeddings = []
    concept_logits = []
    labels = []
    subject_seeks = []
    sighting_seeks = []
    indices = []
    encounter_ids: list[str] = []
    ele_ids: list[str] = []
    subject_ids: list[str] = []

    backbone.layer.eval()
    backbone_for_concepts.layer.eval()
    concept_head.layer.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            ele_id_label = batch[2].to(device)
            subject_seek = batch[4].to(device)
            sighting_seek = batch[5].to(device)
            left_ears = batch[6].to(device)
            right_ears = batch[7].to(device)
            idx = batch[10]

            visual = backbone.forward(images, left_ears, right_ears)
            concept_input = backbone_for_concepts.forward(images, left_ears, right_ears)
            logits = concept_head.layer(concept_input)

            idx_cpu = idx.cpu() if torch.is_tensor(idx) else torch.as_tensor(idx)
            for raw_idx in idx_cpu.tolist():
                row = dataset.dictonary.iloc[int(raw_idx)]
                encounter_ids.append(str(row["encounter_id"]))
                ele_ids.append(str(row["ele_id"]))
                subject_ids.append(str(row["subject_id"]))

            visual_embeddings.append(visual.detach().cpu())
            concept_logits.append(logits.detach().cpu())
            labels.append(ele_id_label.detach().cpu())
            subject_seeks.append(subject_seek.detach().cpu())
            sighting_seeks.append(sighting_seek.detach().cpu())
            indices.append(idx_cpu.detach().cpu())

    return SplitCache(
        visual_embeddings=torch.cat(visual_embeddings),
        concept_logits=torch.cat(concept_logits),
        labels=torch.cat(labels).long(),
        subject_seeks=torch.cat(subject_seeks),
        sighting_seeks=torch.cat(sighting_seeks),
        indices=torch.cat(indices).long(),
        encounter_ids=encounter_ids,
        ele_ids=ele_ids,
        subject_ids=subject_ids,
    )


def alpha_value(projector: Projector):
    return projector.alpha if torch.is_tensor(projector.alpha) else float(projector.alpha)


def project_from_seek(
    projector: Projector,
    visual_embeddings: torch.Tensor,
    seek_vectors: torch.Tensor,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    output = []
    device = projector.device
    projector.layer.eval()
    with torch.no_grad():
        for start in range(0, seek_vectors.shape[0], batch_size):
            end = start + batch_size
            visual = visual_embeddings[start:end].to(device)
            seek = seek_vectors[start:end].to(device)
            projected = projector.layer(seek)
            if projector.layer_norm:
                visual = F.layer_norm(visual, visual.size()[1:])
                projected = F.layer_norm(projected, projected.size()[1:])
            output.append(((1 - alpha_value(projector)) * visual + alpha_value(projector) * projected).detach().cpu())
    return torch.cat(output)


def embeddings_from_cache(
    projector: Projector,
    cache: SplitCache,
    intervention_fn,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    output = []
    device = projector.device
    projector.layer.eval()
    with torch.no_grad():
        for start in range(0, cache.labels.numel(), batch_size):
            end = start + batch_size
            visual = cache.visual_embeddings[start:end].to(device)
            logits = cache.concept_logits[start:end].to(device)
            subject_seek = cache.subject_seeks[start:end].to(device)
            sighting_seek = cache.sighting_seeks[start:end].to(device)
            labels = cache.labels[start:end].to(device)
            idx = cache.indices[start:end]
            edited = intervention_fn(logits, subject_seek, sighting_seek, ele_id=labels, idx=idx)
            projected = projector.layer(edited.to(device))
            if projector.layer_norm:
                visual = F.layer_norm(visual, visual.size()[1:])
                projected = F.layer_norm(projected, projected.size()[1:])
            output.append(((1 - alpha_value(projector)) * visual + alpha_value(projector) * projected).detach().cpu())
    return torch.cat(output)


def select_query_positions(cache: SplitCache, max_sightings: int | None, seed: int) -> list[int]:
    by_sighting = group_positions(cache.encounter_ids)
    sightings = list(by_sighting)
    if max_sightings is not None:
        if max_sightings <= 0:
            raise ValueError("--max-sightings must be positive")
        rng = random.Random(seed)
        rng.shuffle(sightings)
        sightings = sorted(sightings[:max_sightings])
    selected: list[int] = []
    for sighting in sightings:
        selected.extend(by_sighting[sighting])
    return sorted(selected)


def select_indices_for_sightings(
    dataset: EleHandler,
    indices: list[int],
    max_sightings: int | None,
    seed: int,
) -> list[int]:
    if max_sightings is None:
        return indices
    if max_sightings <= 0:
        raise ValueError("--max-sightings must be positive")
    sightings: dict[str, list[int]] = {}
    for idx in indices:
        encounter_id = str(dataset.dictonary.iloc[int(idx)]["encounter_id"])
        sightings.setdefault(encounter_id, []).append(int(idx))
    keys = list(sightings)
    rng = random.Random(seed)
    rng.shuffle(keys)
    selected_keys = set(keys[:max_sightings])
    return [idx for idx in indices if str(dataset.dictonary.iloc[int(idx)]["encounter_id"]) in selected_keys]


def select_gallery_indices(
    dataset: EleHandler,
    indices: list[int],
    max_gallery_images: int | None,
    seed: int,
    required_ele_ids: set[str] | None = None,
) -> list[int]:
    if max_gallery_images is None:
        return indices
    if max_gallery_images <= 0:
        raise ValueError("--max-gallery-images must be positive")
    if max_gallery_images >= len(indices):
        return indices
    required_ele_ids = required_ele_ids or set()
    required = [
        idx
        for idx in indices
        if str(dataset.dictonary.iloc[int(idx)]["ele_id"]) in required_ele_ids
    ]
    required = sorted(set(required))
    if len(required) >= max_gallery_images:
        return required
    rng = random.Random(seed)
    remaining = [idx for idx in indices if idx not in set(required)]
    fill = rng.sample(remaining, max_gallery_images - len(required))
    return sorted(required + fill)


def group_positions(encounter_ids: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for pos, encounter_id in enumerate(encounter_ids):
        groups.setdefault(str(encounter_id), []).append(pos)
    return groups


def scorer_for_sighting(
    projector: Projector,
    retrieval: Retrieval,
    cache: SplitCache,
    positions: list[int],
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    epsilon: float,
    *,
    include_truth: bool,
    candidates: torch.Tensor | None = None,
):
    visual = cache.visual_embeddings[positions]
    labels = cache.labels[positions] if include_truth else None
    gallery_embeddings_device = gallery_embeddings.to(projector.device)
    gallery_labels_device = gallery_labels.to(projector.device)
    pooler = IdentityScorePooler.from_labels(gallery_labels_device, projector.device)

    def score(seek_vectors: torch.Tensor) -> dict[str, float]:
        query_embeddings = project_from_seek(projector, visual, seek_vectors)
        similarity = retrieval.similarity_matrix(
            query_embeddings.to(projector.device),
            gallery_embeddings_device,
        )
        metrics = summarize_uncertainty(
            similarity,
            gallery_labels_device,
            temperature,
            epsilon,
            pooler=pooler,
        )
        if candidates is not None:
            _, identity_scores = pooler.score_matrix(similarity)
            metrics["entropy"] = float(fixed_candidate_entropy(identity_scores, candidates, temperature).mean().item())
        if include_truth:
            metrics.update(
                true_identity_ranking_metrics(
                    similarity,
                    labels.to(projector.device),
                    gallery_labels_device,
                    pooler=pooler,
                )
            )
        return metrics

    return score


def evaluate_budget(
    projector: Projector,
    retrieval: Retrieval,
    cache: SplitCache,
    current_seek: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    epsilon: float,
    *,
    initial_top_k: int = 0,
) -> dict[str, float]:
    query_embeddings = project_from_seek(projector, cache.visual_embeddings, current_seek)
    gallery_embeddings_device = gallery_embeddings.to(projector.device)
    gallery_labels_device = gallery_labels.to(projector.device)
    similarity = retrieval.similarity_matrix(
        query_embeddings.to(projector.device),
        gallery_embeddings_device,
    )
    metrics = retrieval_metrics(
        similarity,
        cache.labels.to(projector.device),
        gallery_labels_device,
        ks=(1, 5),
    )
    pooler = IdentityScorePooler.from_labels(gallery_labels_device, projector.device)
    uncertainty = summarize_uncertainty(
        similarity,
        gallery_labels_device,
        temperature,
        epsilon,
        pooler=pooler,
    )
    extra = {}
    if initial_top_k:
        identities, scores = pooler.score_matrix(similarity)
        extra["_initial_candidates"] = initial_candidates(scores, initial_top_k).cpu()
        extra["_gallery_identities"] = identities.cpu()
    top_entries = similarity.argmax(dim=1)
    return {
        **extra,
        "_hits": (gallery_labels_device[top_entries] == cache.labels.to(projector.device)).cpu().numpy(),
        "recall_at_1": metrics["recall_at_1"] * 100.0,
        "recall_at_5": metrics["recall_at_5"] * 100.0,
        "mrr": metrics["mrr"],
        "retrieval_entropy": uncertainty["entropy"],
        "top1_top2_log_margin": uncertainty["margin_log_ratio"],
        "uncertainty_temperature": uncertainty["temperature"],
    }


def calibrate_temperature(
    projector: Projector,
    retrieval: Retrieval,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | str]]:
    if args.temperature is not None:
        print(f"Using provided retrieval temperature: {args.temperature}", flush=True)
        return float(args.temperature), {"mode": "provided", "temperature": float(args.temperature)}

    print("Fitting retrieval temperature on train-gallery leave-one-out identity NLL", flush=True)
    positions = list(range(gallery_labels.numel()))
    if args.max_calibration_images is not None and args.max_calibration_images < len(positions):
        rng = random.Random(config["seed"])
        positions = sorted(rng.sample(positions, args.max_calibration_images))

    query_embeddings = gallery_embeddings[positions]
    query_labels = gallery_labels[positions]
    gallery_embeddings_device = gallery_embeddings.to(projector.device)
    gallery_labels_device = gallery_labels.to(projector.device)
    pooler = IdentityScorePooler.from_labels(gallery_labels_device, projector.device)
    similarity = retrieval.similarity_matrix(
        query_embeddings.to(projector.device),
        gallery_embeddings_device,
    )
    row = torch.arange(len(positions), device=projector.device)
    similarity[row, torch.as_tensor(positions, device=projector.device)] = -torch.inf
    grid = config["temperature_grid"]
    tau, nll = fit_temperature_grid(
        similarity,
        query_labels.to(projector.device),
        gallery_labels_device,
        min_tau=float(grid["min"]),
        max_tau=float(grid["max"]),
        steps=int(grid["steps"]),
        pooler=pooler,
    )
    print(f"Selected retrieval temperature: {tau:.6g} (identity NLL={nll})", flush=True)
    return tau, {
        "mode": "train_gallery_leave_one_out",
        "temperature": tau,
        "identity_nll": nll,
        "calibration_images": len(positions),
    }


def make_policy(
    policy_key: str,
    seed: int,
    config: dict[str, Any],
    *,
    margin_lambda: float | None = None,
):
    if policy_key == "random":
        return RandomPolicy(seed=seed)
    if policy_key in {"entropy", "entropy_full"}:
        return ExpectedEntropyReductionPolicy()
    if policy_key == "margin":
        if margin_lambda is None:
            margin_lambda = float(config.get("selected_margin_lambda", 1.0))
        return MarginRatioPolicy(margin_lambda=margin_lambda, eps=float(config["epsilon"]))
    if policy_key == "oracle":
        return GreedyOraclePolicy()
    raise ValueError(f"Unknown policy: {policy_key}")


def margin_validation_split(
    cache: SplitCache,
    config: dict[str, Any],
    max_validation_sightings: int | None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    validation_cfg = config.get("margin_validation", {})
    fraction = float(validation_cfg.get("fraction", 0.2))
    if not 0.0 < fraction < 1.0:
        raise ValueError("margin_validation.fraction must be between 0 and 1")

    positions_by_sighting = group_positions(cache.encounter_ids)
    sighting_keys = list(positions_by_sighting)
    rng = random.Random(int(config["seed"]))
    rng.shuffle(sighting_keys)

    target_sightings = max(1, round(len(sighting_keys) * fraction))
    if max_validation_sightings is not None:
        if max_validation_sightings <= 0:
            raise ValueError("--max-validation-sightings must be positive")
        target_sightings = min(target_sightings, max_validation_sightings)

    remaining_label_counts = Counter(int(label) for label in cache.labels.tolist())
    validation_positions: list[int] = []
    selected_sightings: list[str] = []
    for sighting in sighting_keys:
        positions = positions_by_sighting[sighting]
        sighting_counts = Counter(int(cache.labels[pos].item()) for pos in positions)
        if any(remaining_label_counts[label] - count <= 0 for label, count in sighting_counts.items()):
            continue
        validation_positions.extend(positions)
        selected_sightings.append(sighting)
        remaining_label_counts.subtract(sighting_counts)
        if len(selected_sightings) >= target_sightings:
            break

    if not validation_positions:
        raise RuntimeError(
            "Could not construct a non-leaky margin validation split from the train gallery. "
            "Use a larger gallery subset or provide more train images."
        )

    validation_set = set(validation_positions)
    gallery_positions = [pos for pos in range(cache.labels.numel()) if pos not in validation_set]
    info = {
        "validation_fraction": fraction,
        "target_validation_sightings": target_sightings,
        "validation_sightings": len(selected_sightings),
        "validation_images": len(validation_positions),
        "validation_gallery_images": len(gallery_positions),
    }
    return sorted(validation_positions), gallery_positions, info


def tune_margin_lambda(
    projector: Projector,
    retrieval: Retrieval,
    train_cache: SplitCache,
    gallery_fn,
    temperature: float,
    config: dict[str, Any],
    args: argparse.Namespace,
    max_budget: int,
) -> dict[str, Any]:
    if args.margin_lambda is not None:
        print(f"Using provided margin lambda: {args.margin_lambda}", flush=True)
        return {
            "mode": "provided",
            "margin_lambda": float(args.margin_lambda),
            "early_budget_k": int(config.get("margin_validation", {}).get("early_budget_k", 4)),
        }

    validation_cfg = config.get("margin_validation", {})
    early_budget_k = min(int(validation_cfg.get("early_budget_k", 4)), max_budget, len(SEEK.attribute_names))
    if early_budget_k <= 0:
        raise ValueError("Margin lambda tuning requires at least one non-zero validation budget")

    validation_positions, gallery_positions, split_info = margin_validation_split(
        train_cache,
        config,
        args.max_validation_sightings,
    )
    validation_cache = train_cache.subset(validation_positions)
    validation_gallery_cache = train_cache.subset(gallery_positions)
    validation_gallery_embeddings = embeddings_from_cache(projector, validation_gallery_cache, gallery_fn)
    validation_gallery_labels = validation_gallery_cache.labels

    candidates: list[dict[str, Any]] = []
    print(
        f"Tuning margin lambda on {validation_cache.labels.numel()} validation images "
        f"against {validation_gallery_labels.numel()} gallery images",
        flush=True,
    )
    for grid_index, candidate_lambda in enumerate(config["margin_lambda_grid"]):
        candidate_lambda = float(candidate_lambda)
        rows, _ = run_single_policy(
            "margin",
            int(config["seed"]),
            projector,
            retrieval,
            validation_cache,
            validation_gallery_embeddings,
            validation_gallery_labels,
            temperature,
            config,
            early_budget_k,
            margin_lambda=candidate_lambda,
            log_to_wandb=False,
        )
        frame = pd.DataFrame(rows)
        recall_by_budget = {
            int(budget): float(value)
            for budget, value in frame[
                (frame["budget"] >= 1) & (frame["budget"] <= early_budget_k)
            ].groupby("budget")["recall_at_1"].mean().items()
        }
        objective_values = [
            recall_by_budget[budget]
            for budget in range(1, early_budget_k + 1)
            if budget in recall_by_budget
        ]
        if not objective_values:
            raise RuntimeError("No validation rows were produced for margin lambda tuning")
        objective = float(sum(objective_values) / len(objective_values))
        candidate = {
            "grid_index": grid_index,
            "margin_lambda": candidate_lambda,
            "validation_objective": objective,
            "validation_recall_at_1_by_budget": recall_by_budget,
        }
        candidates.append(candidate)
        print(
            f"  margin_lambda={candidate_lambda:g}: "
            f"mean R@1 budgets 1..{early_budget_k} = {objective:.3f}",
            flush=True,
        )
        wandb.log(
            {
                "margin_validation/lambda": candidate_lambda,
                "margin_validation/objective": objective,
                **{
                    f"margin_validation/recall_at_1_budget_{budget}": recall
                    for budget, recall in recall_by_budget.items()
                },
            }
        )

    best = max(candidates, key=lambda item: (item["validation_objective"], -item["grid_index"]))
    print(
        f"Selected margin lambda: {best['margin_lambda']:g} "
        f"(validation objective={best['validation_objective']:.3f})",
        flush=True,
    )
    return {
        "mode": "validation_grid",
        "margin_lambda": float(best["margin_lambda"]),
        "early_budget_k": early_budget_k,
        "selection_objective": "mean Recall@1 over budgets 1..K",
        "best_validation_objective": float(best["validation_objective"]),
        "candidates": candidates,
        **split_info,
    }


def choose_attribute(
    policy_key: str,
    policy,
    state: PolicyState,
    expert_seek: torch.Tensor,
    scorer,
) -> PolicyDecision:
    if policy_key == "random":
        return policy.select(state)
    if policy_key in {"entropy", "entropy_full", "margin"}:
        current_metrics = scorer(state.current_seek)
        return policy.select(
            state,
            scorer,
            current_metrics=current_metrics,
        )
    if policy_key == "oracle":
        return policy.select(state, expert_seek, scorer)
    raise ValueError(f"Unknown policy: {policy_key}")


def run_single_policy(
    policy_key: str,
    seed: int,
    projector: Projector,
    retrieval: Retrieval,
    query_cache: SplitCache,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    temperature: float,
    config: dict[str, Any],
    max_budget: int,
    *,
    margin_lambda: float | None = None,
    log_to_wandb: bool = True,
    evaluation_trace: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = make_policy(policy_key, seed, config, margin_lambda=margin_lambda)
    epsilon = float(config["epsilon"])
    current_seek = SEEK.closest_valid_one_hot(query_cache.concept_logits.to(projector.device)).detach().cpu()
    corrected_masks = {
        sighting: torch.zeros(len(SEEK.attribute_names), dtype=torch.bool)
        for sighting in group_positions(query_cache.encounter_ids)
    }
    positions_by_sighting = group_positions(query_cache.encounter_ids)
    interventions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    frozen_candidates = None

    def record_budget(budget: int) -> None:
        nonlocal frozen_candidates
        metrics = evaluate_budget(
            projector,
            retrieval,
            query_cache,
            current_seek,
            gallery_embeddings,
            gallery_labels,
            temperature,
            epsilon,
            initial_top_k=int(config["entropy_initial_top_k"]) if policy_key == "entropy" and budget == 0 else 0,
        )
        hits = metrics.pop("_hits")
        if evaluation_trace is not None:
            evaluation_trace.setdefault("hits", []).append(hits)
        if "_initial_candidates" in metrics:
            frozen_candidates = metrics.pop("_initial_candidates")
            identities = metrics.pop("_gallery_identities")
            if evaluation_trace is not None:
                evaluation_trace["initial_candidate_identities"] = identities[frozen_candidates].numpy()
                evaluation_trace["initial_top20_true_identity_coverage"] = float(
                    (identities[frozen_candidates] == query_cache.labels[:, None]).any(dim=1).float().mean().item()
                )
        row = {
            "policy": POLICY_DISPLAY[policy_key],
            "policy_key": policy_key,
            "seed": seed,
            "budget": budget,
            "gallery_correction": 20,
            "gallery_correction_fn": config["gallery_correction_fn"],
            "checkpoint": config["heatmap_reference"]["wandb_run_name"],
            "temperature": temperature,
            **metrics,
        }
        if policy_key == "margin":
            row["margin_lambda"] = policy.margin_lambda
        rows.append(row)
        if log_to_wandb:
            wandb.log(row)

    record_budget(0)
    print(f"[{policy_key}:{seed}] recorded budget 0/{max_budget}", flush=True)
    for budget in range(1, max_budget + 1):
        for sighting, positions in positions_by_sighting.items():
            mask = corrected_masks[sighting]
            if bool(mask.all().item()):
                continue
            pos_tensor = torch.as_tensor(positions, dtype=torch.long)
            state = PolicyState(
                current_seek=current_seek[pos_tensor],
                concept_logits=query_cache.concept_logits[pos_tensor],
                corrected_mask=mask,
            )
            expert_seek = query_cache.sighting_seeks[pos_tensor]
            scorer = scorer_for_sighting(
                projector,
                retrieval,
                query_cache,
                positions,
                gallery_embeddings,
                gallery_labels,
                temperature,
                epsilon,
                include_truth=policy_key == "oracle",
                candidates=frozen_candidates[pos_tensor] if frozen_candidates is not None else None,
            )
            logging_scorer = scorer_for_sighting(
                projector,
                retrieval,
                query_cache,
                positions,
                gallery_embeddings,
                gallery_labels,
                temperature,
                epsilon,
                include_truth=True,
                candidates=frozen_candidates[pos_tensor] if frozen_candidates is not None else None,
            )
            decision = choose_attribute(policy_key, policy, state, expert_seek, scorer)
            current_seek[pos_tensor] = copy_attribute_from_expert(
                current_seek[pos_tensor],
                expert_seek,
                decision.attribute_index,
            )
            mask[decision.attribute_index] = True
            current_metrics = logging_scorer(current_seek[pos_tensor])
            intervention_record = {
                "budget": budget,
                "policy": POLICY_DISPLAY[policy_key],
                "policy_key": policy_key,
                "seed": seed,
                "attribute_index": decision.attribute_index,
                "attribute_name": decision.attribute_name,
                "policy_score": decision.score,
                "scores_by_attribute": decision.scores_by_attribute,
                "policy_diagnostics": decision.diagnostics,
                "retrieval_entropy": current_metrics["entropy"],
                "top1_top2_log_margin": current_metrics["margin_log_ratio"],
                "uncertainty_temperature": current_metrics["temperature"],
                "true_recall_at_1": current_metrics["true_recall_at_1"],
                "true_mrr": current_metrics["true_mrr"],
                "true_id_margin": current_metrics["true_id_margin"],
            }
            if policy_key == "margin":
                selected_diagnostics = decision.diagnostics.get("selected", {})
                intervention_record.update(
                    {
                        "margin_lambda": decision.diagnostics.get("margin_lambda"),
                        "margin_concept_uncertainty": selected_diagnostics.get("margin_concept_uncertainty"),
                        "margin_importance": selected_diagnostics.get("margin_importance"),
                        "margin_combined_score": selected_diagnostics.get("margin_combined_score"),
                    }
                )
                if log_to_wandb:
                    wandb.log(
                        {
                            "margin_lambda": intervention_record["margin_lambda"],
                            "margin_concept_uncertainty": intervention_record["margin_concept_uncertainty"],
                            "margin_importance": intervention_record["margin_importance"],
                            "margin_combined_score": intervention_record["margin_combined_score"],
                            "intervention/budget": budget,
                            "intervention/seed": seed,
                        }
                    )
            interventions[sighting].append(intervention_record)
        record_budget(budget)
        print(f"[{policy_key}:{seed}] recorded budget {budget}/{max_budget}", flush=True)
    if max_budget >= len(SEEK.attribute_names):
        if not all(bool(mask.all().item()) for mask in corrected_masks.values()):
            raise AssertionError("Budget-16 invariant failed: at least one sighting has uncorrected attributes")
        if not torch.equal(current_seek, query_cache.sighting_seeks):
            raise AssertionError("Budget-16 invariant failed: query SEEK vectors are not fully sighting-corrected")
    return rows, dict(interventions)


def endpoint_checks(rows: list[dict[str, Any]], max_budget: int, tolerance: float) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    frame = pd.DataFrame(rows)
    for budget in [0, max_budget]:
        if budget == 16 or budget == 0:
            values = frame[frame["budget"] == budget].groupby("policy_key")["recall_at_1"].mean()
            spread = float(values.max() - values.min()) if not values.empty else float("nan")
            checks[f"budget_{budget}_recall_at_1_spread"] = spread
            if spread > tolerance:
                raise AssertionError(
                    f"Recall@1 endpoint invariant failed at budget {budget}: spread={spread}"
                )
    if max_budget < 16:
        checks["budget_16_check"] = "skipped because max_budget < 16"
    return checks


def reference_checks(rows: list[dict[str, Any]], config: dict[str, Any], strict: bool) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    checks: dict[str, Any] = {
        "note": (
            "The local CHAIR heatmap export is mostly diagonal in gallery/query "
            "correction. Exact endpoint comparison is only possible when the fixed "
            "gallery/query endpoint matches an exported metric."
        )
    }
    if frame.empty:
        return checks
    refs = config["heatmap_reference"]["reference_metrics_pct"]
    budget0 = float(frame[frame["budget"] == 0]["recall_at_1"].mean())
    checks["budget_0_recall_at_1"] = budget0
    if config["gallery_correction_fn"] in {"image_0.0", "sighting_0.0"}:
        reference = refs["Pr/Test Recall@1 with 0% Image Correction correction"]
        checks["budget_0_reference_metric"] = "Pr/Test Recall@1 with 0% Image Correction correction"
        checks["budget_0_reference_recall_at_1"] = reference
        checks["budget_0_reference_delta"] = budget0 - reference
        if strict and abs(budget0 - reference) > 1e-5:
            raise AssertionError(f"Budget-0 reference check failed: {budget0} vs {reference}")
    else:
        checks["budget_0_reference_metric"] = "no exact exported fixed-gallery G_S(20), query=0 cell"

    budget16_rows = frame[frame["budget"] == 16]
    if not budget16_rows.empty:
        budget16 = float(budget16_rows["recall_at_1"].mean())
        checks["budget_16_recall_at_1"] = budget16
        if config["gallery_correction_fn"] == "sighting_1.0":
            reference = refs["Pr/Test Recall@1 with 100% Sighting Correction correction"]
            checks["budget_16_reference_metric"] = "Pr/Test Recall@1 with 100% Sighting Correction correction"
            checks["budget_16_reference_recall_at_1"] = reference
            checks["budget_16_reference_delta"] = budget16 - reference
            if strict and abs(budget16 - reference) > 1e-5:
                raise AssertionError(f"Budget-16 reference check failed: {budget16} vs {reference}")
        else:
            checks["budget_16_reference_metric"] = "no exact exported fixed-gallery G_S(20), query=sighting_1.0 cell"
    return checks


def write_outputs(
    rows: list[dict[str, Any]],
    interventions: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    sanity: dict[str, Any],
    temperature_info: dict[str, Any],
    margin_lambda_info: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(rows)
    results_frame.to_csv(output_dir / "prioritization_results.csv", index=False)
    with (output_dir / "prioritization_results.json").open("w") as f:
        json.dump(
            {
                "config": config,
                "temperature": temperature_info,
                "margin_lambda": margin_lambda_info,
                "sanity_checks": sanity,
                "rows": rows,
            },
            f,
            indent=2,
        )
    with (output_dir / "per_sighting_interventions.json").open("w") as f:
        json.dump(interventions, f, indent=2)


def run_plotter(output_dir: Path) -> None:
    from experiments.seek_intervention_top20.plot_results import plot_results

    plot_results(
        output_dir / "prioritization_results.csv",
        output_dir / "intervention_prioritization.pdf",
        output_dir / "intervention_prioritization.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default.json")
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a small deterministic subset and print budgets 1-3.")
    parser.add_argument("--max-sightings", type=int, default=None)
    parser.add_argument("--max-gallery-images", type=int, default=None)
    parser.add_argument("--max-budget", type=int, default=None)
    parser.add_argument("--policies", nargs="+", default=["random", "entropy", "entropy_full", "margin", "oracle"], choices=list(POLICY_DISPLAY))
    parser.add_argument("--random-rollouts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-calibration-images", type=int, default=None)
    parser.add_argument("--margin-lambda", type=float, default=None, help="Use a preselected frozen margin lambda instead of validation tuning.")
    parser.add_argument("--max-validation-sightings", type=int, default=None, help="Limit validation sightings for margin-lambda tuning.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict-reference", action="store_true")
    parser.add_argument("--endpoint-tolerance", type=float, default=1e-6)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        args.max_sightings = args.max_sightings or 5
        args.max_gallery_images = args.max_gallery_images or 1024
        args.max_calibration_images = args.max_calibration_images or 512
        args.max_validation_sightings = args.max_validation_sightings or args.max_sightings
        if args.random_rollouts is None:
            args.random_rollouts = 2
    max_budget = args.max_budget if args.max_budget is not None else int(config["budgets"])
    random_rollouts = args.random_rollouts if args.random_rollouts is not None else int(config["random_rollouts"])
    output_dir = repo_path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config["random_rollouts"] = random_rollouts
    config["output_dir"] = str(output_dir)

    checkpoint_dir = resolve_checkpoint_dir(config, download=args.download_checkpoint)
    dataset = EleHandler(subset=config["subset"], dataset_type=config["dataset"])
    train_indices, test_indices = load_indices(
        repo_path(config["train_indices_path"]),
        repo_path(config["test_indices_path"]),
    )
    test_indices = select_indices_for_sightings(dataset, test_indices, args.max_sightings, config["seed"])
    required_ele_ids = {str(dataset.dictonary.iloc[int(idx)]["ele_id"]) for idx in test_indices}
    train_indices = select_gallery_indices(
        dataset,
        train_indices,
        args.max_gallery_images,
        config["seed"],
        required_ele_ids,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )

    wandb_cfg = config["wandb"]
    run_name = config["experiment_name"] + ("_smoke" if args.smoke else "")
    with wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        group=wandb_cfg["group"],
        name=run_name,
        tags=wandb_cfg["tags"],
        config=config,
        mode=args.wandb_mode,
    ):
        with (output_dir / "run_metadata.json").open("w") as handle:
            json.dump({"wandb_run_id": wandb.run.id, "wandb_url": wandb.run.url,
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, handle, indent=2)
        projector, backbone_for_concepts, backbone, concept_head = build_models(config, checkpoint_dir, dataset)
        print("Collecting train cache", flush=True)
        train_cache = collect_cache(train_loader, dataset, backbone_for_concepts, backbone, concept_head, projector.device)

        retrieval = Retrieval(experiment_code=config["experiment_name"])
        gallery_fn = SEEK.get_correction_policy(config["gallery_correction_fn"], dataset, default_scope="sighting")
        print("Building fixed G_S(20) gallery embeddings", flush=True)
        gallery_embeddings = embeddings_from_cache(projector, train_cache, gallery_fn)
        gallery_labels = train_cache.labels
        temperature, temperature_info = calibrate_temperature(
            projector,
            retrieval,
            gallery_embeddings,
            gallery_labels,
            config,
            args,
        )
        wandb.summary.update({"temperature": temperature, **{f"temperature/{k}": v for k, v in temperature_info.items()}})

        margin_lambda_info: dict[str, Any] = {"mode": "not_requested"}
        if "margin" in args.policies:
            margin_lambda_info = tune_margin_lambda(
                projector,
                retrieval,
                train_cache,
                gallery_fn,
                temperature,
                config,
                args,
                max_budget,
            )
            config["selected_margin_lambda"] = margin_lambda_info["margin_lambda"]
            wandb.config.update({"selected_margin_lambda": config["selected_margin_lambda"]}, allow_val_change=True)
            wandb.summary.update(
                {
                    "margin_lambda": margin_lambda_info["margin_lambda"],
                    **{
                        f"margin_lambda/{key}": value
                        for key, value in margin_lambda_info.items()
                        if key not in {"candidates"} and not isinstance(value, (dict, list))
                    },
                }
            )

        print("Collecting test cache", flush=True)
        test_cache = collect_cache(test_loader, dataset, backbone_for_concepts, backbone, concept_head, projector.device)
        query_positions = select_query_positions(test_cache, args.max_sightings, config["seed"])
        query_cache = test_cache.subset(query_positions)
        np.savez_compressed(output_dir / "query_metadata.npz", labels=query_cache.labels.numpy(),
                            encounter_ids=np.asarray(query_cache.encounter_ids), indices=query_cache.indices.numpy())
        print(
            f"Evaluating {len(group_positions(query_cache.encounter_ids))} query sightings "
            f"({query_cache.labels.numel()} query images)",
            flush=True,
        )

        all_rows: list[dict[str, Any]] = []
        all_interventions: dict[str, Any] = {}
        for policy_key in args.policies:
            print(f"Starting policy: {policy_key}", flush=True)
            seeds = [config["seed"]]
            if policy_key == "random":
                seeds = [config["seed"] + i for i in range(random_rollouts)]
            for seed in seeds:
                trace = {}
                rows, interventions = run_single_policy(
                    policy_key,
                    seed,
                    projector,
                    retrieval,
                    query_cache,
                    gallery_embeddings,
                    gallery_labels,
                    temperature,
                    config,
                    max_budget,
                    evaluation_trace=trace,
                )
                trace["hits"] = np.stack(trace["hits"], axis=1)
                assert np.allclose(trace["hits"].mean(axis=0) * 100,
                                   [row["recall_at_1"] for row in rows], atol=1e-5)
                np.savez_compressed(output_dir / f"trace_{policy_key}_{seed}.npz", **trace)
                if "initial_top20_true_identity_coverage" in trace:
                    wandb.summary["initial_top20_true_identity_coverage"] = trace["initial_top20_true_identity_coverage"]
                all_rows.extend(rows)
                all_interventions[f"{policy_key}:{seed}"] = interventions
                pd.DataFrame(all_rows).to_csv(output_dir / "prioritization_results.csv", index=False)

        sanity = endpoint_checks(all_rows, max_budget, args.endpoint_tolerance)
        sanity["heatmap_reference"] = reference_checks(all_rows, config, args.strict_reference)
        sanity["checkpoint_dir"] = str(checkpoint_dir)
        wandb.summary.update({f"sanity/{key}": value for key, value in sanity.items() if not isinstance(value, dict)})
        write_outputs(all_rows, all_interventions, config, output_dir, sanity, temperature_info, margin_lambda_info)
        if not args.no_plot:
            run_plotter(output_dir)
            wandb.log({"retrieval_with_95ci": wandb.Image(str(output_dir / "intervention_prioritization.png")),
                       "confidence_intervals": wandb.Table(dataframe=pd.read_csv(output_dir / "confidence_intervals.csv"))})

    if args.smoke:
        interventions_path = output_dir / "per_sighting_interventions.json"
        with interventions_path.open() as f:
            interventions = json.load(f)
        for policy_seed, by_sighting in interventions.items():
            first_sighting = next(iter(by_sighting), None)
            if first_sighting is None:
                continue
            selected = [
                item["attribute_name"]
                for item in by_sighting[first_sighting]
                if item["budget"] in {1, 2, 3}
            ]
            print(f"{policy_seed} sighting {first_sighting} budgets 1-3: {selected}")


if __name__ == "__main__":
    main()
