#!/usr/bin/env python3
"""Backfill image-inference projector metrics for existing heatmap checkpoints."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import wandb


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "methods"
sys.path.insert(0, str(METHODS_DIR))

from backbone import Backbone
from concept_head_tunneled import ConceptHeadTunneled, ThreeHeadNN, categorical_CE_loss
from data_handler import EleHandler
from projector import Projector
from retrieval import Retrieval
from seek_code import SEEK


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
GROUPS = (
    "Projector_heatmap_three_alpha_learned_min10",
    "Projector_heatmap_three_alpha_learned_image_training_min10",
)
EXPERIMENT_ROOT = REPO_ROOT.parent / "experiments"


def load_indices() -> tuple[list[int], list[int]]:
    weights_dir = REPO_ROOT / "weights"
    with (weights_dir / "train_indices_gold.txt").open() as f:
        train_indices = [int(line.strip()) for line in f]
    with (weights_dir / "test_indicies_gold.txt").open() as f:
        test_indices = [int(line.strip()) for line in f]
    return train_indices, test_indices


def build_models(run, dataset):
    cfg = run.config
    with wandb.init(project="CBM-ReID", mode="disabled"):
        backbone_for_concepts = Backbone(
            model_name="MegaDescriptor",
            pretraining=cfg.get("backbone_for_concepts_pretraining", "backbone_for_concepts_three_gold"),
            experiment_code=run.name,
            num_classes=len(dataset.ele_id_to_label),
        )
        backbone = Backbone(
            model_name="MegaDescriptor",
            pretraining=cfg.get("backbone_pretraining", "backbone_normalized_gold"),
            experiment_code=run.name,
            num_classes=len(dataset.ele_id_to_label),
        )
        concept_head = ConceptHeadTunneled(
            loss=categorical_CE_loss,
            wd=cfg.get("concept_head_wd"),
            experiment_code=run.name,
            layer=ThreeHeadNN(),
            reset_weights=False,
            pretraining=cfg.get("concept_head_pretraining", "concept_head_three_gold"),
        )
        projector = Projector(
            loss_type="ArcFace",
            lr=cfg.get("projector_lr", 1e-4),
            scale=64,
            margin=0.5,
            experiment_code=run.name,
            intervention_fn=None,
            alpha=cfg.get("alpha", 0.5),
            reset_weights=True,
            network_type=cfg.get("projector_architecture", "small"),
            num_classes=len(dataset.ele_id_to_label),
            wd=cfg.get("projector_wd", 1e-6),
            layer_norm=cfg.get("layer_norm", True),
            alpha_learnable=cfg.get("alpha_learnable", True),
        )

    checkpoint_dir = EXPERIMENT_ROOT / f"exp_{run.name}"
    projector.layer.load_state_dict(torch.load(checkpoint_dir / "projector.pt", map_location=projector.device))
    backbone.layer.load_state_dict(torch.load(checkpoint_dir / "backbone_w.pt", map_location=backbone.device))
    alpha_path = checkpoint_dir / "alpha.pt"
    if alpha_path.exists():
        alpha_data = torch.load(alpha_path, map_location=projector.device)
        if projector.alpha_learnable:
            projector.alpha.data.fill_(float(alpha_data["alpha"]))
        else:
            projector.alpha = float(alpha_data["alpha"])

    backbone_for_concepts.freeze()
    concept_head.freeze()
    projector.freeze()
    backbone.freeze()
    return projector, backbone_for_concepts, backbone, concept_head


def collect_cache(loader, backbone_for_concepts, backbone, concept_head, device):
    embeddings = []
    concept_logits = []
    labels = []
    subject_seeks = []
    sighting_seeks = []
    indices = []
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

            embeddings.append(visual.cpu())
            concept_logits.append(logits.cpu())
            labels.append(ele_id_label.cpu())
            subject_seeks.append(subject_seek.cpu())
            sighting_seeks.append(sighting_seek.cpu())
            indices.append(idx.cpu() if torch.is_tensor(idx) else torch.tensor(idx))

    return {
        "embeddings": torch.cat(embeddings),
        "concept_logits": torch.cat(concept_logits),
        "labels": torch.cat(labels),
        "subject_seeks": torch.cat(subject_seeks),
        "sighting_seeks": torch.cat(sighting_seeks),
        "indices": torch.cat(indices),
    }


def embeddings_from_cache(projector, cache, intervention_fn, batch_size=512):
    output = []
    device = projector.device
    projector.layer.eval()
    with torch.no_grad():
        for start in range(0, len(cache["labels"]), batch_size):
            end = start + batch_size
            visual = cache["embeddings"][start:end].to(device)
            logits = cache["concept_logits"][start:end].to(device)
            subject_seek = cache["subject_seeks"][start:end].to(device)
            sighting_seek = cache["sighting_seeks"][start:end].to(device)
            labels = cache["labels"][start:end].to(device)
            idx = cache["indices"][start:end]

            edited = intervention_fn(logits, subject_seek, sighting_seek, ele_id=labels, idx=idx)
            projected = projector.layer(edited.to(device))
            if projector.layer_norm:
                visual = torch.nn.functional.layer_norm(visual, visual.size()[1:])
                projected = torch.nn.functional.layer_norm(projected, projected.size()[1:])
            output.append(((1 - projector.alpha) * visual + projector.alpha * projected).cpu())
    return torch.cat(output)


def eval_policy(projector, retrieval, train_cache, test_cache, name, gallery_fn, query_fn):
    gallery_embeddings = embeddings_from_cache(projector, train_cache, gallery_fn)
    query_embeddings = embeddings_from_cache(projector, test_cache, query_fn)
    gallery_labels = train_cache["labels"]
    query_labels = test_cache["labels"]
    similarity = retrieval.similarity_matrix(query_embeddings.to(projector.device), gallery_embeddings.to(projector.device))
    metrics = {
        k: retrieval.compute_recall_at_k(
            similarity_matrix=similarity,
            query_labels=query_labels.to(projector.device),
            gallery_labels=gallery_labels.to(projector.device),
            k=k,
        )
        for k in [1, 5, 10, 20, 100]
    }
    return {f"Pr/Test Recall@{k} with {name} correction": v * 100 for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    dataset = EleHandler(subset="2+encounters", dataset_type="mara")
    train_indices, test_indices = load_indices()
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

    api = wandb.Api(timeout=60)
    seen_names = set()
    for group in GROUPS:
        runs = api.runs(PROJECT_PATH, filters={"group": group}, order="-created_at")
        for run in runs:
            if run.state != "finished" or run.name in seen_names:
                continue
            if args.run_name is not None and run.name != args.run_name:
                continue
            print(f"Evaluating {run.name}", flush=True)
            if not (EXPERIMENT_ROOT / f"exp_{run.name}" / "projector.pt").exists():
                print(f"Skipping {run.name}: missing local checkpoint", flush=True)
                continue
            seen_names.add(run.name)
            projector, backbone_for_concepts, backbone, concept_head = build_models(run, dataset)
            train_cache = collect_cache(train_loader, backbone_for_concepts, backbone, concept_head, projector.device)
            test_cache = collect_cache(test_loader, backbone_for_concepts, backbone, concept_head, projector.device)
            retrieval = Retrieval(experiment_code="image_inference_backfill")

            updates = {}
            for percentage in range(0, 101, 10):
                probability = percentage / 100.0
                image_fn = SEEK.get_correction_policy(f"image_{probability}", dataset)
                updates.update(
                    eval_policy(
                        projector,
                        retrieval,
                        train_cache,
                        test_cache,
                        f"{percentage}% Image Correction",
                        image_fn,
                        image_fn,
                    )
                )

            sighting_fn = SEEK.get_correction_policy("sighting_1.0", dataset)
            updates.update(
                eval_policy(
                    projector,
                    retrieval,
                    train_cache,
                    test_cache,
                    "100% Sighting Correction",
                    sighting_fn,
                    sighting_fn,
                )
            )
            for key, value in updates.items():
                run.summary[key] = value
            run.summary["projector_image_inference_backfilled"] = True
            run.summary.update()
            print(f"Updated {run.name} with {len(updates)} metrics", flush=True)
            if args.run_name is not None:
                return


if __name__ == "__main__":
    main()
