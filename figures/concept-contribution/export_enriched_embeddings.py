from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import wandb

from common import CHECKPOINT_DIR, DATA_DIR, LEVELS, ROOT, RUN_IDS, ensure_dirs

sys.path.insert(0, str(ROOT / "methods"))
from backbone import Backbone  # noqa: E402
from concept_head_tunneled import (  # noqa: E402
    ConceptHeadTunneled,
    CrossedHeadNN,
    MultiHeadNN,
    ThreeHeadNN,
    categorical_CE_loss,
)
from data_handler import EleHandler  # noqa: E402
from projector import Projector  # noqa: E402
from seek_code import SEEK  # noqa: E402


WANDB_ENTITY = "antoinesalaun-massachusetts-institute-of-technology"
WANDB_PROJECT = "CBM-ReID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mara")
    parser.add_argument("--subset", default="2+encounters")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    parser.add_argument("--download_from_wandb", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--concept_head_architecture", default="three", choices=["three", "crossed", "multi"])
    parser.add_argument("--concept_head_pretraining", default="concept_head_three_gold")
    parser.add_argument("--backbone_for_concepts_pretraining", default="backbone_for_concepts_three_gold")
    parser.add_argument("--projector_architecture", default="small")
    parser.add_argument("--layer_norm", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--run_ids", default=None, help="Optional comma list like 25:abc,50:def.")
    return parser.parse_args()


def parse_run_ids(raw: str | None) -> dict[int, str]:
    if not raw:
        return RUN_IDS
    out = {}
    for item in raw.split(","):
        level, run_id = item.split(":", 1)
        out[int(level)] = run_id.strip()
    return out


def download_run_files(run_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = {"projector.pt", "backbone_w.pt", "alpha.pt"}
    if all((out_dir / name).exists() for name in needed):
        return

    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    available = {file.name: file for file in run.files()}
    for name in needed:
        if name not in available:
            raise FileNotFoundError(f"{name} is missing from W&B run {run_id}")
        available[name].download(root=str(out_dir), replace=False)


def build_concept_head_layer(name: str):
    if name == "three":
        return ThreeHeadNN()
    if name == "crossed":
        return CrossedHeadNN()
    if name == "multi":
        return MultiHeadNN()
    raise ValueError(name)


def read_indices(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_models(args: argparse.Namespace, ckpt_dir: Path, num_classes: int):
    md_for_concepts = Backbone(
        model_name="MegaDescriptor",
        pretraining=args.backbone_for_concepts_pretraining,
        experiment_code="concept_contribution_backbone_for_concepts",
        num_classes=num_classes,
    )
    backbone = Backbone(
        model_name="MegaDescriptor",
        pretraining=str(ckpt_dir / "backbone_w.pt"),
        experiment_code="concept_contribution_backbone",
        num_classes=num_classes,
    )
    concept_head = ConceptHeadTunneled(
        loss=categorical_CE_loss,
        experiment_code="concept_contribution_concept_head",
        layer=build_concept_head_layer(args.concept_head_architecture),
        reset_weights=False,
        pretraining=args.concept_head_pretraining,
    )
    projector = Projector(
        experiment_code="concept_contribution_projector",
        reset_weights=True,
        network_type=args.projector_architecture,
        num_classes=num_classes,
        layer_norm=args.layer_norm,
        alpha_learnable=False,
    )
    projector.layer.load_state_dict(torch.load(ckpt_dir / "projector.pt", map_location=projector.device))
    alpha_data = torch.load(ckpt_dir / "alpha.pt", map_location=projector.device)
    projector.alpha = float(alpha_data["alpha"])
    return md_for_concepts, backbone, concept_head, projector


def collect_split(
    loader,
    dataset,
    md_for_concepts,
    backbone,
    concept_head,
    projector,
    correction_name: str,
    with_embeddings: bool,
) -> dict[str, torch.Tensor]:
    device = projector.device
    correction_fn = SEEK.get_correction_policy(correction_name, dataset)

    embeddings_out = []
    concepts_out = []
    labels_out = []
    indices_out = []

    projector.layer.eval()
    backbone.layer.eval()
    md_for_concepts.layer.eval()
    concept_head.layer.eval()

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        subject_seek = batch[4]
        sighting_seek = batch[5]
        left_ears = batch[6].to(device)
        right_ears = batch[7].to(device)
        idx = batch[10]

        with torch.no_grad():
            concept_features = md_for_concepts.forward(images, left_ears, right_ears)
            concept_logits = concept_head.layer(concept_features)
            edited_concepts = correction_fn(
                concept_logits,
                subject_seek,
                sighting_seek,
                ele_id=labels,
                idx=idx,
            )

            if with_embeddings:
                visual_embeddings = backbone.forward(images, left_ears, right_ears)
                projected_concepts = projector.layer(edited_concepts.to(device))
                if projector.layer_norm:
                    visual_embeddings = torch.nn.functional.layer_norm(
                        visual_embeddings,
                        visual_embeddings.size()[1:],
                    )
                    projected_concepts = torch.nn.functional.layer_norm(
                        projected_concepts,
                        projected_concepts.size()[1:],
                    )
                edited_embeddings = (1 - projector.alpha) * visual_embeddings + projector.alpha * projected_concepts
                embeddings_out.append(edited_embeddings.detach().cpu())

        concepts_out.append(edited_concepts.detach().cpu())
        labels_out.append(labels.detach().cpu())
        indices_out.append(idx.detach().cpu() if torch.is_tensor(idx) else torch.tensor(idx))

    out = {
        "concepts": torch.cat(concepts_out, dim=0),
        "labels": torch.cat(labels_out, dim=0),
        "indices": torch.cat(indices_out, dim=0).long(),
    }
    if with_embeddings:
        out["embeddings"] = torch.cat(embeddings_out, dim=0)
    return out


def main() -> None:
    args = parse_args()
    ensure_dirs()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This exporter uses the project models, which currently require CUDA.")

    run_ids = parse_run_ids(args.run_ids)
    train_indices = read_indices(ROOT / "weights" / "train_indices_gold.txt")
    test_indices = read_indices(ROOT / "weights" / "test_indicies_gold.txt")

    dataset = EleHandler(subset=args.subset, dataset_type=args.dataset)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with wandb.init(mode="disabled"):
        for level in args.levels:
            probability = level / 100.0
            ckpt_dir = args.checkpoint_dir / f"{level:03d}"
            if args.download_from_wandb:
                download_run_files(run_ids[level], ckpt_dir)

            models = load_models(args, ckpt_dir, num_classes=len(dataset.ele_id_to_label))
            train = collect_split(
                train_loader,
                dataset,
                *models,
                correction_name=f"sighting_{probability}",
                with_embeddings=True,
            )
            test = collect_split(
                test_loader,
                dataset,
                *models,
                correction_name=f"image_{probability}",
                with_embeddings=False,
            )

            torch.save(train, args.out_dir / f"train_{level:03d}.pt")
            torch.save(test, args.out_dir / f"test_{level:03d}.pt")
            print(f"[saved] {args.out_dir / f'train_{level:03d}.pt'}")
            print(f"[saved] {args.out_dir / f'test_{level:03d}.pt'}")

            del models, train, test
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

