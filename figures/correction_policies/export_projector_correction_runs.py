#!/usr/bin/env python3
"""Export projector correction-sweep runs from W&B for the figures.

The sighting-level sweep already has 0..100% runs. The image-level sweep only
needs 10..100%; when any image-level runs are present, the sighting 0% run is
copied as image 0% because both policies are identical at zero correction.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent.parent

SWEEPS = {
    "sighting": "Projector_heatmap_three_alpha_learned_min10",
    "image": "Projector_heatmap_three_alpha_learned_image_training_min10",
}

METRIC_RE = re.compile(r"^Pr/Test Recall@")
MIN_BEST_EPOCH = 10
SKIP_STATES = {"failed", "crashed", "killed"}


def parse_training_correction(value) -> tuple[str | None, float | None]:
    if value is None:
        return None, None
    s = str(value).strip().lower()
    if "_" not in s:
        return None, None
    scope, raw = s.rsplit("_", 1)
    if scope not in {"sighting", "image"}:
        return None, None
    return scope, float(raw) * 100.0


def best_checkpoint_from_history(run) -> dict[str, float | int | None]:
    best = None
    for record in run.scan_history(keys=["Pr/Epoch", "Pr/Val Recall@1", "Pr/Alpha"]):
        epoch = record.get("Pr/Epoch")
        val_recall = record.get("Pr/Val Recall@1")
        if epoch is None or epoch < MIN_BEST_EPOCH or val_recall is None:
            continue
        if best is None or val_recall > best["Pr/Val Recall@1"]:
            best = record

    if best is None:
        return {"epoch": None, "val_recall": None, "alpha": None}
    return {
        "epoch": best.get("Pr/Epoch"),
        "val_recall": best.get("Pr/Val Recall@1"),
        "alpha": best.get("Pr/Alpha"),
    }


def has_required_diagonal_metric(summary: dict, train_pct: float) -> bool:
    pct = int(train_pct)
    keys = (
        f"Pr/Test Recall@1 with {pct}% Image Correction correction",
        f"Pr/Test Recall@1 with {pct}% Correction correction",
    )
    return any(summary.get(key) not in (None, "") for key in keys)


def latest_finished_rows(api: wandb.Api, scope: str, group: str) -> list[dict[str, object]]:
    runs = api.runs(PROJECT_PATH, filters={"group": group}, order="-created_at")
    rows_by_pct: dict[float, dict[str, object]] = {}
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for run in runs:
        if run.state in SKIP_STATES:
            continue
        if run.config.get("layer_norm") is not True:
            continue

        run_scope, train_pct = parse_training_correction(run.config.get("correction_at_training"))
        if run_scope != scope or train_pct is None:
            continue
        if train_pct in rows_by_pct:
            continue

        summary = dict(run.summary)
        if run.state != "finished" and not has_required_diagonal_metric(summary, train_pct):
            continue
        best_checkpoint = best_checkpoint_from_history(run)
        summary_best_epoch = summary.get("Pr/Best Alpha Epoch") or summary.get("Pr/Best Epoch")
        if summary_best_epoch is not None:
            summary_best_epoch = float(summary_best_epoch)
        use_summary_best = summary_best_epoch is not None and summary_best_epoch >= MIN_BEST_EPOCH
        if use_summary_best:
            best_alpha = summary.get("Pr/Best Alpha")
            if best_alpha is None:
                best_alpha = summary.get("learnt_alpha")
            if best_alpha is None:
                best_alpha = summary.get("learned_alpha")
        else:
            best_alpha = best_checkpoint["alpha"]
        row: dict[str, object] = {
            "exported_at_utc": exported_at,
            "Name": run.name,
            "State": run.state,
            "created_at": str(run.created_at),
            "wandb_group": group,
            "wandb_name": run.name,
            "training_correction_scope": scope,
            "train_correction_pct": train_pct,
            "correction_at_training": run.config.get("correction_at_training"),
            "layer_norm": run.config.get("layer_norm"),
            "alpha_learnable": run.config.get("alpha_learnable"),
            "projector_training_policy": run.config.get("projector_training_policy"),
            "projector_lr": run.config.get("projector_lr"),
            "projector_wd": run.config.get("projector_wd"),
            "backbone_pretraining": run.config.get("backbone_pretraining"),
            "backbone_for_concepts_pretraining": run.config.get("backbone_for_concepts_pretraining"),
            "concept_head_pretraining": run.config.get("concept_head_pretraining"),
            "Pr/Alpha": summary.get("Pr/Alpha"),
            "Pr/Best Alpha": best_alpha,
            "Pr/Best Alpha Epoch": summary.get("Pr/Best Alpha Epoch") if use_summary_best else best_checkpoint["epoch"],
            "Pr/Best Epoch": summary.get("Pr/Best Epoch") if use_summary_best else best_checkpoint["epoch"],
            "Pr/Best Val Recall@1": summary.get("Pr/Best Val Recall@1") if use_summary_best else best_checkpoint["val_recall"],
            "learnt_alpha": best_alpha,
            "learned_alpha": best_alpha,
        }

        for key, value in summary.items():
            if METRIC_RE.match(key):
                row[key] = value

        rows_by_pct[train_pct] = row

    return [rows_by_pct[pct] for pct in sorted(rows_by_pct)]


def add_shared_image_zero(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scopes = {row["training_correction_scope"] for row in rows}
    if "image" not in scopes:
        return rows

    sighting_zero = next(
        (
            row
            for row in rows
            if row["training_correction_scope"] == "sighting"
            and float(row["train_correction_pct"]) == 0.0
        ),
        None,
    )
    has_image_zero = any(
        row["training_correction_scope"] == "image"
        and float(row["train_correction_pct"]) == 0.0
        for row in rows
    )
    if sighting_zero is None or has_image_zero:
        return rows

    image_zero = dict(sighting_zero)
    image_zero["Name"] = "Heatmap_three_image_00cor_shared"
    image_zero["wandb_name"] = "Heatmap_three_image_00cor_shared"
    image_zero["training_correction_scope"] = "image"
    image_zero["correction_at_training"] = "image_0.0"
    rows.append(image_zero)
    return rows


def write_outputs(df: pd.DataFrame) -> None:
    paths = [
        OUT_DIR / "CHAIR_vs_cor.csv",
        OUT_DIR / "column_oracle.csv",
        REPO_ROOT / "figures" / "performance_plot" / "CHAIR_vs_cor.csv",
        REPO_ROOT / "figures" / "alpha" / "alpha_vs_cor.csv",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Wrote {path}")


def main() -> None:
    api = wandb.Api(timeout=60)
    rows: list[dict[str, object]] = []
    for scope, group in SWEEPS.items():
        scope_rows = latest_finished_rows(api, scope, group)
        rows.extend(scope_rows)
        expected = set(range(0 if scope == "sighting" else 10, 101, 10))
        present = {int(row["train_correction_pct"]) for row in scope_rows}
        missing = sorted(expected - present)
        if missing:
            print(f"Missing finished {scope} runs: {missing}")

    rows = add_shared_image_zero(rows)
    if not rows:
        raise RuntimeError("No finished correction-sweep runs found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["training_correction_scope", "train_correction_pct"])
    write_outputs(df)


if __name__ == "__main__":
    main()
