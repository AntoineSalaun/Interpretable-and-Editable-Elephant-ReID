#!/usr/bin/env python3
"""Export ElephantBook-style fusion runs for the performance plots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_PATH = Path(__file__).resolve().parent / "ELEPHANTBOOK_vs_cor.csv"
GROUPS = {
    "ElephantBook_body_mega_baseline": "megadescriptor",
    "ElephantBook_baseline": "triple_crop_encoder",
}
OUTPUT_COLUMNS = [
    "exported_at_utc",
    "Name",
    "State",
    "created_at",
    "wandb_group",
    "wandb_name",
    "elephantbook_variant",
    "gallery_correction_scope",
    "correction_pct",
    "gallery_correction_fn",
    "query_correction_fn",
    "backbone_pretraining",
    "backbone_layer_norm",
    "elephantbook_backbone_weight",
    "elephantbook_wildcard_distance",
    "elephantbook_visual_with_ears",
    "Test-Recall@1",
]


def parse_policy(value) -> tuple[str | None, float | None]:
    if value is None:
        return None, None
    text = str(value).strip().lower()
    if "_" not in text:
        return None, None
    scope, raw = text.rsplit("_", 1)
    if scope not in {"sighting", "image"}:
        return None, None
    return scope, float(raw) * 100.0


def recall_at_1(run) -> float | None:
    for key in ("Test-Recall@1", "ElephantBook/Combined Recall@1"):
        value = run.summary.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def main() -> None:
    api = wandb.Api(timeout=60)
    rows_by_cell = {}
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for group, variant in GROUPS.items():
        runs = api.runs(
            PROJECT_PATH,
            filters={"group": group, "config.pipeline_name": "ElephantBook"},
            order="-created_at",
        )
        for run in runs:
            if run.state != "finished":
                continue
            config = dict(run.config or {})

            gallery_scope, gallery_pct = parse_policy(config.get("gallery_correction_fn"))
            query_scope, query_pct = parse_policy(config.get("query_correction_fn"))
            if gallery_scope not in {"sighting", "image"} or query_scope != "image":
                continue
            if gallery_pct is None or query_pct is None or abs(gallery_pct - query_pct) > 1e-6:
                continue

            cell = (variant, gallery_scope, gallery_pct)
            if cell in rows_by_cell:
                continue

            rows_by_cell[cell] = {
                "exported_at_utc": exported_at,
                "Name": run.name,
                "State": run.state,
                "created_at": str(run.created_at),
                "wandb_group": run.group,
                "wandb_name": run.name,
                "elephantbook_variant": variant,
                "gallery_correction_scope": gallery_scope,
                "correction_pct": gallery_pct,
                "gallery_correction_fn": config.get("gallery_correction_fn"),
                "query_correction_fn": config.get("query_correction_fn"),
                "backbone_pretraining": config.get("backbone_pretraining"),
                "backbone_layer_norm": config.get("backbone_layer_norm"),
                "elephantbook_backbone_weight": config.get("elephantbook_backbone_weight"),
                "elephantbook_wildcard_distance": config.get("elephantbook_wildcard_distance"),
                "elephantbook_visual_with_ears": config.get("elephantbook_visual_with_ears"),
                "Test-Recall@1": recall_at_1(run),
            }

    rows = [rows_by_cell[key] for key in sorted(rows_by_cell, key=lambda item: (item[0], item[1], item[2]))]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    for scope in ("sighting", "image"):
        for variant in GROUPS.values():
            scoped = df[
                (df["elephantbook_variant"] == variant)
                & (df["gallery_correction_scope"] == scope)
            ]
            present = set(scoped["correction_pct"].astype(int)) if not scoped.empty else set()
            missing = sorted(set(range(0, 101, 10)) - present)
            if missing and (scope == "sighting" or present):
                print(f"Missing finished ElephantBook {variant} {scope} gallery runs: {missing}")


if __name__ == "__main__":
    main()
