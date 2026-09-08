#!/usr/bin/env python3
"""Export diagonal SEEK-only retrieval runs for the performance plots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_PATH = Path(__file__).resolve().parent / "SEEK_vs_cor.csv"
GROUP = "SEEK_retrieval_homemade"
DISTANCE = "seek_homemade_3"


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
    for key in ("Test-Recall@1", "Test Recall@1"):
        value = run.summary.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def main() -> None:
    api = wandb.Api(timeout=60)
    runs = api.runs(
        PROJECT_PATH,
        filters={"group": GROUP, "config.pipeline_name": "Retrieval_on_SEEK_codes"},
        order="-created_at",
    )
    rows_by_cell = {}
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for run in runs:
        if run.state != "finished":
            continue
        config = dict(run.config or {})
        if config.get("concept_distance") != DISTANCE:
            continue

        gallery_scope, gallery_pct = parse_policy(config.get("gallery_correction_fn"))
        query_scope, query_pct = parse_policy(config.get("query_correction_fn"))
        if gallery_scope not in {"sighting", "image"} or query_scope != "image":
            continue
        if gallery_pct is None or query_pct is None or abs(gallery_pct - query_pct) > 1e-6:
            continue

        cell = (gallery_scope, gallery_pct)
        if cell in rows_by_cell:
            continue

        rows_by_cell[cell] = {
            "exported_at_utc": exported_at,
            "Name": run.name,
            "State": run.state,
            "created_at": str(run.created_at),
            "wandb_group": run.group,
            "wandb_name": run.name,
            "concept_distance": config.get("concept_distance"),
            "gallery_correction_scope": gallery_scope,
            "correction_pct": gallery_pct,
            "gallery_correction_fn": config.get("gallery_correction_fn"),
            "query_correction_fn": config.get("query_correction_fn"),
            "Test-Recall@1": recall_at_1(run),
        }

    rows = [rows_by_cell[key] for key in sorted(rows_by_cell, key=lambda item: (item[0], item[1]))]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    for scope in ("sighting", "image"):
        present = set(df[df["gallery_correction_scope"] == scope]["correction_pct"].astype(int)) if not df.empty else set()
        missing = sorted(set(range(0, 101, 10)) - present)
        if missing:
            print(f"Missing finished SEEK-only {scope} gallery runs: {missing}")


if __name__ == "__main__":
    main()
