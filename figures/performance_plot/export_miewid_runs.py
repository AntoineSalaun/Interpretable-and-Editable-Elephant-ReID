#!/usr/bin/env python3
"""Export MiewID baseline runs for the performance plots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_PATH = Path(__file__).resolve().parent / "MIEWID_vs_cor.csv"
GROUP = "MiewID_Mara"
OUTPUT_COLUMNS = [
    "exported_at_utc",
    "Name",
    "State",
    "created_at",
    "wandb_group",
    "wandb_name",
    "miewid_mode",
    "backbone_pretraining",
    "Test-Recall@1",
]


def recall_at_1(run) -> float | None:
    for key in ("Test-Recall@1", "MiewID/Test Recall@1"):
        value = run.summary.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def main() -> None:
    api = wandb.Api(timeout=60)
    runs = api.runs(
        PROJECT_PATH,
        filters={"group": GROUP, "config.pipeline_name": "MiewID"},
        order="-created_at",
    )
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows_by_mode = {}

    for run in runs:
        if run.state != "finished":
            continue
        config = dict(run.config or {})
        mode = str(config.get("miewid_mode", "")).lower()
        if mode in {"zero_shot", "out-of-box"}:
            mode = "out_of_box"
        if mode not in {"out_of_box", "finetuning"}:
            continue
        if mode in rows_by_mode:
            continue

        rows_by_mode[mode] = {
            "exported_at_utc": exported_at,
            "Name": run.name,
            "State": run.state,
            "created_at": str(run.created_at),
            "wandb_group": run.group,
            "wandb_name": run.name,
            "miewid_mode": mode,
            "backbone_pretraining": config.get("backbone_pretraining"),
            "Test-Recall@1": recall_at_1(run),
        }

    df = pd.DataFrame(
        [rows_by_mode[key] for key in sorted(rows_by_mode)],
        columns=OUTPUT_COLUMNS,
    )
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    missing = sorted({"out_of_box", "finetuning"} - set(rows_by_mode))
    if missing:
        print(f"Missing finished MiewID runs: {missing}")


if __name__ == "__main__":
    main()
