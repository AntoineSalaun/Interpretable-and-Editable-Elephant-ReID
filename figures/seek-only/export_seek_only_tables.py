#!/usr/bin/env python3
"""Export recent SEEK-only retrieval tables from W&B.

These tables use runs launched from ``bash/seek_retrieval.sh``: gallery
correction is sighting-level, and query correction is either image-level or full
sighting-level.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_DIR = Path(__file__).resolve().parent

GALLERY_ROWS = (
    ("0\\% sighting", "sighting_0.0"),
    ("25\\% sighting", "sighting_0.25"),
    ("50\\% sighting", "sighting_0.5"),
    ("75\\% sighting", "sighting_0.75"),
    ("100\\% sighting", "sighting_1.0"),
)

QUERY_COLUMNS = (
    ("0\\% image", "image_0.0"),
    ("25\\% image", "image_0.25"),
    ("50\\% image", "image_0.5"),
    ("75\\% image", "image_0.75"),
    ("100\\% image", "image_1.0"),
    ("100\\% sighting", "sighting_1.0"),
)


@dataclass(frozen=True)
class DistanceSpec:
    distance: str
    group: str
    short_name: str
    caption_distance: str
    label: str


DISTANCES = (
    DistanceSpec(
        distance="cosine_sim",
        group="SEEK_retrieval_cosine",
        short_name="cosine",
        caption_distance=r"\textbf{cosine similarity}",
        label="tab:SEEK_cosine",
    ),
    DistanceSpec(
        distance="seek_homemade_3",
        group="SEEK_retrieval_homemade",
        short_name="homemade",
        caption_distance=r"\textbf{our SEEK-matching} distance",
        label="tab:SEEK_homemade",
    ),
)


def recall_at_1(run: Any) -> float:
    for key in ("Test-Recall@1", "Test Recall@1"):
        value = run.summary.get(key)
        if value not in (None, ""):
            return float(value)
    raise RuntimeError(f"Run {run.name} has no Recall@1 summary")


def normalize_policy(policy: Any) -> str:
    return str(policy).strip().lower()


def fetch_recent_rows(api: wandb.Api, spec: DistanceSpec) -> list[dict[str, Any]]:
    rows_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    gallery_labels = {policy: label for label, policy in GALLERY_ROWS}
    query_labels = {policy: label for label, policy in QUERY_COLUMNS}

    runs = api.runs(
        PROJECT_PATH,
        filters={"group": spec.group, "config.pipeline_name": "Retrieval_on_SEEK_codes"},
        order="-created_at",
    )

    for run in runs:
        if run.state != "finished":
            continue
        cfg = dict(run.config or {})
        if cfg.get("concept_distance") != spec.distance:
            continue

        gallery_policy = normalize_policy(cfg.get("gallery_correction_fn"))
        query_policy = normalize_policy(cfg.get("query_correction_fn"))
        if gallery_policy not in gallery_labels or query_policy not in query_labels:
            continue
        cell = (gallery_policy, query_policy)
        if cell in rows_by_cell:
            continue

        rows_by_cell[cell] = {
            "table": spec.short_name,
            "gallery_label": gallery_labels[gallery_policy],
            "query_label": query_labels[query_policy],
            "wandb_group": run.group,
            "wandb_name": run.name,
            "created_at": str(run.created_at),
            "state": run.state,
            "concept_distance": cfg.get("concept_distance"),
            "gallery_correction_fn": cfg.get("gallery_correction_fn"),
            "query_correction_fn": cfg.get("query_correction_fn"),
            "recall_at_1": recall_at_1(run),
        }

    rows = []
    for gallery_label, gallery_policy in GALLERY_ROWS:
        for query_label, query_policy in QUERY_COLUMNS:
            row = rows_by_cell.get((gallery_policy, query_policy))
            if row is None:
                row = {
                    "table": spec.short_name,
                    "gallery_label": gallery_label,
                    "query_label": query_label,
                    "wandb_group": spec.group,
                    "wandb_name": "",
                    "created_at": "",
                    "state": "missing",
                    "concept_distance": spec.distance,
                    "gallery_correction_fn": gallery_policy,
                    "query_correction_fn": query_policy,
                    "recall_at_1": None,
                }
            rows.append(row)
    return rows


def make_table(spec: DistanceSpec, rows: list[dict[str, Any]]) -> str:
    caption = (
        r"\caption{Recall@1 (\%) for retrieval solely on SEEK codes using "
        + spec.caption_distance
        + r". Rows correspond to sighting-level gallery corrections; columns correspond to query treatments.}"
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        caption,
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccccc}",
        r"\headrow",
        r"\thead{Gallery correction} & "
        + " & ".join(r"\thead{" + label + r"}" for label, _ in QUERY_COLUMNS)
        + r"\\",
    ]

    values = {
        (row["gallery_label"], row["query_label"]): row["recall_at_1"]
        for row in rows
    }
    for gallery_label, _ in GALLERY_ROWS:
        entries = []
        for query_label, _ in QUERY_COLUMNS:
            value = values[(gallery_label, query_label)]
            entries.append("--" if value is None or pd.isna(value) else f"{float(value):.2f}")
        lines.append(f"{gallery_label} & " + " & ".join(entries) + r" \\")

    lines.extend(
        [
            r"\hline  % Please only put a hline at the end of the table",
            r"\end{tabular}",
            r"\end{threeparttable}",
            rf"\label{{{spec.label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    api = wandb.Api(timeout=60)
    export_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_rows: list[dict[str, Any]] = []
    tables: list[str] = []
    for spec in DISTANCES:
        rows = fetch_recent_rows(api, spec)
        for row in rows:
            row["exported_at_utc"] = export_time
        all_rows.extend(rows)
        table = make_table(spec, rows)
        tables.append(table)
        (OUT_DIR / f"SEEK_{spec.short_name}.tex").write_text(table)

    write_csv(OUT_DIR / "seek_only_table_raw.csv", all_rows)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "seek_only_table.csv", index=False)
    (OUT_DIR / "seek_only_tables.tex").write_text("\n".join(tables))

    (OUT_DIR / "README.md").write_text(
        "# SEEK-only retrieval tables\n\n"
        f"Exported from W&B at `{export_time}`.\n\n"
        f"- Project: `{PROJECT_PATH}`\n"
        "- Source runs: `bash/seek_retrieval.sh`, groups "
        "`SEEK_retrieval_cosine` and `SEEK_retrieval_homemade`.\n"
        "- Gallery correction: `sighting_p`.\n"
        "- Gallery rows: `sighting_0.0`, `sighting_0.25`, `sighting_0.5`, "
        "`sighting_0.75`, and `sighting_1.0`.\n"
        "- Query columns: `image_0.0`, `image_0.25`, `image_0.5`, "
        "`image_0.75`, `image_1.0`, and `sighting_1.0`.\n"
        "- Missing cells are written as `--` until the corresponding W&B runs finish.\n\n"
        "Generated files:\n\n"
        "- `seek_only_tables.tex`\n"
        "- `SEEK_cosine.tex`\n"
        "- `SEEK_homemade.tex`\n"
        "- `seek_only_table_raw.csv`\n"
    )

    print(f"Wrote {OUT_DIR / 'seek_only_tables.tex'}")
    print(f"Wrote {OUT_DIR / 'seek_only_table_raw.csv'}")


if __name__ == "__main__":
    main()
