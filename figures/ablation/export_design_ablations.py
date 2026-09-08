#!/usr/bin/env python3
"""Export the SEEK-CBM design ablation table from W&B."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
GROUP = "Projector_training_alpha_ablation_three"
OUT_DIR = Path(__file__).resolve().parent
MIN_CREATED_AT = "2026-05-20"


@dataclass(frozen=True)
class AblationSpec:
    run_name: str
    training: str
    fusion: str


SPECS = (
    AblationSpec("Ablation_Seq_fixed_alpha", "Sequential", r"$\alpha=0.5$"),
    AblationSpec("Ablation_Seq_learned_alpha", "Sequential", r"$\alpha^*$"),
    AblationSpec("Ablation_Joint_fixed_alpha", "Joint", r"$\alpha=0.5$"),
    AblationSpec("Ablation_Joint_learned_alpha", "Joint", r"$\alpha^*$"),
)

METRICS = (
    ("0", "0\\%"),
    ("50", "50\\%"),
    ("100", "100\\%"),
    ("ORACLE", "Sighting"),
)


def metric_key(token: str) -> str:
    name = "ORACLE" if token == "ORACLE" else f"{token}% Image Correction"
    return f"Pr/Test Recall@1 with {name} correction"


def metric_value(run: wandb.apis.public.Run, token: str) -> float:
    if token == "ORACLE":
        candidates = (
            "Pr/Test Recall@1 with 100% Sighting Correction correction",
            "Pr/Test Recall@1 with ORACLE correction",
            "Pr/Test Recall@1 with ORACLE correction aggregate_gallery_seeks=False",
        )
    else:
        candidates = (
            metric_key(token),
            f"Pr/Test Recall@1 with {token}% Sighting-gallery/Image-query Correction correction",
            f"Pr/Test Recall@1 with {token}% Correction correction",
            f"Pr/Test Recall@1 with {token}% Correction correction aggregate_gallery_seeks=False",
        )
    for key in candidates:
        if key in run.summary:
            return float(run.summary[key])
    raise KeyError(f"Run {run.name} has none of: {candidates}")


def latest_runs(api: wandb.Api) -> dict[str, wandb.apis.public.Run]:
    runs = api.runs(PROJECT_PATH, filters={"group": GROUP}, order="-created_at")
    out = {}
    for run in runs:
        if run.state != "finished":
            continue
        if str(run.created_at) < MIN_CREATED_AT:
            continue
        if run.name not in out:
            out[run.name] = run
    return out


def bold_best(values: list[float], value: float) -> str:
    if abs(value - max(values)) < 1e-10:
        return f"\\textbf{{{value:.2f}}}"
    return f"{value:.2f}"


def make_table(rows: list[dict[str, object]]) -> str:
    best = {
        token: [float(row[f"r1_{token}"]) for row in rows]
        for token, _ in METRICS
    }

    lines = [
        r"\begin{table}",
        r"\caption{\textbf{Design ablations for SEEK-CBM} trained with 50\% correction. Recall@1 (\%) is reported at four inference-time correction levels. We compare sequential training vs.\ joint training, and fixed $\alpha$ vs.\ learned $\alpha$. Each entry reports the Recall@1 (\%) for the latest completed rerun of that setting.}",
        r"\centering",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llcccc}",
        r"\headrow",
        r"\thead{Training} & \thead{Fusion} & \thead{0\%} & \thead{50\%} & \thead{100\%} & \thead{Sighting}\\",
    ]

    for row in rows:
        values = [
            bold_best(best[token], float(row[f"r1_{token}"]))
            for token, _ in METRICS
        ]
        lines.append(
            f"{row['training']} & {row['fusion']} & "
            + " & ".join(values)
            + r" \\"
        )
        if row["training"] == "Sequential" and row["fusion"] == r"$\alpha^*$":
            lines.append(r"\hline")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{threeparttable}",
            r"\label{tab:design_ablations}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    api = wandb.Api(timeout=60)
    runs_by_name = latest_runs(api)
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []

    for spec in SPECS:
        run = runs_by_name.get(spec.run_name)
        if run is None:
            raise RuntimeError(f"Missing finished rerun: {GROUP}/{spec.run_name}")

        row = {
            "exported_at_utc": exported_at,
            "wandb_group": GROUP,
            "wandb_name": run.name,
            "created_at": str(run.created_at),
            "training": spec.training,
            "fusion": spec.fusion,
            "projector_training_policy": run.config.get("projector_training_policy"),
            "alpha_learnable": run.config.get("alpha_learnable"),
            "layer_norm": run.config.get("layer_norm"),
            "correction_at_training": run.config.get("correction_at_training"),
            "projector_lr": run.config.get("projector_lr"),
            "projector_wd": run.config.get("projector_wd"),
        }
        for token, _ in METRICS:
            row[f"r1_{token}"] = metric_value(run, token)
        rows.append(row)

    csv_path = OUT_DIR / "design_ablations.csv"
    tex_path = OUT_DIR / "design_ablations.tex"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path.write_text(make_table(rows))
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
