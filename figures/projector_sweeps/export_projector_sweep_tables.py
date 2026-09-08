#!/usr/bin/env python3
"""Export projector sweep Recall@1 tables from W&B.

The four sweeps are defined in bash/projector.sh. This script intentionally
uses the exact W&B groups and run-name patterns from that file, then selects the
latest finished run for every expected (lr, wd) pair.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


PROJECT_PATH = "antoinesalaun-massachusetts-institute-of-technology/CBM-ReID"
OUT_DIR = Path(__file__).resolve().parent

LRS = ("1e-3", "1e-4", "1e-5")
WDS = ("1e-4", "1e-5", "1e-6")

POLICIES = (
    ("0", "0\\% correction"),
    ("50", "50\\% correction"),
    ("100", "100\\% correction"),
    ("ORACLE", "ORACLE correction"),
)


@dataclass(frozen=True)
class SweepSpec:
    group: str
    name_prefix: str
    caption: str
    label: str
    training_policy: str
    alpha_desc: str
    alpha_learnable: bool


SWEEPS = (
    SweepSpec(
        group="Projector_sweep_seq_learned_alpha_100cor",
        name_prefix="SeqLearned_100cor",
        caption=(
            "PROJECTOR TRAINING: Sequential - "
            "\\texttt{aggregate\\_gallery\\_seeks=False} -- alpha=LEARNABLE"
        ),
        label="tab:projector_seq_learned_alpha_r1_noagg_inference_correction",
        training_policy="sequential",
        alpha_desc="learnable",
        alpha_learnable=True,
    ),
    SweepSpec(
        group="Projector_sweep_seq_fixed_alpha_100cor",
        name_prefix="SeqFixed_100cor",
        caption=(
            "PROJECTOR TRAINING: Sequential - "
            "\\texttt{aggregate\\_gallery\\_seeks=False} -- alpha=0.5"
        ),
        label="tab:projector_seq_fixed_alpha_r1_noagg_inference_correction",
        training_policy="sequential",
        alpha_desc="0.5",
        alpha_learnable=False,
    ),
    SweepSpec(
        group="Projector_sweep_joint_learned_alpha_100cor",
        name_prefix="JointLearned_100cor",
        caption=(
            "PROJECTOR TRAINING: Joint - "
            "\\texttt{aggregate\\_gallery\\_seeks=False} -- alpha=LEARNABLE"
        ),
        label="tab:projector_joint_learned_alpha_r1_noagg_inference_correction",
        training_policy="joint",
        alpha_desc="learnable",
        alpha_learnable=True,
    ),
    SweepSpec(
        group="Projector_sweep_joint_fixed_alpha_100cor",
        name_prefix="JointFixed_100cor",
        caption=(
            "PROJECTOR TRAINING: Joint - "
            "unaggregated SEEK inference -- alpha=0.5"
        ),
        label="tab:projector_joint_fixed_alpha_r1_inference_correction",
        training_policy="joint",
        alpha_desc="0.5",
        alpha_learnable=False,
    ),
)


def metric_keys(policy: str) -> list[str]:
    if policy == "ORACLE":
        name = "ORACLE"
    else:
        name = f"{policy}% Correction"
    return [
        f"Pr/Test Recall@1 with {name} correction",
        f"Pr/Test Recall@1 with {name} correction aggregate_gallery_seeks=False",
    ]


def expected_name(spec: SweepSpec, lr: str, wd: str) -> str:
    return f"{spec.name_prefix}_lr{lr}_wd{wd}"


def config_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else None


def sci_token(value: Any) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def summary_value(summary: Any, key: str) -> float | None:
    value = summary.get(key, None)
    if value in ("", None):
        return None
    return float(value)


def latest_finished_by_name(api: wandb.Api, spec: SweepSpec) -> dict[str, Any]:
    runs = list(
        api.runs(
            PROJECT_PATH,
            filters={"group": spec.group},
            order="-created_at",
        )
    )
    by_name: dict[str, Any] = {}
    for run in runs:
        if run.state != "finished":
            continue
        if run.name not in by_name:
            by_name[run.name] = run
    return by_name


def bold_best(values: list[float | None], value: float | None) -> str:
    if value is None:
        return "--"
    present = [v for v in values if v is not None]
    if present and abs(value - max(present)) < 1e-10:
        return f"\\textbf{{{value:.2f}}}"
    return f"{value:.2f}"


def make_table(spec: SweepSpec, rows: list[dict[str, Any]]) -> str:
    columns = [policy for policy, _ in POLICIES]
    best_values = {
        policy: [row[f"r1_{policy}"] for row in rows]
        for policy in columns
    }

    lines = [
        "% ----------------------------",
        "\\begin{table}[H]",
        f"\\caption{{{spec.caption}}}",
        "\\begin{threeparttable}",
        "\\begin{tabular}{lcccc}",
        "\\headrow",
        "\\thead{(lr, wd)} & "
        + " & ".join(f"\\thead{{{heading}}}" for _, heading in POLICIES)
        + r"\\",
    ]

    for row in rows:
        vals = [
            bold_best(best_values[policy], row[f"r1_{policy}"])
            for policy in columns
        ]
        lines.append(
            f"({row['projector_lr']}, {row['projector_wd']}) & "
            + " & ".join(vals)
            + r"\\"
        )

    lines.extend(
        [
            "\\hline  % Please only put a hline at the end of the table",
            "\\end{tabular}",
            "\\end{threeparttable}",
            f"\\label{{{spec.label}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    api = wandb.Api(timeout=60)
    export_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_rows: list[dict[str, Any]] = []
    tables: list[str] = []

    for spec in SWEEPS:
        runs_by_name = latest_finished_by_name(api, spec)
        sweep_rows: list[dict[str, Any]] = []

        for lr in LRS:
            for wd in WDS:
                name = expected_name(spec, lr, wd)
                run = runs_by_name.get(name)
                if run is None:
                    raise RuntimeError(f"Missing finished W&B run: group={spec.group}, name={name}")

                config = dict(run.config or {})
                summary = run.summary
                row: dict[str, Any] = {
                    "exported_at_utc": export_time,
                    "wandb_project": PROJECT_PATH,
                    "group": spec.group,
                    "run_name": run.name,
                    "run_id": run.id,
                    "state": run.state,
                    "created_at": str(getattr(run, "created_at", "")),
                    "updated_at": str(getattr(run, "updated_at", "")),
                    "projector_training_policy": config_value(config, "projector_training_policy"),
                    "alpha": config_value(config, "alpha"),
                    "alpha_learnable": config_value(config, "alpha_learnable"),
                    "correction_at_training": config_value(config, "correction_at_training"),
                    "backbone_pretraining": config_value(config, "backbone_pretraining"),
                    "concept_head_pretraining": config_value(config, "concept_head_pretraining"),
                    "projector_lr": lr,
                    "projector_wd": wd,
                }

                for policy, _ in POLICIES:
                    key = None
                    value = None
                    for candidate_key in metric_keys(policy):
                        value = summary_value(summary, candidate_key)
                        if value is not None:
                            key = candidate_key
                            break
                    if value is None:
                        raise RuntimeError(
                            "Missing required summary metric "
                            f"{metric_keys(policy)!r} for group={spec.group}, run={run.name}"
                        )
                    row[f"r1_{policy}"] = value
                    row[f"metric_key_{policy}"] = key

                cfg_lr = config_value(config, "projector_lr")
                cfg_wd = config_value(config, "projector_wd")
                if cfg_lr is not None and sci_token(cfg_lr) != lr:
                    raise RuntimeError(f"Unexpected lr in {run.name}: config={cfg_lr}, expected={lr}")
                if cfg_wd is not None and sci_token(cfg_wd) != wd:
                    raise RuntimeError(f"Unexpected wd in {run.name}: config={cfg_wd}, expected={wd}")

                raw_rows.append(row)
                sweep_rows.append(row)

        tables.append(make_table(spec, sweep_rows))

    csv_path = OUT_DIR / "projector_sweep_r1_agg_false_raw.csv"
    tex_path = OUT_DIR / "projector_section.tex"
    meta_path = OUT_DIR / "README.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    section_intro = r"""\subsubsection{Projector}

	In all of the following hyperparameter searches, we train with a 100\% correction policy. The reported values are Recall@1 (\%) at inference with unaggregated SEEK codes. The optimal hyperparameters are not necessarily consistent across inference correction policies: runs that perform best with high correction can underperform when no correction is available, which is expected because the training distribution is biased toward fully corrected SEEK codes.

"""
    tex_path.write_text(section_intro + "\n".join(tables))

    meta_path.write_text(
        "# Projector Sweep Tables\n\n"
        f"Exported from W&B at `{export_time}`.\n\n"
        f"- Project: `{PROJECT_PATH}`\n"
        "- Source sweep definitions: `bash/projector.sh`, SWEEP 1 to SWEEP 4\n"
        "- Metric: `Pr/Test Recall@1 ... correction` with legacy false-aggregation fallback\n"
        "- Inference correction columns: 0%, 50%, 100%, ORACLE\n"
        "- Run selection: latest finished run for each expected W&B run name\n\n"
        "Generated files:\n\n"
        "- `projector_section.tex`: LaTeX section and tables\n"
        "- `projector_sweep_r1_agg_false_raw.csv`: raw W&B values and run metadata\n"
    )

    print(f"Wrote {tex_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
