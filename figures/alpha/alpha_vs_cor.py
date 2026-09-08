#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# House style ----------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from figures.style import (
    set_style, get_figsize, savefig, get_color, format_metric_name, LABELS,
)
set_style()

TICK_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 22
X_TICKS = np.array([0, 25, 50, 75, 100])
MARKERSIZE = 8
SPECIAL_CORRECTION_LABEL = "S"
SCOPE_ORDER = ["sighting", "image"]
SCOPE_LABELS = {
    "sighting": "sighting-level correction",
    "image": "image-level correction",
}


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_float(v):
    """Extract first float from a value (handles strings like 'sighting_0.3', '30%', etc.)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip().lower()

    # handle special cases
    if "oracle" in s:
        return 1.0

    m = FLOAT_RE.search(s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except ValueError:
        return float("nan")


def parse_correction_pct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")

    s = str(v).strip().lower()
    if "oracle" in s:
        return 110.0

    x = extract_float(v)
    if pd.isna(x):
        return float("nan")
    return x * 100.0 if x <= 1.0 else x


def parse_training_scope(v):
    s = str(v).strip().lower()
    if s.startswith("sighting_"):
        return "sighting"
    if s.startswith("image_"):
        return "image"
    return None


def pick_best_numeric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the candidate column that yields the most numeric rows after parsing."""
    best_col, best_count = None, -1
    for c in candidates:
        if c not in df.columns:
            continue
        parsed = df[c].map(extract_float)
        count = int(parsed.notna().sum())
        if count > best_count:
            best_col, best_count = c, count
    return best_col


def pick_preferred_numeric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first candidate with any numeric values, preserving priority."""
    for c in candidates:
        if c not in df.columns:
            continue
        if df[c].map(extract_float).notna().any():
            return c
    return None


def main():
    # --- locate csv: same folder as script OR hardcode your path here ---
    # If you already pass a path in your script, keep that.
    csv_path = Path("figures/alpha/alpha_vs_cor.csv").resolve()  # <-- adjust if needed
    # Alternatively: csv_path = Path("CHAIR_vs_cor.csv").resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Prefer the alpha from the best validation checkpoint.
    # Pr/Alpha is the last logged alpha, and Projector/alpha is the initial config.
    alpha_candidates = [
        "Pr/Best Alpha",
        "learnt_alpha",
        "learned_alpha",
        "Pr/Alpha",
        "Projector/alpha",
        "alpha",
    ]
    alpha_col = pick_preferred_numeric_column(df, alpha_candidates)
    if alpha_col is None:
        raise ValueError(f"Could not find alpha column. Tried: {alpha_candidates}")

    # Gallery correction: prefer true gallery-correction columns, else fall back to correction_at_training
    x_candidates = [
        "train_correction_pct",
        "gallery_correction",
        "gallery_correction_pct",
        "correction_at_inference",
        "correction_at_eval",
        "correction",
        "correction_at_training",  # in your CSV, this is the useful one (e.g. sighting_0.3)
    ]
    x_col = pick_preferred_numeric_column(df, x_candidates)
    if x_col is None:
        raise ValueError(f"Could not find any usable x column among: {x_candidates}")

    x = df[x_col].map(parse_correction_pct)
    y = pd.to_numeric(df[alpha_col], errors="coerce")

    if "training_correction_scope" in df.columns:
        scope = df["training_correction_scope"].astype(str).str.lower()
    elif "correction_at_training" in df.columns:
        scope = df["correction_at_training"].map(parse_training_scope)
    else:
        scope = "correction"

    data = pd.DataFrame({"x": x, "alpha": y, "scope": scope}).dropna(subset=["x", "alpha"])
    if data.empty:
        raise ValueError(
            f"No valid numeric rows after parsing x='{x_col}' and alpha='{alpha_col}'."
        )

    data["scope"] = data["scope"].fillna("correction")
    scopes = [s for s in SCOPE_ORDER if s in set(data["scope"])]
    scopes += sorted(set(data["scope"]) - set(scopes))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, scope_name in enumerate(scopes):
        scope_data = (
            data[data["scope"] == scope_name]
            .drop_duplicates(subset="x", keep="first")
            .sort_values("x")
        )
        label = SCOPE_LABELS.get(scope_name, scope_name)
        ax.plot(
            scope_data["x"],
            scope_data["alpha"],
            marker="o",
            color=get_color(2 + i),
            markersize=MARKERSIZE,
            label=label,
        )

    ax.set_xlabel("Training correction (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"learnt fusion $\alpha^*$", fontsize=AXIS_LABEL_FONTSIZE)
    has_special = bool((data["x"] > 100).any())
    ax.set_xlim(-2, 112 if has_special else 102)
    ax.set_ylim(0, 0.6)
    xticks = list(X_TICKS) + ([110] if has_special else [])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(t)) for t in X_TICKS] + ([SPECIAL_CORRECTION_LABEL] if has_special else []))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if len(scopes) > 1:
        ax.legend(fontsize=12, loc="best")
    fig.tight_layout(pad=0.0)

    out_path = csv_path.with_name(f"alpha_vs_cor.pdf")
    savefig(fig, out_path)


if __name__ == "__main__":
    main()
