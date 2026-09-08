from pathlib import Path
from collections import defaultdict
import os
import sys
import re
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import seaborn as sns  # only for sns.heatmap – no theme

# House style ----------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from figures.style import (
    set_style, get_figsize, savefig, get_color, format_metric_name, LABELS,
)
set_style()

# ========= CONFIG =========
CSV_PATH = Path("figures/correction_policies/CHAIR_vs_cor.csv")   # <- your file
ORACLE_COLUMN_PATH = Path("figures/correction_policies/column_oracle.csv")
K = 1                                           # Recall@K
OUT_DPI = 200
LABEL_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 22
SPECIAL_CORRECTION_LABEL = "S"
INCLUDE_ORACLE = False
# ==========================


TRAIN_COL_CANDIDATES = [
    "correction_at_training",          # <-- in your CHAIR_vs_cor.csv
    "train_correction_pct",
    "training_correction_pct",
    "train_correction",
    "training_correction",
    "correction_at_training_pct",
]
SCOPE_COL_CANDIDATES = ["training_correction_scope", "train_correction_scope"]
SCOPE_ORDER = ["sighting", "image"]


def find_train_col(df: pd.DataFrame) -> str:
    for c in TRAIN_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Couldn't find a training correction column. Tried: {TRAIN_COL_CANDIDATES}\n"
        f"Available columns (first 50): {list(df.columns)[:50]}"
    )


def parse_train_pct(v) -> float:
    """
    Works with values like:
      - 'sighting_0.3' -> 30
      - 'oracle_correction' -> 110 (we'll label it ORACLE on the plot)
      - 0.3 -> 30
      - 30 or '30%' -> 30
    """
    s = str(v).strip()

    if s.lower().startswith(("sighting_", "image_", "elephant_")):
        try:
            return float(s.split("_", 1)[1]) * 100.0
        except Exception:
            pass

    if "oracle" in s.lower():
        return 110.0

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))

    try:
        x = float(s)
        return x * 100.0 if x <= 1.0 else x
    except Exception:
        return float("nan")


def parse_train_scope(v) -> str | None:
    s = str(v).strip().lower()
    if s.startswith("sighting_"):
        return "sighting"
    if s.startswith("image_"):
        return "image"
    return None


def add_training_scope(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCOPE_COL_CANDIDATES:
        if col in df.columns:
            df["_train_scope"] = df[col].astype(str).str.lower()
            return df
    if "correction_at_training" in df.columns:
        df["_train_scope"] = df["correction_at_training"].map(parse_train_scope)
    else:
        df["_train_scope"] = None
    return df


def available_training_scopes() -> list[str | None]:
    df = add_training_scope(pd.read_csv(CSV_PATH))
    scopes = [s for s in SCOPE_ORDER if s in set(df["_train_scope"].dropna())]
    return scopes or [None]


def parse_eval_correction(v: str) -> float:
    if v == "ORACLE":
        return 110.0
    return float(v)


def metric_preference(col: str) -> int:
    if "aggregate_gallery_seeks=False" in col:
        return 1
    if "aggregate_gallery_seeks=True" in col:
        return 2
    return 0


def coalesced_metric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    ordered = sorted(cols, key=metric_preference)
    return df[ordered].bfill(axis=1).iloc[:, 0]


def build_heatmap_data(training_scope: str | None = None) -> pd.DataFrame:
    df = add_training_scope(pd.read_csv(CSV_PATH))
    if training_scope is not None:
        df = df[df["_train_scope"] == training_scope].copy()
        if df.empty:
            return pd.DataFrame()

    train_col = find_train_col(df)
    df["_train_pct"] = df[train_col].map(parse_train_pct)

    image_metric_re = re.compile(
        rf"^(Pr/)?Test Recall@{K} with (\d+)% Image Correction correction$"
    )
    sighting_metric = f"Pr/Test Recall@{K} with 100% Sighting Correction correction"

    candidates = defaultdict(list)
    for col in df.columns:
        m = image_metric_re.match(col)
        if m:
            eval_pct = parse_eval_correction(m.group(2))
            candidates[eval_pct].append(col)
        elif col == sighting_metric:
            candidates[110.0].append(col)

    if not candidates:
        raise ValueError(
            "No metric columns matched. "
            "Expected image-inference metrics like "
            f"'Pr/Test Recall@{K} with 90% Image Correction correction'."
        )

    parts = []
    for eval_pct, cols in sorted(candidates.items()):
        parts.append(
            pd.DataFrame(
                {
                    "train_pct": df["_train_pct"],
                    "eval_pct": eval_pct,
                    "value": coalesced_metric(df, cols),
                }
            )
        )

    if INCLUDE_ORACLE:
        oracle_df = pd.read_csv(ORACLE_COLUMN_PATH)
        oracle_train_col = find_train_col(oracle_df)
        oracle_df["_train_pct"] = oracle_df[oracle_train_col].map(parse_train_pct)
        oracle_candidates = []
        for col in oracle_df.columns:
            if col == f"Pr/Test Recall@{K} with ORACLE correction":
                oracle_candidates.append(col)
        if oracle_candidates:
            parts.append(
                pd.DataFrame(
                    {
                        "train_pct": oracle_df["_train_pct"],
                        "eval_pct": 110.0,
                        "value": coalesced_metric(oracle_df, oracle_candidates),
                    }
                )
            )

    long = pd.concat(parts, ignore_index=True).dropna(subset=["train_pct", "value"])
    eval_order = [110.0] + [float(p) for p in range(100, -1, -10)]
    train_order = [float(p) for p in range(0, 101, 10)]
    heat = long.pivot_table(index="eval_pct", columns="train_pct", values="value", aggfunc="mean")
    heat = heat.reindex([p for p in eval_order if p in set(long["eval_pct"])])
    heat = heat.reindex([p for p in train_order if p in set(long["train_pct"])], axis=1)
    return heat


def plot_heatmap(
    heat: pd.DataFrame,
    *,
    annot: bool,
    show_cbar: bool,
    out_name: str,
    square: bool,
    figsize: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize or get_figsize("heatmap"))
    blues = plt.get_cmap("Blues")
    truncated_blues = colors.LinearSegmentedColormap.from_list(
        "truncated_blues", blues(np.linspace(0.06, 1.0, 256))
    )

    sns.heatmap(
        heat,
        annot=annot,
        fmt=".0f",
        linewidths=0.5,
        cmap=truncated_blues,
        cbar=show_cbar,
        square=square,
        cbar_kws={"label": format_metric_name("r1")} if show_cbar else None,
        annot_kws={"size": LABEL_FONT_SIZE},
        ax=ax,
    )

    ax.set_xlabel(format_metric_name("train_cor"), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(format_metric_name("eval_cor"), fontsize=AXIS_LABEL_FONT_SIZE)

    # Replace the special sighting correction label and visually separate it.
    ylabels = [(SPECIAL_CORRECTION_LABEL if y >= 110 else str(int(y))) for y in heat.index]
    xlabels = [(SPECIAL_CORRECTION_LABEL if x >= 110 else str(int(x))) for x in heat.columns]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=LABEL_FONT_SIZE)
    ax.set_xticklabels(xlabels, rotation=0, fontsize=LABEL_FONT_SIZE)

    if 110.0 in heat.index:
        s_row = list(heat.index).index(110.0) + 1
        ax.hlines(s_row, *ax.get_xlim(), colors="white", linewidth=6)
    if 110.0 in heat.columns:
        s_col = list(heat.columns).index(110.0)
        ax.vlines(s_col, *ax.get_ylim(), colors="white", linewidth=6)

    if annot:
        for row_idx, _ in enumerate(heat.index):
            for col_idx, _ in enumerate(heat.columns):
                text_idx = row_idx * len(heat.columns) + col_idx
                if text_idx < len(ax.texts):
                    ax.texts[text_idx].set_position((col_idx + 0.5, row_idx + 0.5))
                    ax.texts[text_idx].set_verticalalignment("center")
                    ax.texts[text_idx].set_horizontalalignment("center")

    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=LABEL_FONT_SIZE)
        cbar.set_label(format_metric_name("r1"), fontsize=LABEL_FONT_SIZE)

    out_path = CSV_PATH.parent / out_name
    savefig(fig, out_path)


def main():
    base_width, base_height = get_figsize("heatmap")
    balanced_annot_figsize = (base_width, 0.5 * (base_height + base_width))

    for scope in available_training_scopes():
        heat = build_heatmap_data(scope)
        if heat.empty:
            continue

        prefix = "heatmap" if scope is None else f"heatmap_{scope}"
        plot_heatmap(
            heat,
            annot=True,
            show_cbar=False,
            out_name=f"{prefix}_with_number.pdf",
            square=False,
            figsize=balanced_annot_figsize,
        )
        plot_heatmap(
            heat,
            annot=False,
            show_cbar=True,
            out_name=f"{prefix}_no_numbers.pdf",
            square=False,
        )
        shutil.copyfile(CSV_PATH.parent / f"{prefix}_no_numbers.pdf", CSV_PATH.parent / f"{prefix}.pdf")

        if scope in {None, "sighting"}:
            plot_heatmap(
                heat,
                annot=True,
                show_cbar=False,
                out_name="heatmap_with_number.pdf",
                square=False,
                figsize=balanced_annot_figsize,
            )
            plot_heatmap(
                heat,
                annot=False,
                show_cbar=True,
                out_name="heatmap_no_numbers.pdf",
                square=False,
            )
            shutil.copyfile(CSV_PATH.parent / "heatmap_no_numbers.pdf", CSV_PATH.parent / "heatmap.pdf")


if __name__ == "__main__":
    main()
