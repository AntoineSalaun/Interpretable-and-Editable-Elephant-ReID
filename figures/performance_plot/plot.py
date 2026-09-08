import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# House style ----------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from figures.style import (
    set_style, get_figsize, savefig, get_color, format_metric_name, LABELS,
)
set_style()

SEEK_PATH  = "figures/performance_plot/SEEK_vs_cor.csv"
CHAIR_PATH = "figures/performance_plot/CHAIR_vs_cor.csv"
ELEPHANTBOOK_PATH = "figures/performance_plot/ELEPHANTBOOK_vs_cor.csv"
MIEWID_PATH = "figures/performance_plot/MIEWID_vs_cor.csv"
OUT_NAME   = "performance.pdf"

TRIPLE_FINETUNED_R1 = 19.65  # constant baseline line (%)
MEGA_FINETUNED_R1 = 1.95  # constant baseline line (%)
MIEWID_FINETUNED_R1 = 1.70  # constant baseline line (%)
VISUAL_BASELINE_COLOR = get_color(0)
SEEK_ONLY_COLOR = get_color(1)
SEEK_CBM_COLOR = get_color(2)
SEEK_CBM_IMAGE_COLOR = get_color(3)
ELEPHANTBOOK_COLOR = "#4D4D4D"
ELEPHANTBOOK_TRIPLE_COLOR = ELEPHANTBOOK_COLOR
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 22
Y_AXIS_LABEL_FONTSIZE = 20
X_TICKS = np.array([0, 25, 50, 75, 100])
Y_TICKS = np.array([0, 20, 40, 60, 80])
Y_LIMIT = 80
MARKERSIZE = 8
TICK_WIDTH = 16
SCOPE_ORDER = ["sighting", "image"]
SCOPE_LABELS = {
    "sighting": "SEEK-CBM, sighting-level training",
    "image": "SEEK-CBM, image-level training",
}
SEEK_SCOPE_LABELS = {
    "sighting": "SEEK-only, sighting gallery",
    "image": "SEEK-only, image gallery",
}
ELEPHANTBOOK_SCOPE_LABELS = {
    "sighting": "ElephantBook, sighting gallery",
    "image": "ElephantBook, image gallery",
}
ELEPHANTBOOK_VARIANT_ORDER = ["megadescriptor", "triple_crop_encoder"]
ELEPHANTBOOK_VARIANT_LABELS = {
    "megadescriptor": "ElephantBook (MD)",
    "triple_crop_encoder": "ElephantBook (triple)",
}
ELEPHANTBOOK_VARIANT_COLORS = {
    "megadescriptor": ELEPHANTBOOK_COLOR,
    "triple_crop_encoder": ELEPHANTBOOK_TRIPLE_COLOR,
}
ELEPHANTBOOK_VARIANT_MARKERS = {
    "megadescriptor": "D",
    "triple_crop_encoder": "X",
}


# ----------------------------
# helpers
# ----------------------------
def parse_pct_from_correction_policy(s) -> float:
    """Extracts the XX.X from older text summaries."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    m = re.search(r"with\s+(\d+(?:\.\d+)?)%\s+of\s+(?:sightings|elephants)\s+corrected", s)
    if m:
        return float(m.group(1))
    # fallback: first percent occurrence
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_pct_from_gallery_correction_fn(s) -> float:
    """Extracts 0.9 from 'sighting_0.9' and returns 90.0."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    if "oracle" in s.lower():
        return 110.0
    m = re.search(r"_(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1)) * 100.0
    return np.nan


def parse_training_scope(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if s.startswith("sighting_"):
        return "sighting"
    if s.startswith("image_"):
        return "image"
    return None


def parse_gallery_scope(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if s.startswith("sighting_"):
        return "sighting"
    if s.startswith("image_"):
        return "image"
    return None


def pick_seek_ycol(df) -> str:
    """SEEK file has Test-Recall@1 populated (in your exports)."""
    for c in ["Test-Recall@1", "Test Recall@1"]:
        if c in df.columns and df[c].notna().any():
            return c
    raise RuntimeError("Couldn't find a populated SEEK Recall@1 column (expected 'Test-Recall@1').")


def build_chair_colmap(df):
    """
    Map {percent(float) -> column_name} for CHAIR diagonal columns:
    'Pr/Test Recall@1 with XX% Image Correction correction'

    The older generic 'XX% Correction' metric used sighting-gallery/image-query
    inference in some exports. Prefer the explicit image/image metric, then keep
    the generic metric only as a fallback for historical CSVs.
    """
    patterns = [
        re.compile(r"^Pr/Test Recall@1 with (\d+(?:\.\d+)?)% Image Correction correction$"),
        re.compile(
            r"^Pr/Test Recall@1 with (\d+(?:\.\d+)?)% Correction correction"
            r"(?: aggregate_gallery_seeks=False)?$"
        ),
    ]
    out = {}
    for pat in patterns:
        for c in df.columns:
            m = pat.match(c)
            if m:
                out.setdefault(float(m.group(1)), c)
    return out


def closest_key(keys, target):
    keys = np.array(sorted(keys), dtype=float)
    return float(keys[np.argmin(np.abs(keys - float(target)))])


def infer_elephantbook_variant(df: pd.DataFrame) -> pd.Series:
    if "elephantbook_variant" in df.columns:
        return df["elephantbook_variant"].fillna("megadescriptor").astype(str)

    pretraining = (
        df["backbone_pretraining"].fillna("").astype(str)
        if "backbone_pretraining" in df.columns
        else pd.Series("", index=df.index)
    )
    with_ears = (
        df["elephantbook_visual_with_ears"]
        if "elephantbook_visual_with_ears" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    with_ears_bool = with_ears.map(
        lambda value: True
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes"}
    )
    is_triple = pretraining.str.contains("backbone", case=False, na=False) | with_ears_bool
    return pd.Series(np.where(is_triple, "triple_crop_encoder", "megadescriptor"), index=df.index)


def elephantbook_label(variant: str, scope: str, scope_filter: str | None) -> str:
    base = ELEPHANTBOOK_VARIANT_LABELS.get(variant, f"ElephantBook ({variant})")
    if scope_filter is not None:
        return base
    return f"{base}, {scope} gallery"


# ----------------------------
# LOAD + SEEK curve (already diagonal)
# ----------------------------
seek = pd.read_csv(SEEK_PATH)
seek_ycol = pick_seek_ycol(seek)

seek_corr = (
    pd.to_numeric(seek["correction_pct"], errors="coerce")
    if "correction_pct" in seek.columns
    else pd.Series(np.nan, index=seek.index)
)

if "gallery_correction_fn" in seek.columns:
    seek_corr = seek_corr.fillna(seek["gallery_correction_fn"].apply(parse_pct_from_gallery_correction_fn))

seek_scope = (
    seek["gallery_correction_scope"].astype(str).str.lower()
    if "gallery_correction_scope" in seek.columns
    else pd.Series(np.nan, index=seek.index)
)

if "gallery_correction_fn" in seek.columns:
    seek_scope = seek_scope.fillna(seek["gallery_correction_fn"].apply(parse_gallery_scope))

if "query_correction_fn" in seek.columns:
    query_pct = seek["query_correction_fn"].apply(parse_pct_from_gallery_correction_fn)
    seek_corr = seek_corr.where((query_pct.isna()) | (np.isclose(query_pct, seek_corr)), np.nan)

seek_curve = (
    pd.DataFrame({
        "gallery_correction_scope": seek_scope.fillna("sighting"),
        "correction_pct": seek_corr,
        "seek_r1": seek[seek_ycol],
    })
      .dropna(subset=["correction_pct", "seek_r1"])
      .groupby(["gallery_correction_scope", "correction_pct"], as_index=False)
      .mean()
      .sort_values("correction_pct")
)

legacy_seek_curve = (
    pd.DataFrame({"correction_pct": seek_corr, "seek_r1": seek[seek_ycol]})
      .dropna(subset=["correction_pct", "seek_r1"])
      .groupby("correction_pct", as_index=False)
      .mean()
      .sort_values("correction_pct")
)

if seek_curve.empty and not legacy_seek_curve.empty:
    seek_curve = legacy_seek_curve.assign(gallery_correction_scope="sighting")

if "correction_policy" in seek.columns and seek_curve.empty:
    seek_corr = seek["correction_policy"].apply(parse_pct_from_correction_policy)
    seek_curve = (
        pd.DataFrame({"gallery_correction_scope": "sighting", "correction_pct": seek_corr, "seek_r1": seek[seek_ycol]})
          .dropna(subset=["correction_pct", "seek_r1"])
          .groupby(["gallery_correction_scope", "correction_pct"], as_index=False)
          .mean()
          .sort_values("correction_pct")
    )

# ----------------------------
# LOAD + ElephantBook-style fusion curve (optional)
# ----------------------------
if os.path.exists(ELEPHANTBOOK_PATH):
    elephantbook = pd.read_csv(ELEPHANTBOOK_PATH)
    if elephantbook.empty:
        elephantbook_curve = pd.DataFrame(columns=["elephantbook_variant", "gallery_correction_scope", "correction_pct", "elephantbook_r1"])
    else:
        elephantbook_ycol = pick_seek_ycol(elephantbook)
        elephantbook_variant = infer_elephantbook_variant(elephantbook)
        elephantbook_corr = (
            pd.to_numeric(elephantbook["correction_pct"], errors="coerce")
            if "correction_pct" in elephantbook.columns
            else pd.Series(np.nan, index=elephantbook.index)
        )
        if "gallery_correction_fn" in elephantbook.columns:
            elephantbook_corr = elephantbook_corr.fillna(elephantbook["gallery_correction_fn"].apply(parse_pct_from_gallery_correction_fn))

        elephantbook_scope = (
            elephantbook["gallery_correction_scope"].astype(str).str.lower()
            if "gallery_correction_scope" in elephantbook.columns
            else pd.Series(np.nan, index=elephantbook.index)
        )
        if "gallery_correction_fn" in elephantbook.columns:
            elephantbook_scope = elephantbook_scope.fillna(elephantbook["gallery_correction_fn"].apply(parse_gallery_scope))

        elephantbook_curve = (
            pd.DataFrame({
                "elephantbook_variant": elephantbook_variant,
                "gallery_correction_scope": elephantbook_scope.fillna("sighting"),
                "correction_pct": elephantbook_corr,
                "elephantbook_r1": elephantbook[elephantbook_ycol],
            })
              .dropna(subset=["correction_pct", "elephantbook_r1"])
              .groupby(["elephantbook_variant", "gallery_correction_scope", "correction_pct"], as_index=False)
              .mean()
              .sort_values("correction_pct")
        )
else:
    elephantbook_curve = pd.DataFrame(columns=["elephantbook_variant", "gallery_correction_scope", "correction_pct", "elephantbook_r1"])

# ----------------------------
# LOAD + MiewID visual baselines (optional)
# ----------------------------
if os.path.exists(MIEWID_PATH):
    miewid = pd.read_csv(MIEWID_PATH)
    if miewid.empty or "Test-Recall@1" not in miewid.columns:
        miewid_curve = pd.DataFrame(columns=["miewid_mode", "miewid_r1"])
    else:
        miewid_curve = (
            miewid.assign(miewid_r1=pd.to_numeric(miewid["Test-Recall@1"], errors="coerce"))
            .dropna(subset=["miewid_mode", "miewid_r1"])
            .drop_duplicates(subset=["miewid_mode"], keep="first")
            [["miewid_mode", "miewid_r1"]]
        )
else:
    miewid_curve = pd.DataFrame(columns=["miewid_mode", "miewid_r1"])

# ----------------------------
# LOAD + CHAIR curve (diagonal: training correction == inference correction)
# ----------------------------
chair = pd.read_csv(CHAIR_PATH)

chair_train = pd.Series(np.nan, index=chair.index)
if "train_correction_pct" in chair.columns:
    chair_train = pd.to_numeric(chair["train_correction_pct"], errors="coerce")
if "correction_at_training" in chair.columns:
    chair_train = chair_train.fillna(chair["correction_at_training"].apply(parse_pct_from_gallery_correction_fn))
if "correction_policy" in chair.columns:
    chair_train = chair_train.fillna(chair["correction_policy"].apply(parse_pct_from_correction_policy))

chair = chair.assign(train_correction_pct=chair_train)
chair = chair.dropna(subset=["train_correction_pct"]).copy()
chair = chair[chair["train_correction_pct"] <= 100].copy()

if "training_correction_scope" in chair.columns:
    chair["training_correction_scope"] = chair["training_correction_scope"].astype(str).str.lower()
elif "correction_at_training" in chair.columns:
    chair["training_correction_scope"] = chair["correction_at_training"].apply(parse_training_scope)
else:
    chair["training_correction_scope"] = "sighting"
chair["training_correction_scope"] = chair["training_correction_scope"].fillna("sighting")

chair_map = build_chair_colmap(chair)
if not chair_map:
    raise RuntimeError("Couldn't find CHAIR columns of the form "
                       "'Pr/Test Recall@1 with XX% Correction correction'.")

rows = []
for _, r in chair.iterrows():
    t = float(r["train_correction_pct"])
    k = closest_key(chair_map.keys(), t)
    rows.append({
        "correction_pct": t,
        "training_correction_scope": r["training_correction_scope"],
        "chair_r1": r[chair_map[k]],
    })

chair_curve = (
    pd.DataFrame(rows)
      .dropna(subset=["correction_pct", "chair_r1"])
      .groupby(["training_correction_scope", "correction_pct"], as_index=False)
      .mean()
      .sort_values("correction_pct")
)

def plot_performance(out_name: str, scope_filter: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    baseline_x = np.array([0, 100])
    ax.plot(
        baseline_x,
        [MEGA_FINETUNED_R1, MEGA_FINETUNED_R1],
        linestyle=(0, (1, 1.4)),
        color=VISUAL_BASELINE_COLOR,
        label="MegaDescriptor",
        linewidth=2.0,
    )
    ax.plot(
        baseline_x,
        [MIEWID_FINETUNED_R1, MIEWID_FINETUNED_R1],
        linestyle="-",
        color=VISUAL_BASELINE_COLOR,
        label="MiewID",
        linewidth=2.0,
    )
    ax.plot(
        baseline_x,
        [TRIPLE_FINETUNED_R1, TRIPLE_FINETUNED_R1],
        linestyle="--",
        color=VISUAL_BASELINE_COLOR,
        label="Triple-crop encoder",
        linewidth=1.5,
    )

    plot_seek = seek_curve.copy()
    plot_chair = chair_curve.copy()
    plot_elephantbook = elephantbook_curve.copy()
    if scope_filter is not None:
        plot_seek = plot_seek[plot_seek["gallery_correction_scope"] == scope_filter]
        plot_chair = plot_chair[plot_chair["training_correction_scope"] == scope_filter]
        plot_elephantbook = plot_elephantbook[plot_elephantbook["gallery_correction_scope"] == scope_filter]

    seek_scopes = [s for s in SCOPE_ORDER if s in set(plot_seek["gallery_correction_scope"])]
    seek_scopes += sorted(set(plot_seek["gallery_correction_scope"]) - set(seek_scopes))
    for i, scope in enumerate(seek_scopes):
        scope_curve = plot_seek[plot_seek["gallery_correction_scope"] == scope].sort_values("correction_pct")
        label = "SEEK-only" if scope_filter is not None else SEEK_SCOPE_LABELS.get(scope, f"SEEK-only, {scope} gallery")
        ax.plot(
            scope_curve["correction_pct"],
            scope_curve["seek_r1"],
            marker="s",
            linestyle=["-", "--"][i % 2],
            color=SEEK_ONLY_COLOR,
            label=label,
            markersize=MARKERSIZE,
        )

    elephantbook_variants = [v for v in ELEPHANTBOOK_VARIANT_ORDER if v in set(plot_elephantbook["elephantbook_variant"])]
    elephantbook_variants += sorted(set(plot_elephantbook["elephantbook_variant"]) - set(elephantbook_variants))
    for variant in elephantbook_variants:
        variant_curve = plot_elephantbook[plot_elephantbook["elephantbook_variant"] == variant]
        elephantbook_scopes = [s for s in SCOPE_ORDER if s in set(variant_curve["gallery_correction_scope"])]
        elephantbook_scopes += sorted(set(variant_curve["gallery_correction_scope"]) - set(elephantbook_scopes))
        for i, scope in enumerate(elephantbook_scopes):
            scope_curve = variant_curve[variant_curve["gallery_correction_scope"] == scope].sort_values("correction_pct")
            ax.plot(
                scope_curve["correction_pct"],
                scope_curve["elephantbook_r1"],
                marker=ELEPHANTBOOK_VARIANT_MARKERS.get(variant, "^"),
                linestyle=["-", "--"][i % 2],
                color=ELEPHANTBOOK_VARIANT_COLORS.get(variant, ELEPHANTBOOK_COLOR),
                label=elephantbook_label(variant, scope, scope_filter),
                markersize=MARKERSIZE,
            )

    chair_scopes = [s for s in SCOPE_ORDER if s in set(plot_chair["training_correction_scope"])]
    chair_scopes += sorted(set(plot_chair["training_correction_scope"]) - set(chair_scopes))
    for i, scope in enumerate(chair_scopes):
        scope_curve = plot_chair[plot_chair["training_correction_scope"] == scope].sort_values("correction_pct")
        label = "SEEK-CBM" if scope_filter is not None else SCOPE_LABELS.get(scope, f"SEEK-CBM, {scope} training")
        ax.plot(
            scope_curve["correction_pct"],
            scope_curve["chair_r1"],
            marker="o",
            color=[SEEK_CBM_COLOR, SEEK_CBM_IMAGE_COLOR][i % 2],
            label=label,
            markersize=MARKERSIZE,
        )

    ax.set_xlabel(format_metric_name("correction_pct"), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Retrieval recall@1 (%)", fontsize=Y_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-2, 102)
    ax.set_xticks(X_TICKS)
    chair_max = plot_chair["chair_r1"].max() if not plot_chair.empty else 0
    seek_max = plot_seek["seek_r1"].max() if not plot_seek.empty else 0
    elephantbook_max = plot_elephantbook["elephantbook_r1"].max() if not plot_elephantbook.empty else 0
    miewid_max = miewid_curve["miewid_r1"].max() if not miewid_curve.empty else 0
    ax.set_ylim(0, max(Y_LIMIT, np.ceil(max(seek_max, chair_max, elephantbook_max, miewid_max))))
    ax.set_yticks(Y_TICKS)
    ax.tick_params(axis="both", labelsize=TICK_WIDTH)
    handles, labels = ax.get_legend_handles_labels()
    triple_idx = [i for i, label in enumerate(labels) if "ElephantBook (triple)" in label]
    if triple_idx and len(labels) > 1:
        first_triple = triple_idx[0]
        handle = handles.pop(first_triple)
        label = labels.pop(first_triple)
        insert_at = max(len(labels) - 1, 0)
        handles.insert(insert_at, handle)
        labels.insert(insert_at, label)

    ax.legend(
        handles,
        labels,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.11),
        borderaxespad=0.0,
        framealpha=0.85,
    )
    fig.tight_layout(pad=0.3)

    out_path = os.path.join(os.path.dirname(SEEK_PATH), out_name)
    savefig(fig, out_path)


plot_performance(OUT_NAME)
plot_performance("performance_sighting.pdf", "sighting")
plot_performance("performance_image.pdf", "image")
