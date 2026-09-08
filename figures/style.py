"""
House-style module for Methods in Ecology & Evolution figures.

Design principles
-----------------
* **One import, one call**: ``from figures.style import *; set_style()``
  gives you publication-ready rcParams everywhere.
* **Colorblind-safe** palette with huez; light blue + orange as primary pair.
* **Vector PDF** output by default (tight bbox, small padding, 300 dpi raster
  fallback for embedded bitmaps).

Sizing reference (MEE)
----------------------
Single-column width  ≈  84 mm  →  3.31 in
Double-column width  ≈ 174 mm  →  6.85 in
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# Module-level flag – set by set_style(); scripts can check it.
_USE_TEX: bool = False

# ---------------------------------------------------------------------------
# Huez integration – provides colorblind-checked palettes
# ---------------------------------------------------------------------------
try:
    import huez as _huez

    # Activate the "lancet" scheme (clean, journal-friendly).
    # Suppress Altair adapter warning that fires on some Python 3.13 builds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _huez.use("lancet")
    _HUEZ_OK = True
except Exception:
    _HUEZ_OK = False

# ---------------------------------------------------------------------------
# Colour system
# ---------------------------------------------------------------------------
# Primary pair: light blue + light orange (colorblind-safe, in line with the
# default request).  The rest of the palette extends to 8 distinguishable
# colours for categorical data.  Each colour is a hex string.
BLUE = "#5B9BD5"
ORANGE = "#ED7D31"

# Extended palette — first two entries are the primary pair.
# Built from a mix of Tol's Bright + custom picks; verified with huez check.
PALETTE: list[str] = [
    BLUE,       # 0 – blue  (primary)
    ORANGE,     # 1 – orange (primary)
    "#70AD47",  # 2 – green
    "#C44E52",  # 3 – red
    "#8E6EA6",  # 4 – purple
    "#FFD966",  # 5 – gold / yellow
    "#00B0F0",  # 6 – cyan
    "#A5A5A5",  # 7 – grey
]


def get_color(i: int) -> str:
    """Return the *i*-th palette colour, wrapping if *i* ≥ len(PALETTE)."""
    return PALETTE[i % len(PALETTE)]


def categorical_cmap(n: int) -> list[str]:
    """Return *n* distinct colours.

    For n ≤ len(PALETTE) the fixed palette is used.  For larger *n* huez
    generates a perceptually-spaced set, falling back to matplotlib's "tab20"
    if huez is unavailable.
    """
    if n <= len(PALETTE):
        return PALETTE[:n]
    if _HUEZ_OK:
        try:
            return _huez.get_colors(n)
        except Exception:
            pass
    # Fallback: matplotlib tab20
    cm = plt.colormaps.get_cmap("tab20")
    import numpy as np
    return [mpl.colors.rgb2hex(cm(i / max(n - 1, 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Label / metric name helpers
# ---------------------------------------------------------------------------
# Two versions: LaTeX-safe (\%) and plain (%).
_LABELS_TEX: dict[str, str] = {
    "r1": r"Recall@1 (\%)",
    "r5": r"Recall@5 (\%)",
    "r10": r"Recall@10 (\%)",
    "gallery_cor": r"Gallery correction (\%)",
    "query_cor": r"Query correction (\%)",
    "correction_pct": r"Correction (\%)",
    "alpha": r"$\alpha^*$",
    "train_cor": r"Training correction (\%)",
    "eval_cor": r"Inference correction (\%)",
}

_LABELS_PLAIN: dict[str, str] = {
    "r1": "Recall@1 (%)",
    "r5": "Recall@5 (%)",
    "r10": "Recall@10 (%)",
    "gallery_cor": "Gallery correction (%)",
    "query_cor": "Query correction (%)",
    "correction_pct": "Correction (%)",
    "alpha": r"$\alpha^*$",  # mathtext handles $...$ fine
    "train_cor": "Training correction (%)",
    "eval_cor": "Inference correction (%)",
}

# Public alias – updated by set_style() to match the active renderer.
LABELS: dict[str, str] = _LABELS_PLAIN


def format_metric_name(key: str) -> str:
    """Map a short metric key to a formatted axis label.

    Falls back to *key* with underscores replaced by spaces and title-cased
    if no explicit mapping exists.
    """
    if key in LABELS:
        return LABELS[key]
    return key.replace("_", " ").title()


def pct(value: float) -> str:
    """Format *value* with a percent sign safe for the active renderer."""
    sym = r"\%" if _USE_TEX else "%"
    return f"{value:.1f}{sym}"


# ---------------------------------------------------------------------------
# Figure sizing
# ---------------------------------------------------------------------------
# Methods in Ecology & Evolution column dimensions (inches).
_SINGLE_COL = 3.31   # 84 mm
_DOUBLE_COL = 6.85   # 174 mm
_GOLDEN = (1 + 5 ** 0.5) / 2  # ≈ 1.618

_FIGSIZE_MAP: dict[str, tuple[float, float]] = {
    "single":  (_SINGLE_COL, _SINGLE_COL / _GOLDEN),        # ~3.31 × 2.05
    "single_square": (_SINGLE_COL, _SINGLE_COL),            # 3.31 × 3.31
    "single_tall":   (_SINGLE_COL, _SINGLE_COL * _GOLDEN),  # 3.31 × 5.35
    "double":  (_DOUBLE_COL, _DOUBLE_COL / _GOLDEN),        # ~6.85 × 4.23
    "double_square": (_DOUBLE_COL, _DOUBLE_COL),
    "double_wide":   (_DOUBLE_COL, _DOUBLE_COL / 2.2),      # wider, squatter
    "heatmap": (_DOUBLE_COL, _DOUBLE_COL * 0.65),           # good for heatmaps
}


def get_figsize(kind: str = "single") -> tuple[float, float]:
    """Return *(width, height)* in inches for the requested layout.

    Parameters
    ----------
    kind : str
        One of ``"single"``, ``"single_square"``, ``"single_tall"``,
        ``"double"``, ``"double_square"``, ``"double_wide"``, ``"heatmap"``.
    """
    if kind not in _FIGSIZE_MAP:
        raise ValueError(
            f"Unknown figsize kind {kind!r}. Choose from: "
            + ", ".join(sorted(_FIGSIZE_MAP))
        )
    return _FIGSIZE_MAP[kind]


# ---------------------------------------------------------------------------
# rcParams – the core style setter
# ---------------------------------------------------------------------------
def _latex_available() -> bool:
    """Return True if a working ``latex`` binary is on PATH.

    Also checks common user-space TeX Live installations (e.g. ~/texlive)
    and prepends to ``$PATH`` if found so matplotlib can invoke it.
    """
    if shutil.which("latex") is not None:
        return True
    # Probe user-space TeX Live installs (installed via install-tl).
    import os, glob
    home = Path.home()
    candidates = sorted(
        glob.glob(str(home / "texlive" / "*" / "bin" / "*")),
        reverse=True,  # highest year first
    )
    for d in candidates:
        if (Path(d) / "latex").is_file():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def set_style(use_tex: bool = True, context: str = "paper") -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Parameters
    ----------
    use_tex : bool
        If *True* (default), render text with LaTeX **if available**;
        falls back to mathtext automatically when ``latex`` is not found.
    context : str
        ``"paper"`` (default) gives compact font sizes suitable for a
        two-column journal layout.  ``"talk"`` bumps everything up for
        slides / posters.
    """
    global _USE_TEX, LABELS  # noqa: PLW0603

    # Reset to defaults first so calls are idempotent.
    mpl.rcdefaults()

    # --- Font sizes (pt) ---------------------------------------------------
    if context == "paper":
        fs_title = 8
        fs_label = 7
        fs_tick = 6
        fs_legend = 6
    elif context == "talk":
        fs_title = 14
        fs_label = 12
        fs_tick = 10
        fs_legend = 10
    else:
        raise ValueError(f"Unknown context {context!r}; use 'paper' or 'talk'.")

    # --- TeX / mathtext (auto-detect) ---------------------------------------
    _USE_TEX = use_tex and _latex_available()
    LABELS = _LABELS_TEX if _USE_TEX else _LABELS_PLAIN

    if _USE_TEX:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"         # Latin Modern — matches most LaTeX docs
        )
        mpl.rcParams["font.family"] = "serif"
    else:
        if use_tex:
            warnings.warn(
                "LaTeX not found on PATH – falling back to mathtext. "
                "Install texlive-full for full LaTeX rendering.",
                stacklevel=2,
            )
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.fontset"] = "cm"  # Computer Modern look-alike
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["DejaVu Serif", "Bitstream Vera Serif",
                                       "Computer Modern Roman"]

    # --- Font sizes ---------------------------------------------------------
    mpl.rcParams["font.size"] = fs_label          # base (used by tick labels)
    mpl.rcParams["axes.titlesize"] = fs_title
    mpl.rcParams["axes.labelsize"] = fs_label
    mpl.rcParams["xtick.labelsize"] = fs_tick
    mpl.rcParams["ytick.labelsize"] = fs_tick
    mpl.rcParams["legend.fontsize"] = fs_legend
    mpl.rcParams["figure.titlesize"] = fs_title

    # --- Lines & markers ----------------------------------------------------
    mpl.rcParams["lines.linewidth"] = 1.0
    mpl.rcParams["lines.markersize"] = 3.5
    mpl.rcParams["lines.markeredgewidth"] = 0.0

    # --- Axes & spines ------------------------------------------------------
    mpl.rcParams["axes.linewidth"] = 0.6
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)

    # --- Ticks --------------------------------------------------------------
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["xtick.major.size"] = 3.0
    mpl.rcParams["ytick.major.size"] = 3.0
    mpl.rcParams["xtick.minor.visible"] = False
    mpl.rcParams["ytick.minor.visible"] = False
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"

    # --- Grid (off by default) ----------------------------------------------
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["grid.linewidth"] = 0.4
    mpl.rcParams["grid.alpha"] = 0.3

    # --- Legend -------------------------------------------------------------
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.loc"] = "best"
    mpl.rcParams["legend.handlelength"] = 1.5
    mpl.rcParams["legend.handletextpad"] = 0.4
    mpl.rcParams["legend.columnspacing"] = 1.0
    mpl.rcParams["legend.markerscale"] = 1.0
    mpl.rcParams["legend.borderaxespad"] = 0.3

    # --- Figure defaults ----------------------------------------------------
    mpl.rcParams["figure.figsize"] = get_figsize("single")
    mpl.rcParams["figure.dpi"] = 150           # screen preview
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["figure.autolayout"] = False  # we handle layout ourselves

    # --- Saving defaults ----------------------------------------------------
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["savefig.format"] = "pdf"
    mpl.rcParams["pdf.fonttype"] = 42          # TrueType → editable in Illustrator


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def savefig(
    fig: mpl.figure.Figure,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Save *fig* as a PDF with tight bounding box and publication metadata.

    Always saves as PDF regardless of the extension in *path* (the extension
    is forced to ``.pdf``).
    """
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        metadata={"Creator": "CBM_reid – figures/style.py"},
    )
    plt.close(fig)
    print(f"Saved → {path}")
