"""Export the original Mara dataset histograms for two manuscript minipages.

Run from any directory with ``python /path/to/figures/dataset_distribution/plot.py``.
Counts and bins reproduce cells 6 and 9 of mara_dataset_processing.ipynb.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from figures.style import set_style

OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO_ROOT / "data/mara/25-10-30_individual_sightings.tsv"

# Preserve the original notebook's blue, independently of the house palette.
HISTOGRAM_BLUE = "#1F77B4"
# Match figures/performance_plot/plot.py, including its effective tick size
# (TICK_WIDTH=16, rather than the unused TICK_FONTSIZE=12).
X_LABEL_FONTSIZE = 22
Y_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
FIGSIZE = (6, 4)


def load_distributions(source: Path) -> tuple[pd.Series, pd.Series]:
    sightings = pd.read_csv(source, sep="\t", comment="#", low_memory=False)
    # The original panel A includes all rows, including unidentified elephants.
    images_per_sighting = sightings["bounding_boxes"].apply(
        lambda cell: len(ast.literal_eval(cell))
    )
    # Panel B requires an identified elephant and counts distinct sightings.
    sightings_per_elephant = (
        sightings.dropna(subset=["individual_id", "individual_sighting_id"])
        .groupby("individual_id")["individual_sighting_id"]
        .nunique()
    )
    return images_per_sighting, sightings_per_elephant


def plot_histogram(values, *, bins: int, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, _, _ = ax.hist(values, bins=bins, color=HISTOGRAM_BLUE)
    ax.set_yscale("log")
    # Leave count-one bars visible; empty bins remain empty on the log axis.
    ax.set_ylim(0.8, 10 ** np.ceil(np.log10(counts.max() * 1.1)))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout(pad=0.3)
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    images, sightings = load_distributions(args.source)

    # The existing performance PDFs use the house style's DejaVu Serif font.
    set_style(use_tex=False)
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    panels = [
        ("images_per_sighting", images, 160,
         "Images per individual sighting", "Number of sightings"),
        ("sightings_per_elephant", sightings, 40,
         "Sightings per elephant", "Number of elephants"),
    ]
    figures = [
        (name, *plot_histogram(values, bins=bins, xlabel=xlabel, ylabel=ylabel))
        for name, values, bins, xlabel, ylabel in panels
    ]
    # Equal page dimensions and axes positions keep both minipages aligned and
    # give them identical font sizes when included at the same LaTeX width.
    margins = {
        "left": max(fig.subplotpars.left for _, fig, _ in figures),
        "right": min(fig.subplotpars.right for _, fig, _ in figures),
        "bottom": max(fig.subplotpars.bottom for _, fig, _ in figures),
        "top": min(fig.subplotpars.top for _, fig, _ in figures),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig, _ in figures:
        fig.subplots_adjust(**margins)
        with plt.rc_context({"savefig.bbox": None}):
            for extension in ("pdf", "png"):
                path = args.output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=300, facecolor="white")
                print(f"Saved {path}")
        plt.close(fig)

    print(f"A: {len(images):,} sightings, {images.sum():,} images")
    print(f"B: {len(sightings):,} identified elephants, {sightings.sum():,} sightings")


if __name__ == "__main__":
    main()
