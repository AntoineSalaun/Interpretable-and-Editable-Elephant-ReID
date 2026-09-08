from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# House style ----------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from figures.style import set_style, savefig
# NOTE: We call set_style() but the tSNE panels use a custom discrete colormap
# and marker sizes that are swept, so we intentionally do NOT standardise those.
set_style()


# -----------------------------
# Args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", type=str, default="figures/tSNE")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Directory where sweep PDFs will be written. Defaults to pt_dir/sweeps")

    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--n_iter", type=int, default=1200)

    ap.add_argument("--topk_elephants", type=int, default=50)
    ap.add_argument("--max_per_elephant", type=int, default=500)
    ap.add_argument("--max_points_total", type=int, default=25000)

    ap.add_argument("--cmap_name", type=str, default="tab20")

    # Layout / style
    ap.add_argument("--fig_w", type=float, default=18.0)
    ap.add_argument("--fig_h", type=float, default=5.2)
    ap.add_argument("--spine_lw", type=float, default=1.2)
    ap.add_argument("--title_fs", type=float, default=15)

    return ap.parse_args()


def default_paths(args):
    if args.out_dir is None:
        args.out_dir = os.path.join(args.pt_dir, "sweeps")
    return args


# -----------------------------
# Loading
# -----------------------------
def load_pt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict) or "embeddings" not in d or "labels" not in d:
        raise ValueError(f"{path} must contain dict keys: embeddings, labels")

    X = d["embeddings"].detach().cpu().float().numpy()
    y = d["labels"].detach().cpu().numpy().astype(np.int64)

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"Bad shapes in {path}: X {X.shape}, y {y.shape}")
    return X, y


# -----------------------------
# Subsetting helpers
# -----------------------------
def topk_ids(labels: np.ndarray, k: int) -> List[int]:
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    return [int(x) for x in ids[order[:k]]]


def pick_indices_for_ids(
    labels: np.ndarray,
    ids_keep: List[int],
    seed: int,
    max_per_id: int,
    max_total: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    idx = np.nonzero(np.isin(labels, np.array(ids_keep, dtype=np.int64)))[0]
    by_id: Dict[int, List[int]] = {}
    for i in idx:
        by_id.setdefault(int(labels[i]), []).append(int(i))

    chosen: List[int] = []
    for eid in ids_keep:
        inds = by_id.get(int(eid), [])
        if not inds:
            continue
        if max_per_id and max_per_id > 0 and len(inds) > max_per_id:
            inds = rng.choice(inds, size=max_per_id, replace=False).tolist()
        chosen.extend(inds)

    chosen = np.array(chosen, dtype=np.int64)
    if chosen.size > max_total:
        chosen = rng.choice(chosen, size=max_total, replace=False).astype(np.int64)

    return chosen


# -----------------------------
# Colors
# -----------------------------
def make_discrete_cmap(k: int, base_name: str) -> ListedColormap:
    base = plt.colormaps.get_cmap(base_name)
    colors = base(np.linspace(0, 1, k))
    return ListedColormap(colors)


def labels_to_index(labels: np.ndarray, ids_keep: List[int]) -> np.ndarray:
    lut = {eid: i for i, eid in enumerate(ids_keep)}
    return np.array([lut[int(x)] for x in labels], dtype=np.int64)


# -----------------------------
# t-SNE (sklearn only)
# -----------------------------
def tsne_2d(X: np.ndarray, seed: int, perplexity: float, n_iter: int) -> np.ndarray:
    Xr = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=n_iter,
        verbose=0,
    )
    return tsne.fit_transform(Xr)


# -----------------------------
# Plot one figure
# -----------------------------
def render_figure(
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ids_keep: List[int],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    args,
    perplexity: float,
    marker_size: float,
    alpha: float,
    out_path: str,
):
    panels = [
        ("25% correction",  "embeddings_25.pt"),
        ("50% correction",  "embeddings_50.pt"),
        ("75% correction",  "embeddings_75.pt"),
        ("100% correction", "embeddings_100.pt"),
    ]

    K = len(ids_keep)

    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    fig.patch.set_facecolor("white")

    # Manual layout: 1x4 + (optional) colorbar area
    left, right = 0.03, 0.995
    top, bottom = 0.93, 0.15   # a bit more room since we don't draw colorbar
    wspace = 0.018
    panel_w = (right - left - 3 * wspace) / 4
    panel_h = top - bottom

    axes = []
    for i in range(4):
        ax = fig.add_axes([left + i * (panel_w + wspace), bottom, panel_w, panel_h])
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(args.spine_lw)
        axes.append(ax)

    last_mappable = None

    for i, (title, _) in enumerate(panels):
        ax = axes[i]
        if title not in loaded:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=12)
            continue

        X, y = loaded[title]

        idx = pick_indices_for_ids(
            labels=y,
            ids_keep=ids_keep,
            seed=args.seed,
            max_per_id=args.max_per_elephant,
            max_total=args.max_points_total,
        )

        Xs = X[idx]
        ys = y[idx]
        ele_idx = labels_to_index(ys, ids_keep)

        Y2 = tsne_2d(Xs, seed=args.seed + i, perplexity=perplexity, n_iter=args.n_iter)

        last_mappable = ax.scatter(
            Y2[:, 0], Y2[:, 1],
            s=marker_size,
            alpha=alpha,
            c=ele_idx,
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )

    # -----------------------------
    # COLORBAR: how to remove it
    # -----------------------------
    # To have NO colorbar, do nothing here (this block stays commented / absent).
    #
    # If you ever want it back, uncomment this entire block and add a cax axis.
    #
    # if last_mappable is not None:
    #     cax = fig.add_axes([0.10, 0.05, 0.80, 0.08])
    #     cb = fig.colorbar(last_mappable, cax=cax, orientation="horizontal")
    #     ticks = np.arange(K)
    #     cb.set_ticks(ticks)
    #     cb.set_ticklabels([str(t + 1) for t in ticks])
    #     cb.set_label("")
    # -----------------------------

    fig.savefig(out_path, format="pdf", bbox_inches="tight",
                pad_inches=0.02, facecolor="white", dpi=300)
    plt.close(fig)


# -----------------------------
# Main (sweep)
# -----------------------------
def main():
    args = default_paths(parse_args())
    os.makedirs(args.pt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    panels = [
        ("25% correction",  "embeddings_25.pt"),
        ("50% correction",  "embeddings_50.pt"),
        ("75% correction",  "embeddings_75.pt"),
        ("100% correction", "embeddings_100.pt"),
    ]

    loaded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for title, fname in panels:
        fpath = os.path.join(args.pt_dir, fname)
        if os.path.exists(fpath):
            loaded[title] = load_pt(fpath)
        else:
            print(f"[warn] missing {fpath}")

    if not loaded:
        raise SystemExit("No panel .pt files found in pt_dir.")

    # Reference for top-K selection: prefer 25%
    ref_title = "25% correction" if "25% correction" in loaded else next(iter(loaded.keys()))
    _, ref_y = loaded[ref_title]
    ids_keep = topk_ids(ref_y, args.topk_elephants)
    K = len(ids_keep)

    cmap = make_discrete_cmap(K, args.cmap_name)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, K + 0.5, 1), ncolors=K)

    # -----------------------------
    # Sweep space (wide but sane)
    # -----------------------------
    marker_sizes = [6, 10, 14, 20, 28, 40]
    perplexities = [10, 20, 25, 30, 40, 50, 70]
    alphas = [0.35, 0.55, 0.70, 0.85, 0.95]

    total = len(marker_sizes) * len(perplexities) * len(alphas)
    print(f"[info] Will render {total} figures into: {args.out_dir}")

    for ms in marker_sizes:
        for perp in perplexities:
            for a in alphas:
                # Safe-ish filename, easy to sort/grep
                out_name = f"tSNE.pdf"
                out_path = os.path.join(args.out_dir, out_name)

                print(f"[render] perp={perp:>3}  ms={ms:>2}  alpha={a:.2f}  -> {out_name}")
                render_figure(
                    loaded=loaded,
                    ids_keep=ids_keep,
                    cmap=cmap,
                    norm=norm,
                    args=args,
                    perplexity=float(perp),
                    marker_size=float(ms),
                    alpha=float(a),
                    out_path=out_path,
                )

    print("[done] sweep finished")


if __name__ == "__main__":
    main()
