from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DATA_DIR = THIS_DIR / "data"

EMBEDDINGS_25_PATH = REPO_ROOT / "figures" / "tSNE" / "embeddings_25.pt"
MARA_DICT_PATH = REPO_ROOT / "data" / "mara" / "image_dictionary_optimized.csv"
TRAIN_INDICES_PATH = REPO_ROOT / "weights" / "train_indices_gold.txt"
ORIGINAL_IMAGE_ROOT = Path("/archive/vision/beery/animal_reid/datasets/elephants_new/images")

CORRECTION_SPECS = [
    {
        "code": "00",
        "name": "0% correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_00.pt",
        "seed": 100,
    },
    {
        "code": "25",
        "name": "25% correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_25.pt",
        "seed": 101,
    },
    {
        "code": "50",
        "name": "50% correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_50.pt",
        "seed": 102,
    },
    {
        "code": "75",
        "name": "75% correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_75.pt",
        "seed": 103,
    },
    {
        "code": "100",
        "name": "100% correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_100.pt",
        "seed": 104,
    },
    {
        "code": "oracle",
        "name": "Oracle correction",
        "pt_path": REPO_ROOT / "figures" / "tSNE" / "embeddings_oracle.pt",
        "seed": 105,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached data already exists.")
    return parser.parse_args()


def load_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "embeddings" not in data or "labels" not in data:
        raise ValueError(f"{path} must contain dict keys: embeddings, labels")

    embeddings = data["embeddings"].detach().cpu().float().numpy()
    labels = data["labels"].detach().cpu().numpy().astype(np.int64)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatched shapes in {path}: {embeddings.shape} vs {labels.shape}")
    return embeddings, labels


def topk_ids(labels: np.ndarray, k: int) -> list[int]:
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    return [int(x) for x in ids[order[:k]]]


def pick_indices_for_ids(
    labels: np.ndarray,
    ids_keep: list[int],
    seed: int,
    max_per_id: int,
    max_total: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    idx = np.nonzero(np.isin(labels, np.array(ids_keep, dtype=np.int64)))[0]
    by_id: dict[int, list[int]] = {}
    for i in idx:
        by_id.setdefault(int(labels[i]), []).append(int(i))

    chosen: list[int] = []
    for label_id in ids_keep:
        indices = by_id.get(int(label_id), [])
        if not indices:
            continue
        if max_per_id > 0 and len(indices) > max_per_id:
            indices = rng.choice(indices, size=max_per_id, replace=False).tolist()
        chosen.extend(indices)

    chosen_array = np.array(chosen, dtype=np.int64)
    if chosen_array.size > max_total:
        chosen_array = rng.choice(chosen_array, size=max_total, replace=False).astype(np.int64)

    return chosen_array


def tsne_2d(embeddings: np.ndarray, seed: int, perplexity: float, n_iter: int) -> np.ndarray:
    reduced = PCA(n_components=min(50, embeddings.shape[1]), random_state=seed).fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=n_iter,
        verbose=0,
    )
    return tsne.fit_transform(reduced)


def load_filtered_train_metadata() -> pd.DataFrame:
    data_frame = pd.read_csv(MARA_DICT_PATH)

    encounter_counts = data_frame.groupby("ele_id")["encounter_id"].nunique()
    valid_elephants = encounter_counts[encounter_counts >= 2].index
    filtered = data_frame[data_frame["ele_id"].isin(valid_elephants)].reset_index(drop=True)

    ele_id_to_label = {ele_id: i for i, ele_id in enumerate(sorted(filtered["ele_id"].unique()))}

    with TRAIN_INDICES_PATH.open("r") as handle:
        train_indices = [int(line.strip()) for line in handle]

    train_frame = filtered.iloc[train_indices].copy().reset_index(drop=True)
    train_frame["label"] = train_frame["ele_id"].map(ele_id_to_label).astype(np.int64)
    train_frame["original_image_path"] = train_frame["image"].map(lambda rel: str(ORIGINAL_IMAGE_ROOT / rel))
    train_frame["logical_rank_within_label"] = train_frame.groupby("label").cumcount()
    return train_frame


def align_metadata_to_label_order(train_frame: pd.DataFrame, point_labels: np.ndarray) -> pd.DataFrame:
    label_counts_metadata = Counter(train_frame["label"].tolist())
    label_counts_points = Counter(point_labels.tolist())
    if label_counts_metadata != label_counts_points:
        raise ValueError("Per-label counts do not match between metadata and saved embeddings.")

    label_to_indices: dict[int, list[int]] = {
        int(label): group.index.tolist()
        for label, group in train_frame.groupby("label", sort=False)
    }

    positions: defaultdict[int, int] = defaultdict(int)
    aligned_indices: list[int] = []
    for label in point_labels.tolist():
        label = int(label)
        aligned_indices.append(label_to_indices[label][positions[label]])
        positions[label] += 1

    aligned = train_frame.iloc[aligned_indices].copy().reset_index(drop=True)
    if not np.array_equal(aligned["label"].to_numpy(dtype=np.int64), point_labels.astype(np.int64)):
        raise ValueError("Aligned metadata labels do not match saved embedding labels.")
    return aligned


def output_paths(code: str) -> tuple[Path, Path]:
    base_name = f"mara_embedding_{code}_top50"
    return DATA_DIR / f"{base_name}.parquet", DATA_DIR / f"{base_name}.build.json"


def build_dataframe(spec: dict, ids_keep: list[int]) -> tuple[pd.DataFrame, dict]:
    embeddings, labels = load_pt(spec["pt_path"])
    train_metadata = load_filtered_train_metadata()
    aligned_metadata = align_metadata_to_label_order(train_metadata, labels)
    chosen_indices = pick_indices_for_ids(
        labels=labels,
        ids_keep=ids_keep,
        seed=101,
        max_per_id=500,
        max_total=25000,
    )

    chosen_embeddings = embeddings[chosen_indices]
    projection = tsne_2d(chosen_embeddings, seed=spec["seed"], perplexity=10, n_iter=1200)

    selected = aligned_metadata.iloc[chosen_indices].copy().reset_index(drop=True)
    selected = selected.rename(
        columns={
            "subject-SEEK": "subject_seek",
            "ele-SEEK": "ele_seek",
            "picture_time": "picture_time_utc",
        }
    )

    rank_map = {int(label_id): rank for rank, label_id in enumerate(ids_keep)}
    selected["atlas_row_id"] = [f"pt_{i:05d}" for i in range(len(selected))]
    selected["pt_row_index"] = chosen_indices.astype(np.int64)
    selected["tsne_x"] = projection[:, 0]
    selected["tsne_y"] = projection[:, 1]
    selected["correction_code"] = spec["code"]
    selected["correction_name"] = spec["name"]
    selected["original_label"] = selected["label"].astype(np.int64)
    selected["label"] = selected["original_label"].map(rank_map).astype(np.int64)
    selected["original_ele_id"] = selected["ele_id"]
    selected["compact_elephant_id"] = selected["label"].map(lambda value: f"E{value:02d}")
    selected["ele_id"] = selected["label"].map(lambda value: f"C{value % 10:02d}")
    selected["projection_name"] = f"{spec['name']} / perplexity=10"
    selected["atlas_text"] = selected.apply(
        lambda row: (
            f"ele_color={row['ele_id']} | elephant={row['compact_elephant_id']} "
            f"(orig={row['original_ele_id']}) | label={row['label']} | "
            f"ele_SEEK={row['ele_seek']} | subject_SEEK={row['subject_seek']} | "
            f"subject_id={row['subject_id']}"
        ),
        axis=1,
    )
    selected["original_image_exists"] = selected["original_image_path"].map(lambda path: Path(path).exists())

    columns = [
        "atlas_row_id",
        "pt_row_index",
        "tsne_x",
        "tsne_y",
        "label",
        "ele_id",
        "compact_elephant_id",
        "original_label",
        "original_ele_id",
        "subject_id",
        "encounter_id",
        "ele_seek",
        "subject_seek",
        "image",
        "original_image_path",
        "original_image_exists",
        "logical_rank_within_label",
        "picture_time_utc",
        "atlas_text",
        "projection_name",
    ]
    selected = selected[columns]

    build_info = {
        "correction_code": spec["code"],
        "correction_name": spec["name"],
        "rows": int(len(selected)),
        "unique_labels": int(selected["label"].nunique()),
        "source_embeddings": str(spec["pt_path"]),
        "reference_labels": str(EMBEDDINGS_25_PATH),
        "mapping_strategy": "per-label logical order within the 2+encounters train split",
        "projection_recipe": {
            "reference_panel_for_topk": "25% correction",
            "topk_elephants": 50,
            "max_per_elephant": 500,
            "max_points_total": 25000,
            "selection_seed": 101,
            "projection_seed": spec["seed"],
            "perplexity": 10,
            "n_iter": 1200,
            "marker_size_in_reference_pdf": 20,
            "alpha_in_reference_pdf": 0.85,
        },
        "original_image_root": str(ORIGINAL_IMAGE_ROOT),
    }

    return selected, build_info


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _, labels_25 = load_pt(EMBEDDINGS_25_PATH)
    ids_keep = topk_ids(labels_25, k=50)

    for spec in CORRECTION_SPECS:
        output_path, info_path = output_paths(spec["code"])
        if output_path.exists() and info_path.exists() and not args.force:
            print(f"[skip] Cached data already exists at {output_path}")
            continue

        viewer_df, build_info = build_dataframe(spec, ids_keep)
        viewer_df.to_parquet(output_path, index=False)
        info_path.write_text(json.dumps(build_info, indent=2))

        print(f"[done] Wrote {len(viewer_df)} rows to {output_path}")
        print(f"[done] Wrote build info to {info_path}")


if __name__ == "__main__":
    main()
