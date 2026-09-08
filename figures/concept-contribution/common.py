from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
METRICS_DIR = SCRIPT_DIR / "metrics"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"

sys.path.insert(0, str(ROOT / "methods"))
from seek_code import SEEK  # noqa: E402


LEVELS = [25, 50, 75, 100]
RUN_IDS = {
    25: "e7qlnpqs",
    50: "qqqvgtl3",
    75: "jjex4uvi",
    100: "y236jtda",
}

CONCEPT_NAMES = SEEK.attribute_names

CONCEPT_GROUPS = {
    "sex": ["sex"],
    "age": ["age"],
    "tusks": ["right_tusk", "left_tusk"],
    "tear_1": ["R_tear_1", "L_tear_1"],
    "tear_2": ["R_tear_2", "L_tear_2"],
    "hole_1": ["R_hole_1", "L_hole_1"],
    "hole_2": ["R_hole_2", "L_hole_2"],
    "extreme_special": ["right_extreme", "left_extreme", "ear_special", "body_special"],
}
GROUP_NAMES = list(CONCEPT_GROUPS)


def read_indices(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def concept_slices() -> dict[str, slice]:
    out = {}
    start = 0
    for name in CONCEPT_NAMES:
        stop = start + SEEK.lengths[name]
        out[name] = slice(start, stop)
        start = stop
    return out


SLICES = concept_slices()


def one_hot_to_categories(concepts: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(concepts):
        values = concepts.detach().cpu().numpy()
    else:
        values = np.asarray(concepts)

    cats = []
    for name in CONCEPT_NAMES:
        block = values[:, SLICES[name]]
        cats.append(np.argmax(block, axis=1))
    return np.stack(cats, axis=1).astype(np.int64)


def concept_block(concepts: torch.Tensor | np.ndarray, concept: str) -> np.ndarray:
    if torch.is_tensor(concepts):
        values = concepts.detach().cpu().numpy()
    else:
        values = np.asarray(concepts)
    return values[:, SLICES[concept]].astype(np.float32)


def group_block(concepts: torch.Tensor | np.ndarray, group: str) -> np.ndarray:
    if torch.is_tensor(concepts):
        values = concepts.detach().cpu().numpy()
    else:
        values = np.asarray(concepts)
    return np.concatenate(
        [values[:, SLICES[concept]] for concept in CONCEPT_GROUPS[group]],
        axis=1,
    ).astype(np.float32)


def grouped_feature_matrix(concepts: torch.Tensor | np.ndarray) -> np.ndarray:
    return np.concatenate([group_block(concepts, group) for group in GROUP_NAMES], axis=1)


def feature_without_group(concepts: torch.Tensor | np.ndarray, group: str) -> np.ndarray:
    if torch.is_tensor(concepts):
        values = concepts.detach().cpu().numpy()
    else:
        values = np.asarray(concepts)
    mask = np.ones(values.shape[1], dtype=bool)
    for concept in CONCEPT_GROUPS[group]:
        mask[SLICES[concept]] = False
    return values[:, mask].astype(np.float32)


def group_categories(categories: np.ndarray, group: str) -> np.ndarray:
    concept_to_idx = {name: idx for idx, name in enumerate(CONCEPT_NAMES)}
    group_values = categories[:, [concept_to_idx[name] for name in CONCEPT_GROUPS[group]]]
    _, encoded = np.unique(group_values, axis=0, return_inverse=True)
    return encoded.astype(np.int64)


def safe_torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        METRICS_DIR,
        CHECKPOINT_DIR,
        SCRIPT_DIR / "geometry",
        SCRIPT_DIR / "single-concept-identity",
        SCRIPT_DIR / "leave-one-out",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def pct_label(level: int) -> str:
    return f"{level}%"
