from __future__ import annotations

from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "methods"
DATA_ROOT = Path(os.environ.get("CBM_REID_DATA_ROOT", REPO_ROOT / "data")).expanduser()
MARA_DATA_ROOT = DATA_ROOT / "mara"
MARA_ORIGINAL_ROOT = Path(os.environ.get(
    "CBM_REID_MARA_ORIGINAL_ROOT",
    "/archive/vision/beery/animal_reid/datasets/elephants_new",
)).expanduser()

STATE_DIR = Path(os.environ.get(
    "CBM_REID_MARIMO_STATE_DIR",
    REPO_ROOT / "marimo_interface" / "state",
)).expanduser()
UPLOAD_DIR = STATE_DIR / "uploads"
CACHE_DIR = STATE_DIR / "cache"
DISPLAY_CROP_DIR = STATE_DIR / "display_crops"
UI_CATALOG_PATH = STATE_DIR / "catalog.csv"
SQLITE_PATH = STATE_DIR / "marimo_ui.sqlite3"

UI_WEIGHTS_DIR = REPO_ROOT / "weights"
EAR_DETECTOR_WEIGHTS = UI_WEIGHTS_DIR / "ear_YOLOv5_n.pt"
TRAIN_INDICES_PATH = UI_WEIGHTS_DIR / "train_indices_gold.txt"
TEST_INDICES_PATH = UI_WEIGHTS_DIR / "test_indicies_gold.txt"
MARA_DICTIONARY_PATH = MARA_DATA_ROOT / "image_dictionary_optimized.csv"

DEFAULT_EXPERIMENT_DIR = Path(os.environ.get(
    "CBM_REID_UI_EXPERIMENT_DIR",
    REPO_ROOT.parent / "experiments" / "exp_Heatmap_three_100cor",
)).expanduser()

BACKBONE_FOR_CONCEPTS_PRETRAINING = "backbone_for_concepts_three_gold"
CONCEPT_HEAD_PRETRAINING = "concept_head_three_gold"
BACKBONE_PRETRAINING = "backbone_normalized_gold"

DEFAULT_TOP_K = int(os.environ.get("CBM_REID_UI_TOP_K", "10"))
RANK_CACHE_TOP_K = int(os.environ.get("CBM_REID_MARIMO_RANK_CACHE_TOP_K", "100"))


def ensure_state_dirs() -> None:
    for path in (STATE_DIR, UPLOAD_DIR, CACHE_DIR, DISPLAY_CROP_DIR):
        path.mkdir(parents=True, exist_ok=True)

