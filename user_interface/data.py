from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
from PIL import Image

try:
    from .config import (
        DISPLAY_CROP_DIR,
        MARA_DICTIONARY_PATH,
        MARA_DATA_ROOT,
        MARA_ORIGINAL_ROOT,
        TEST_INDICES_PATH,
        TRAIN_INDICES_PATH,
        UI_CATALOG_PATH,
        ensure_state_dirs,
    )
    from .preprocessing import save_display_crops
    from .seek_utils import normalize_seek
except ImportError:
    from config import (
        DISPLAY_CROP_DIR,
        MARA_DICTIONARY_PATH,
        MARA_DATA_ROOT,
        MARA_ORIGINAL_ROOT,
        TEST_INDICES_PATH,
        TRAIN_INDICES_PATH,
        UI_CATALOG_PATH,
        ensure_state_dirs,
    )
    from preprocessing import save_display_crops
    from seek_utils import normalize_seek


def read_indices(path: Path):
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def canonical_seek(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return normalize_seek(value)
    except Exception:
        return str(value)


class DataCatalog:
    def __init__(
        self,
        dictionary_path: Path = UI_CATALOG_PATH,
        source_dictionary_path: Path = MARA_DICTIONARY_PATH,
        train_indices_path: Path = TRAIN_INDICES_PATH,
        test_indices_path: Path = TEST_INDICES_PATH,
    ):
        ensure_state_dirs()
        self.dictionary_path = dictionary_path
        self.source_dictionary_path = source_dictionary_path
        self.ensure_catalog_copy()
        self.df = pd.read_csv(dictionary_path)
        self.train_indices = read_indices(train_indices_path)
        self.test_indices = read_indices(test_indices_path)

    def ensure_catalog_copy(self):
        if self.dictionary_path.exists():
            return
        if not self.source_dictionary_path.exists():
            raise FileNotFoundError(f"Missing source catalog: {self.source_dictionary_path}")
        self.dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source_dictionary_path, self.dictionary_path)

    def row(self, idx):
        row = self.df.iloc[int(idx)].copy()
        row["idx"] = int(idx)
        row["query_id"] = f"dataset:{int(idx)}"
        return row

    def train_rows(self):
        return self.df.iloc[self.train_indices].copy().assign(idx=self.train_indices)

    def test_rows(self):
        rows = self.df.iloc[self.test_indices].copy().assign(idx=self.test_indices)
        rows["query_id"] = rows["idx"].map(lambda value: f"dataset:{int(value)}")
        return rows

    def resolve_original_path(self, row):
        return MARA_ORIGINAL_ROOT / "images" / str(row["image"])

    def resolve_preprocessed_path(self, path_value):
        if pd.isna(path_value) or str(path_value).lower() == "nan" or not str(path_value):
            return None
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        return MARA_DATA_ROOT / path

    def image_paths(self, row):
        return {
            "raw": self.resolve_original_path(row),
            "body": self.resolve_preprocessed_path(row.get("preprocessed_image_path", "")),
            "left_ear": self.resolve_preprocessed_path(row.get("left_ear_path", "")),
            "right_ear": self.resolve_preprocessed_path(row.get("right_ear_path", "")),
        }

    def display_crop_dir_for_idx(self, idx):
        return DISPLAY_CROP_DIR / f"dataset_{int(idx)}"

    def display_image_paths(self, row, force=False):
        idx = int(row["idx"]) if "idx" in row else int(row.name)
        raw_path = self.resolve_original_path(row)
        output_dir = self.display_crop_dir_for_idx(idx)
        body_path = output_dir / "body_rgb.jpg"
        left_path = output_dir / "left_ear_rgb.jpg"
        right_path = output_dir / "right_ear_rgb.jpg"

        if raw_path.exists() and (force or not body_path.exists()):
            save_display_crops(raw_path, output_dir, bbox=row.get("bbox", ""))

        return {
            "raw": raw_path,
            "body": body_path if body_path.exists() else None,
            "left_ear": left_path if left_path.exists() else None,
            "right_ear": right_path if right_path.exists() else None,
        }

    def open_image(self, path):
        if path is None or not Path(path).exists():
            return None
        return Image.open(path).convert("RGB")

    def apply_gallery_corrections(self, gallery, state):
        gallery = gallery.copy()
        gallery["corrected_seek"] = ""
        gallery["correction_scope"] = ""
        gallery["current_ele_seek"] = gallery["ele_seek"].map(canonical_seek)
        if state is None:
            return gallery

        corrections = state.gallery_seek_corrections()
        for _, correction in corrections.iterrows():
            corrected_seek = canonical_seek(correction.get("corrected_seek", ""))
            if not corrected_seek:
                continue
            scope = correction.get("scope", "")
            ele_id = str(correction.get("ele_id", ""))
            subject_id = str(correction.get("subject_id", ""))
            encounter_id = str(correction.get("encounter_id", ""))

            if scope in {"image", "this image only"}:
                mask = gallery["subject_id"] == subject_id
                label = "image"
            elif scope in {"elephant", "all images of this elephant"}:
                mask = gallery["ele_id"] == ele_id
                label = "elephant"
            elif scope in {"elephant_sighting", "all images of this elephant in this sighting"}:
                mask = (gallery["ele_id"] == ele_id) & (gallery["encounter_id"] == encounter_id)
                label = "elephant sighting"
            elif scope == "sighting":
                mask = gallery["encounter_id"] == encounter_id
                label = "sighting"
            else:
                mask = gallery["subject_id"] == subject_id
                label = "image"

            gallery.loc[mask, "current_ele_seek"] = corrected_seek
            gallery.loc[mask, "corrected_seek"] = corrected_seek
            gallery.loc[mask, "correction_scope"] = label
        return gallery

    def metadata_for_idx(self, idx, state=None, item_type="query"):
        row = self.row(idx)
        encounter_id = str(row["encounter_id"])
        current_ele_seek = canonical_seek(row["ele-SEEK"])
        correction_scope = ""
        if state and item_type == "gallery":
            corrected = self.apply_gallery_corrections(pd.DataFrame([{
                "idx": int(idx),
                "subject_id": str(row["subject_id"]),
                "ele_id": str(row["ele_id"]),
                "encounter_id": encounter_id,
                "subject_seek": canonical_seek(row["subject-SEEK"]),
                "ele_seek": canonical_seek(row["ele-SEEK"]),
                "picture_time": row.get("picture_time", ""),
            }]), state).iloc[0]
            current_ele_seek = corrected["current_ele_seek"]
            correction_scope = corrected["correction_scope"]
        query_seek = state.latest_query_seek(row["query_id"]) if state and item_type == "query" else None
        return {
            "idx": int(idx),
            "query_id": row["query_id"],
            "subject_id": str(row["subject_id"]),
            "ele_id": str(row["ele_id"]),
            "encounter_id": encounter_id,
            "subject_seek": canonical_seek(row["subject-SEEK"]),
            "ele_seek": canonical_seek(row["ele-SEEK"]),
            "current_ele_seek": current_ele_seek,
            "corrected_seek": current_ele_seek if correction_scope else "",
            "correction_scope": correction_scope,
            "current_query_seek": query_seek,
            "picture_time": row.get("picture_time", ""),
            "season": row.get("#season", ""),
            "paths": self.image_paths(row),
        }

    def queue_dataframe(self, state):
        statuses = state.query_statuses()
        predictions = state.query_predictions() if state is not None else {}
        corrected = {}
        if state is not None:
            corrections = state.read("seek_corrections.csv")
            corrections = corrections[corrections["item_type"] == "query"]
            for _, row in corrections.iterrows():
                corrected[str(row["query_id"])] = canonical_seek(row["corrected_seek"])
        rows = self.test_rows()
        queue = pd.DataFrame({
            "query_id": rows["query_id"].astype(str),
            "idx": rows["idx"].astype(int),
            "image_id": rows["idx"].astype(str),
            "subject_id": rows["subject_id"].astype(str),
            "ele_id": rows["ele_id"].astype(str),
            "encounter_id": rows["encounter_id"].astype(str),
            "subject_seek": rows["subject-SEEK"].map(canonical_seek),
            "ele_seek": rows["ele-SEEK"].map(canonical_seek),
        })
        status_series = queue["query_id"].map(lambda query_id: statuses.get(query_id, {}).get("status", "pending"))
        selected_series = queue["query_id"].map(lambda query_id: statuses.get(query_id, {}).get("selected_ele_id", ""))
        timestamp_series = queue["query_id"].map(lambda query_id: statuses.get(query_id, {}).get("timestamp", ""))
        queue["status"] = status_series
        queue["selected_ele_id"] = selected_series
        queue["last_decision"] = timestamp_series
        queue["predicted_seek"] = queue["query_id"].map(lambda query_id: predictions.get(query_id, ""))
        queue["corrected_seek"] = queue["query_id"].map(lambda query_id: corrected.get(query_id, ""))
        return queue[[
            "query_id", "idx", "image_id", "subject_id", "ele_id", "encounter_id",
            "status", "selected_ele_id", "last_decision", "predicted_seek",
            "corrected_seek", "subject_seek", "ele_seek",
        ]]

    def gallery_dataframe(self, state):
        hidden = state.hidden_subject_ids()
        rows = self.train_rows()
        gallery = pd.DataFrame({
            "idx": rows["idx"].astype(int),
            "subject_id": rows["subject_id"].astype(str),
            "ele_id": rows["ele_id"].astype(str),
            "encounter_id": rows["encounter_id"].astype(str),
            "subject_seek": rows["subject-SEEK"].map(canonical_seek),
            "ele_seek": rows["ele-SEEK"].map(canonical_seek),
            "picture_time": rows.get("picture_time", ""),
        })
        if hidden:
            gallery = gallery[~gallery["subject_id"].isin(hidden)]
        gallery = self.apply_gallery_corrections(gallery, state)
        return gallery[[
            "idx", "subject_id", "ele_id", "encounter_id", "subject_seek",
            "ele_seek", "current_ele_seek", "corrected_seek", "correction_scope",
            "picture_time",
        ]].reset_index(drop=True)
