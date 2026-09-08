from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

try:
    from .config import STATE_DIR, ensure_state_dirs
except ImportError:
    from config import STATE_DIR, ensure_state_dirs


CSV_SCHEMAS = {
    "seek_corrections.csv": [
        "correction_id", "timestamp", "item_type", "query_id", "subject_id",
        "ele_id", "encounter_id", "original_subject_seek", "original_ele_seek",
        "predicted_seek", "corrected_seek", "scope",
    ],
    "ranking_logs.csv": [
        "ranking_id", "timestamp", "query_id", "seek_code", "top_k",
        "ranking_json",
    ],
    "decisions.csv": [
        "decision_id", "timestamp", "query_id", "subject_id",
        "decision_type", "selected_ele_id", "selected_subject_id", "notes",
    ],
    "hidden_gallery_images.csv": [
        "timestamp", "subject_id", "encounter_id", "reason",
    ],
    "uploads.csv": [
        "query_id", "timestamp", "original_filename", "raw_image_path",
        "body_image_path", "left_ear_path", "right_ear_path", "bbox",
        "status",
    ],
    "query_predictions.csv": [
        "query_id", "timestamp", "predicted_seek",
    ],
    "rank_cache.csv": [
        "query_id", "timestamp", "seek_code", "top_k", "gallery_version",
        "ranking_json",
    ],
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class UIState:
    _lock = threading.RLock()

    def __init__(self, state_dir: Path = STATE_DIR):
        self.state_dir = state_dir
        ensure_state_dirs()
        self.ensure_files()

    def path(self, filename):
        return self.state_dir / filename

    def _empty_dataframe(self, filename):
        return pd.DataFrame(columns=CSV_SCHEMAS[filename])

    def _read_csv(self, filename):
        path = self.path(filename)
        try:
            return pd.read_csv(path, dtype=str).fillna("")
        except pd.errors.EmptyDataError:
            return self._empty_dataframe(filename)

    def _write_csv(self, filename, df):
        path = self.path(filename)
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    def ensure_files(self):
        with self._lock:
            for filename, columns in CSV_SCHEMAS.items():
                path = self.path(filename)
                if not path.exists():
                    self._write_csv(filename, self._empty_dataframe(filename))
                    continue
                df = self._read_csv(filename)
                missing_columns = [column for column in columns if column not in df.columns]
                if missing_columns:
                    for column in missing_columns:
                        df[column] = ""
                    df = df[columns]
                    self._write_csv(filename, df)

    def read(self, filename):
        with self._lock:
            self.ensure_files()
            return self._read_csv(filename)

    def append(self, filename, row):
        with self._lock:
            df = self.read(filename)
            full_row = {col: row.get(col, "") for col in CSV_SCHEMAS[filename]}
            df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
            self._write_csv(filename, df)

    def replace(self, filename, df):
        with self._lock:
            full_df = df.copy()
            for column in CSV_SCHEMAS[filename]:
                if column not in full_df.columns:
                    full_df[column] = ""
            self._write_csv(filename, full_df[CSV_SCHEMAS[filename]])

    def hidden_subject_ids(self):
        hidden = self.read("hidden_gallery_images.csv")
        return set(hidden["subject_id"].astype(str).tolist())

    def hide_gallery_image(self, subject_id, encounter_id="", reason="hidden in UI"):
        if str(subject_id) in self.hidden_subject_ids():
            return
        self.append("hidden_gallery_images.csv", {
            "timestamp": utc_now(),
            "subject_id": subject_id,
            "encounter_id": encounter_id,
            "reason": reason,
        })

    def latest_gallery_seek_by_encounter(self):
        corrections = self.read("seek_corrections.csv")
        corrections = corrections[
            (corrections["item_type"] == "gallery")
            & (corrections["encounter_id"] != "")
        ]
        result = {}
        for _, row in corrections.iterrows():
            result[str(row["encounter_id"])] = row["corrected_seek"]
        return result

    def gallery_seek_corrections(self):
        corrections = self.read("seek_corrections.csv")
        return corrections[corrections["item_type"] == "gallery"].reset_index(drop=True)

    def latest_query_seek(self, query_id):
        corrections = self.read("seek_corrections.csv")
        corrections = corrections[
            (corrections["item_type"] == "query")
            & (corrections["query_id"] == str(query_id))
        ]
        if corrections.empty:
            return None
        return corrections.iloc[-1]["corrected_seek"]

    def query_predictions(self):
        predictions = self.read("query_predictions.csv")
        result = {}
        for _, row in predictions.iterrows():
            result[str(row["query_id"])] = row["predicted_seek"]
        return result

    def log_query_prediction(self, query_id, predicted_seek):
        predictions = self.read("query_predictions.csv")
        existing = predictions[
            (predictions["query_id"] == str(query_id))
            & (predictions["predicted_seek"] == str(predicted_seek))
        ]
        if not existing.empty:
            return
        self.append("query_predictions.csv", {
            "query_id": query_id,
            "timestamp": utc_now(),
            "predicted_seek": predicted_seek,
        })

    def gallery_version(self):
        gallery_corrections = self.read("seek_corrections.csv")
        gallery_corrections = gallery_corrections[gallery_corrections["item_type"] == "gallery"]
        hidden = self.read("hidden_gallery_images.csv")
        return f"g{len(gallery_corrections)}-h{len(hidden)}"

    def cached_rank(self, query_id, seek_code, top_k, gallery_version=None):
        cache = self.read("rank_cache.csv")
        version = gallery_version or self.gallery_version()
        matches = cache[
            (cache["query_id"] == str(query_id))
            & (cache["seek_code"] == str(seek_code))
            & (cache["top_k"] == str(top_k))
            & (cache["gallery_version"] == version)
        ]
        if matches.empty:
            return None
        return json.loads(matches.iloc[-1]["ranking_json"])

    def log_rank_cache(self, query_id, seek_code, top_k, ranking):
        self.append("rank_cache.csv", {
            "query_id": query_id,
            "timestamp": utc_now(),
            "seek_code": seek_code,
            "top_k": str(top_k),
            "gallery_version": self.gallery_version(),
            "ranking_json": json.dumps(ranking),
        })

    def query_statuses(self):
        decisions = self.read("decisions.csv")
        result = {}
        for _, row in decisions.iterrows():
            result[str(row["query_id"])] = {
                "status": row["decision_type"],
                "timestamp": row["timestamp"],
                "selected_ele_id": row["selected_ele_id"],
            }
        return result

    def log_seek_correction(
        self, item_type, query_id, subject_id, encounter_id,
        original_subject_seek, original_ele_seek, predicted_seek,
        corrected_seek, scope, ele_id="",
    ):
        corrections = self.read("seek_corrections.csv")
        self.append("seek_corrections.csv", {
            "correction_id": str(len(corrections) + 1),
            "timestamp": utc_now(),
            "item_type": item_type,
            "query_id": query_id,
            "subject_id": subject_id,
            "ele_id": ele_id,
            "encounter_id": encounter_id,
            "original_subject_seek": original_subject_seek,
            "original_ele_seek": original_ele_seek,
            "predicted_seek": predicted_seek,
            "corrected_seek": corrected_seek,
            "scope": scope,
        })

    def log_ranking(self, query_id, seek_code, top_k, ranking):
        rankings = self.read("ranking_logs.csv")
        self.append("ranking_logs.csv", {
            "ranking_id": str(len(rankings) + 1),
            "timestamp": utc_now(),
            "query_id": query_id,
            "seek_code": seek_code,
            "top_k": str(top_k),
            "ranking_json": json.dumps(ranking),
        })

    def log_decision(
        self, query_id, subject_id, decision_type,
        selected_ele_id="", selected_subject_id="", notes="",
    ):
        decisions = self.read("decisions.csv")
        self.append("decisions.csv", {
            "decision_id": str(len(decisions) + 1),
            "timestamp": utc_now(),
            "query_id": query_id,
            "subject_id": subject_id,
            "decision_type": decision_type,
            "selected_ele_id": selected_ele_id,
            "selected_subject_id": selected_subject_id,
            "notes": notes,
        })
