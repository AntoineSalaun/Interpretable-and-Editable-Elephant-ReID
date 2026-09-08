from __future__ import annotations

import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    from .config import SQLITE_PATH, ensure_state_dirs
except ImportError:
    from config import SQLITE_PATH, ensure_state_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def array_to_blob(array) -> bytes:
    arr = np.asarray(array, dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def blob_to_array(blob: bytes):
    if blob is None:
        return None
    return np.load(io.BytesIO(blob), allow_pickle=False)


class SQLiteState:
    _lock = threading.RLock()

    def __init__(self, db_path: Path = SQLITE_PATH):
        ensure_state_dirs()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def ensure_schema(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS seek_corrections (
                    correction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    query_id TEXT DEFAULT '',
                    subject_id TEXT DEFAULT '',
                    ele_id TEXT DEFAULT '',
                    encounter_id TEXT DEFAULT '',
                    original_subject_seek TEXT DEFAULT '',
                    original_ele_seek TEXT DEFAULT '',
                    predicted_seek TEXT DEFAULT '',
                    corrected_seek TEXT DEFAULT '',
                    scope TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_seek_corrections_item
                    ON seek_corrections(item_type, query_id, encounter_id, ele_id, subject_id);

                CREATE TABLE IF NOT EXISTS ranking_logs (
                    ranking_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    seek_code TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    ranking_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    subject_id TEXT DEFAULT '',
                    decision_type TEXT NOT NULL,
                    selected_ele_id TEXT DEFAULT '',
                    selected_subject_id TEXT DEFAULT '',
                    notes TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_query
                    ON decisions(query_id, timestamp);

                CREATE TABLE IF NOT EXISTS hidden_gallery_images (
                    timestamp TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    encounter_id TEXT DEFAULT '',
                    reason TEXT DEFAULT '',
                    PRIMARY KEY(subject_id)
                );

                CREATE TABLE IF NOT EXISTS uploads (
                    query_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    original_filename TEXT DEFAULT '',
                    raw_image_path TEXT NOT NULL,
                    body_image_path TEXT DEFAULT '',
                    left_ear_path TEXT DEFAULT '',
                    right_ear_path TEXT DEFAULT '',
                    bbox TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS query_predictions (
                    query_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    predicted_seek TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rank_cache (
                    query_id TEXT NOT NULL,
                    seek_code TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    gallery_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ranking_json TEXT NOT NULL,
                    PRIMARY KEY(query_id, seek_code, top_k, gallery_version)
                );

                CREATE TABLE IF NOT EXISTS feature_cache (
                    image_key TEXT PRIMARY KEY,
                    image_type TEXT NOT NULL,
                    source_id TEXT DEFAULT '',
                    predicted_seek TEXT DEFAULT '',
                    backbone_embedding BLOB,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS seek_projection_cache (
                    seek_code TEXT PRIMARY KEY,
                    projection BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projected_embedding_cache (
                    image_key TEXT NOT NULL,
                    seek_code TEXT NOT NULL,
                    projected_embedding BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(image_key, seek_code)
                );
                """
            )

    def execute(self, sql: str, params=()):
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)

    def dataframe(self, table_name: str) -> pd.DataFrame:
        with self._lock, self.connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn).fillna("")

    def hidden_subject_ids(self) -> set[str]:
        with self._lock, self.connect() as conn:
            rows = conn.execute("SELECT subject_id FROM hidden_gallery_images").fetchall()
        return {str(row["subject_id"]) for row in rows}

    def hide_gallery_image(self, subject_id, encounter_id="", reason="hidden in UI") -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hidden_gallery_images
                (timestamp, subject_id, encounter_id, reason)
                VALUES (?, ?, ?, ?)
                """,
                (utc_now(), str(subject_id), str(encounter_id), str(reason)),
            )

    def gallery_seek_corrections(self) -> pd.DataFrame:
        with self._lock, self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM seek_corrections WHERE item_type = 'gallery' ORDER BY timestamp",
                conn,
            ).fillna("")

    def latest_query_seek(self, query_id) -> str | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT corrected_seek FROM seek_corrections
                WHERE item_type = 'query' AND query_id = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (str(query_id),),
            ).fetchone()
        return None if row is None else str(row["corrected_seek"])

    def query_predictions(self) -> dict[str, str]:
        with self._lock, self.connect() as conn:
            rows = conn.execute("SELECT query_id, predicted_seek FROM query_predictions").fetchall()
        return {str(row["query_id"]): str(row["predicted_seek"]) for row in rows}

    def log_query_prediction(self, query_id, predicted_seek) -> None:
        if not predicted_seek:
            return
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO query_predictions(query_id, timestamp, predicted_seek)
                VALUES (?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    predicted_seek = excluded.predicted_seek
                """,
                (str(query_id), utc_now(), str(predicted_seek)),
            )

    def query_statuses(self) -> dict[str, dict[str, str]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.* FROM decisions d
                JOIN (
                    SELECT query_id, MAX(timestamp) AS max_ts
                    FROM decisions GROUP BY query_id
                ) latest
                ON d.query_id = latest.query_id AND d.timestamp = latest.max_ts
                """
            ).fetchall()
        return {
            str(row["query_id"]): {
                "status": str(row["decision_type"]),
                "timestamp": str(row["timestamp"]),
                "selected_ele_id": str(row["selected_ele_id"]),
            }
            for row in rows
        }

    def log_decision(
        self,
        query_id,
        subject_id,
        decision_type,
        selected_ele_id="",
        selected_subject_id="",
        notes="",
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                (decision_id, timestamp, query_id, subject_id, decision_type,
                 selected_ele_id, selected_subject_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    utc_now(),
                    str(query_id),
                    str(subject_id),
                    str(decision_type),
                    str(selected_ele_id),
                    str(selected_subject_id),
                    str(notes),
                ),
            )

    def log_seek_correction(
        self,
        item_type,
        query_id,
        subject_id,
        encounter_id,
        original_subject_seek,
        original_ele_seek,
        predicted_seek,
        corrected_seek,
        scope,
        ele_id="",
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO seek_corrections
                (correction_id, timestamp, item_type, query_id, subject_id, ele_id,
                 encounter_id, original_subject_seek, original_ele_seek,
                 predicted_seek, corrected_seek, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    utc_now(),
                    str(item_type),
                    str(query_id),
                    str(subject_id),
                    str(ele_id),
                    str(encounter_id),
                    str(original_subject_seek),
                    str(original_ele_seek),
                    str(predicted_seek),
                    str(corrected_seek),
                    str(scope),
                ),
            )

    def gallery_version(self) -> str:
        with self._lock, self.connect() as conn:
            corrections = conn.execute(
                "SELECT COUNT(*) AS n FROM seek_corrections WHERE item_type = 'gallery'"
            ).fetchone()["n"]
            hidden = conn.execute(
                "SELECT COUNT(*) AS n FROM hidden_gallery_images"
            ).fetchone()["n"]
        return f"g{corrections}-h{hidden}"

    def cached_rank(self, query_id, seek_code, top_k, gallery_version=None):
        version = gallery_version or self.gallery_version()
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT ranking_json FROM rank_cache
                WHERE query_id = ? AND seek_code = ? AND top_k = ? AND gallery_version = ?
                """,
                (str(query_id), str(seek_code), int(top_k), version),
            ).fetchone()
        return None if row is None else json.loads(row["ranking_json"])

    def log_rank_cache(self, query_id, seek_code, top_k, ranking) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO rank_cache
                (query_id, seek_code, top_k, gallery_version, timestamp, ranking_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_id, seek_code, top_k, gallery_version) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    ranking_json = excluded.ranking_json
                """,
                (
                    str(query_id),
                    str(seek_code),
                    int(top_k),
                    self.gallery_version(),
                    utc_now(),
                    json.dumps(ranking),
                ),
            )

    def log_ranking(self, query_id, seek_code, top_k, ranking) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ranking_logs
                (ranking_id, timestamp, query_id, seek_code, top_k, ranking_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, utc_now(), str(query_id), str(seek_code), int(top_k), json.dumps(ranking)),
            )

    def uploads_dataframe(self) -> pd.DataFrame:
        with self._lock, self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM uploads ORDER BY timestamp DESC", conn).fillna("")

    def add_upload(self, row: dict) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO uploads
                (query_id, timestamp, original_filename, raw_image_path, body_image_path,
                 left_ear_path, right_ear_path, bbox, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["query_id"],
                    row.get("timestamp", utc_now()),
                    row.get("original_filename", ""),
                    row["raw_image_path"],
                    row.get("body_image_path", ""),
                    row.get("left_ear_path", ""),
                    row.get("right_ear_path", ""),
                    row.get("bbox", ""),
                    row.get("status", "pending"),
                ),
            )

    def get_feature(self, image_key) -> dict | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM feature_cache WHERE image_key = ?",
                (str(image_key),),
            ).fetchone()
        if row is None:
            return None
        return {
            "image_key": str(row["image_key"]),
            "image_type": str(row["image_type"]),
            "source_id": str(row["source_id"]),
            "predicted_seek": str(row["predicted_seek"]),
            "backbone_embedding": blob_to_array(row["backbone_embedding"]),
        }

    def upsert_feature(
        self,
        image_key,
        image_type,
        source_id="",
        predicted_seek=None,
        backbone_embedding=None,
    ) -> None:
        current = self.get_feature(image_key) or {}
        predicted = predicted_seek if predicted_seek is not None else current.get("predicted_seek", "")
        embedding = (
            array_to_blob(backbone_embedding)
            if backbone_embedding is not None
            else None
        )
        with self._lock, self.connect() as conn:
            if embedding is None:
                conn.execute(
                    """
                    INSERT INTO feature_cache
                    (image_key, image_type, source_id, predicted_seek, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_key) DO UPDATE SET
                        image_type = excluded.image_type,
                        source_id = excluded.source_id,
                        predicted_seek = excluded.predicted_seek,
                        updated_at = excluded.updated_at
                    """,
                    (str(image_key), str(image_type), str(source_id), str(predicted or ""), utc_now()),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO feature_cache
                    (image_key, image_type, source_id, predicted_seek, backbone_embedding, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_key) DO UPDATE SET
                        image_type = excluded.image_type,
                        source_id = excluded.source_id,
                        predicted_seek = excluded.predicted_seek,
                        backbone_embedding = excluded.backbone_embedding,
                        updated_at = excluded.updated_at
                    """,
                    (str(image_key), str(image_type), str(source_id), str(predicted or ""), embedding, utc_now()),
                )

    def get_seek_projection(self, seek_code):
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT projection FROM seek_projection_cache WHERE seek_code = ?",
                (str(seek_code),),
            ).fetchone()
        return None if row is None else blob_to_array(row["projection"])

    def put_seek_projection(self, seek_code, projection) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO seek_projection_cache(seek_code, projection, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(seek_code) DO UPDATE SET
                    projection = excluded.projection,
                    updated_at = excluded.updated_at
                """,
                (str(seek_code), array_to_blob(projection), utc_now()),
            )

    def get_projected_embedding(self, image_key, seek_code):
        with self._lock, self.connect() as conn:
            row = conn.execute(
                """
                SELECT projected_embedding FROM projected_embedding_cache
                WHERE image_key = ? AND seek_code = ?
                """,
                (str(image_key), str(seek_code)),
            ).fetchone()
        return None if row is None else blob_to_array(row["projected_embedding"])

    def put_projected_embedding(self, image_key, seek_code, embedding) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projected_embedding_cache
                (image_key, seek_code, projected_embedding, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_key, seek_code) DO UPDATE SET
                    projected_embedding = excluded.projected_embedding,
                    updated_at = excluded.updated_at
                """,
                (str(image_key), str(seek_code), array_to_blob(embedding), utc_now()),
            )

    def cache_counts(self) -> dict[str, int]:
        with self._lock, self.connect() as conn:
            names = [
                "feature_cache",
                "seek_projection_cache",
                "projected_embedding_cache",
                "rank_cache",
                "query_predictions",
            ]
            return {
                name: int(conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"])
                for name in names
            }

