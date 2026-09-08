from __future__ import annotations

import base64
import hashlib
import html
import io
import math
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .config import DEFAULT_EXPERIMENT_DIR, DEFAULT_TOP_K  # noqa: E402
    from .data import DataCatalog  # noqa: E402
    from .model_service import ModelService  # noqa: E402
    from .preprocessing import preprocess_upload  # noqa: E402
    from .seek_utils import ATTRIBUTES, attrs_to_seek, is_valid_seek, normalize_seek, seek_to_attrs  # noqa: E402
    from .state import UIState  # noqa: E402
except ImportError:
    from config import DEFAULT_EXPERIMENT_DIR, DEFAULT_TOP_K  # noqa: E402
    from data import DataCatalog  # noqa: E402
    from model_service import ModelService  # noqa: E402
    from preprocessing import preprocess_upload  # noqa: E402
    from seek_utils import ATTRIBUTES, attrs_to_seek, is_valid_seek, normalize_seek, seek_to_attrs  # noqa: E402
    from state import UIState  # noqa: E402


st.set_page_config(
    page_title="Elephant Re-ID",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, .stApp {
        background: #ffffff !important;
        color: #24292f !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: #ffffff !important;
        color: #24292f !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    label, p, span, div {
        color-scheme: light;
    }
    input, textarea,
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #24292f !important;
        border-color: #d8d8d8 !important;
    }
    div[data-testid="stTextInput"] input {
        min-height: 2.25rem !important;
        height: 2.25rem !important;
    }
    div[data-testid="stButton"] > button {
        background-color: #ffffff !important;
        color: #24292f !important;
        border-color: #d8d8d8 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #315f86 !important;
        color: #ffffff !important;
        border-color: #315f86 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] [aria-selected="true"] {
        background-color: #315f86 !important;
        color: #ffffff !important;
        border-color: #315f86 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
    div[data-testid="stSegmentedControl"] [aria-selected="true"] * {
        color: #ffffff !important;
    }
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetric"] { border: 1px solid #ece8df; padding: 0.55rem 0.7rem; border-radius: 8px; }
    .small-muted { color: #6b6b6b; font-size: 0.85rem; }
    .section-title {
        color: #303030;
        font-size: 0.92rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }
    .table-head {
        color: #5f6368;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 1px solid #e8e8e8;
        padding-bottom: 0.12rem;
    }
    .row-separator { border-bottom: 1px solid #f1f1f1; margin: 0.1rem 0 0.28rem 0; }
    .gallery-table {
        border: 1px solid #e9e9e9;
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.68rem;
    }
    .gallery-row {
        display: grid;
        grid-template-columns: 0.72fr 0.72fr 0.72fr 1.08fr 0.55fr 1.9fr 1.9fr 1.9fr;
        min-height: 27px;
        align-items: center;
        border-bottom: 1px solid #f0f0f0;
    }
    .gallery-row:last-child { border-bottom: 0; }
    .gallery-header {
        min-height: 29px;
        background: #fafafa;
        color: #5f6368;
        font-size: 0.62rem;
        font-weight: 760;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .gallery-cell {
        padding: 3px 6px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        line-height: 1.25;
        border-right: 1px solid #f5f5f5;
    }
    .gallery-cell:last-child { border-right: 0; }
    .gallery-link {
        color: #315f86;
        text-decoration: none;
        font-weight: 650;
    }
    .gallery-link:hover { text-decoration: underline; }
    .sort-link {
        color: #5f6368;
        text-decoration: none;
        font-weight: 760;
    }
    .sort-link:hover { color: #315f86; text-decoration: underline; }
    .seek-cell {
        color: #323232;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 0.62rem;
    }
    .compact-row-button button {
        border: 0;
        background: transparent;
        color: #315f86;
        min-height: 1.45rem !important;
        padding: 0.05rem 0 !important;
        text-align: left;
        font-size: 0.72rem;
        font-weight: 650;
    }
    .compact-row-button div[data-testid="stButton"] { margin-bottom: 0 !important; }
    .compact-cell {
        font-size: 0.72rem;
        line-height: 1.2;
        padding-top: 0.15rem;
    }
    .ui-badge {
        display: inline-block;
        padding: 0.05rem 0.32rem;
        border-radius: 999px;
        color: #1f2328;
        font-size: 0.66rem;
        font-weight: 650;
        line-height: 1.2;
        border: 1px solid rgba(31, 35, 40, 0.12);
    }
    .badge-pending { background: #fff1bf; border-color: #e5c653; }
    .badge-matched { background: #ccefd8; border-color: #6bb37d; }
    .badge-uncertain { background: #ded4ff; border-color: #9d84e6; }
    .badge-rejected { background: #ffd1d1; border-color: #df7878; }
    .badge-corrected { background: #eaf2f5; }
    .badge-original { background: #f1f1f1; }
    div[data-testid="stButton"] > button {
        min-height: 1.9rem;
        padding: 0.15rem 0.45rem;
    }
    .seek-editor-label {
        color: #5f6368;
        font-size: 0.58rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: -0.35rem;
        white-space: nowrap;
    }
    div[data-testid="stSelectbox"] label { display: none; }
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        min-height: 1.5rem;
        font-size: 0.6rem;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0.1rem;
        padding-right: 0.1rem;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    div[data-testid="stSelectbox"] [data-baseweb="select"] input {
        font-size: 0.6rem !important;
        line-height: 1rem !important;
    }
    .candidate-card {
        border-top: 1px solid #ececec;
        padding: 0.7rem 0 0.85rem 0;
    }
    .candidate-card:first-child { border-top: 0; padding-top: 0; }
    .candidate-chips {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .candidate-chip {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 0.25rem 0.45rem;
        background: #fbfbfb;
        color: #24292f;
        font-size: 0.75rem;
        font-weight: 720;
        line-height: 1.05;
        min-width: 0;
        height: 2.5rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
        box-sizing: border-box;
        text-align: center;
        white-space: nowrap;
    }
    .candidate-chip small {
        display: block;
        color: #16833a;
        font-size: 0.62rem;
        font-weight: 700;
        margin-top: 0.15rem;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .candidate-chip-link {
        color: #24292f;
        text-decoration: none;
    }
    .candidate-chip-link:hover { color: #315f86; text-decoration: none; }
    .candidate-select-chip {
        background: #315f86;
        border-color: #315f86;
        color: #ffffff;
        text-decoration: none;
    }
    .candidate-select-chip:hover { color: #ffffff; text-decoration: none; }
    .decision-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.45rem;
        align-items: stretch;
        margin-top: 0.15rem;
    }
    .decision-action {
        min-height: 2.25rem;
        border-radius: 8px;
        border: 1px solid rgba(31, 35, 40, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        padding: 0.15rem 0.35rem;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 720;
        line-height: 1;
        text-align: center;
        text-decoration: none;
        white-space: nowrap;
    }
    .decision-action:hover { color: #ffffff; text-decoration: none; }
    .decision-uncertain { background: #c28a00; border-color: #c28a00; }
    .decision-reject { background: #c94f4f; border-color: #c94f4f; }
    .decision-choose { background: #315f86; border-color: #315f86; color: #ffffff; }
    .decision-choose:hover { color: #ffffff; text-decoration: none; }
    .decision-disabled {
        background: #f5f5f5;
        border-color: #d8d8d8;
        color: #888888;
        pointer-events: none;
    }
    .seek-compare {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 0.68rem;
        line-height: 1.45;
        margin-bottom: 0.45rem;
        word-break: break-all;
    }
    .seek-match { background: #e6f1ea; border-radius: 3px; padding: 0 1px; }
    .seek-unknown { background: #f5edd8; border-radius: 3px; padding: 0 1px; }
    .seek-diff { background: #f4e5e5; border-radius: 3px; padding: 0 1px; }
    .modal-label {
        color: #5f6368;
        font-size: 0.78rem;
        font-weight: 720;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_catalog():
    return DataCatalog()


@st.cache_resource
def get_model_service(_catalog, _state):
    return ModelService(_catalog, _state, experiment_dir=DEFAULT_EXPERIMENT_DIR)


def get_state():
    return UIState()


catalog = get_catalog()
state = get_state()
model = get_model_service(catalog, state)


def state_file_fingerprint(filenames):
    fingerprint = []
    for filename in filenames:
        path = state.path(filename)
        if path.exists():
            stat = path.stat()
            fingerprint.append((filename, stat.st_mtime_ns, stat.st_size))
        else:
            fingerprint.append((filename, None, None))
    return tuple(fingerprint)


@st.cache_data(show_spinner=False)
def cached_queue_dataframe(_catalog, _state, fingerprint):
    _ = fingerprint
    return _catalog.queue_dataframe(_state)


@st.cache_data(show_spinner=False)
def cached_upload_query_rows(_state, fingerprint):
    _ = fingerprint
    uploads = _state.read("uploads.csv")
    rows = []
    statuses = _state.query_statuses()
    predictions = _state.query_predictions()
    for _, row in uploads.iterrows():
        query_id = str(row["query_id"])
        status = statuses.get(query_id, {})
        rows.append({
            "query_id": query_id,
            "idx": "",
            "image_id": query_id,
            "subject_id": "",
            "ele_id": "",
            "encounter_id": "",
            "status": status.get("status", row.get("status", "pending")),
            "selected_ele_id": status.get("selected_ele_id", ""),
            "last_decision": status.get("timestamp", ""),
            "predicted_seek": predictions.get(query_id, ""),
            "corrected_seek": _state.latest_query_seek(query_id) or "",
            "subject_seek": "",
            "ele_seek": "",
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_gallery_dataframe(_catalog, _state, fingerprint):
    _ = fingerprint
    return _catalog.gallery_dataframe(_state)


STATUS_LABELS = {
    "pending": "pending",
    "matched": "matched",
    "uncertain": "uncertain",
    "rejected": "rejected",
}

QUERY_REVIEW_RANK_CACHE_TOP_K = 100
QUEUE_TABLE_STATE_FILES = (
    "uploads.csv",
    "decisions.csv",
    "query_predictions.csv",
    "seek_corrections.csv",
)
GALLERY_TABLE_STATE_FILES = (
    "hidden_gallery_images.csv",
    "seek_corrections.csv",
)


class QueryPrecomputeWorker:
    def __init__(
        self,
        top_k=QUERY_REVIEW_RANK_CACHE_TOP_K,
        poll_seconds=5,
        prediction_batch_size=64,
        rank_batch_size=2,
        cycle_pause_seconds=0.75,
    ):
        self.top_k = int(top_k)
        self.poll_seconds = float(poll_seconds)
        self.prediction_batch_size = int(prediction_batch_size)
        self.rank_batch_size = int(rank_batch_size)
        self.cycle_pause_seconds = float(cycle_pause_seconds)
        self._lock = threading.Lock()
        self._thread = None
        self.status = {
            "state": "idle",
            "message": "waiting",
            "predicted": 0,
            "ranked": 0,
            "skipped": 0,
            "failed": 0,
        }

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run_forever,
                name="query-precompute-worker",
                daemon=True,
            )
            self._thread.start()

    def snapshot(self):
        with self._lock:
            return dict(self.status)

    def _set_status(self, **updates):
        with self._lock:
            self.status.update(updates)

    def _increment_status(self, **updates):
        with self._lock:
            for key, value in updates.items():
                self.status[key] = self.status.get(key, 0) + value

    def _run_forever(self):
        while True:
            try:
                did_work = self._run_once()
                if did_work:
                    time.sleep(self.cycle_pause_seconds)
                else:
                    self._set_status(state="idle", message="cache is current")
                    time.sleep(self.poll_seconds)
            except Exception as exc:
                self._set_status(state="error", message=str(exc))
                self._increment_status(failed=1)
                time.sleep(self.poll_seconds)

    def _run_once(self):
        worker_state = UIState()
        worker_catalog = DataCatalog()
        worker_model = ModelService(worker_catalog, worker_state, experiment_dir=DEFAULT_EXPERIMENT_DIR)
        queue = worker_catalog.queue_dataframe(worker_state)
        uploads = worker_state.read("uploads.csv")
        if not uploads.empty:
            predictions = worker_state.query_predictions()
            upload_rows = []
            for _, row in uploads.iterrows():
                upload_rows.append({
                    "query_id": row["query_id"],
                    "idx": "",
                    "predicted_seek": predictions.get(str(row["query_id"]), ""),
                })
            queue = pd.concat([queue, pd.DataFrame(upload_rows)], ignore_index=True)
        if queue.empty:
            return False

        did_work = False
        missing_dataset = queue[
            (queue["predicted_seek"].astype(str) == "")
            & queue["query_id"].astype(str).str.startswith("dataset:")
        ]
        if not missing_dataset.empty:
            did_work = True
            batch_rows = missing_dataset.head(self.prediction_batch_size)
            idxs = batch_rows["idx"].astype(int).tolist()
            query_ids = batch_rows["query_id"].astype(str).tolist()
            self._set_status(
                state="predicting",
                message=(
                    f"predicting SEEK {len(query_ids)} at a time; "
                    f"{len(missing_dataset)} queued queries remain"
                ),
            )
            predictions = worker_model.predict_seek_for_dataset_idxs(idxs)
            predicted_count = 0
            for query_id, idx in zip(query_ids, idxs):
                predicted = predictions.get(int(idx), "")
                if predicted:
                    worker_state.log_query_prediction(query_id, predicted)
                    predicted_count += 1
            self._increment_status(predicted=predicted_count)

        missing_uploads = queue[
            (queue["predicted_seek"].astype(str) == "")
            & queue["query_id"].astype(str).str.startswith("upload:")
        ]
        if not missing_uploads.empty:
            did_work = True
            predicted_count = 0
            upload_query_ids = missing_uploads["query_id"].astype(str).head(self.prediction_batch_size).tolist()
            self._set_status(
                state="predicting",
                message=(
                    f"predicting SEEK {len(upload_query_ids)} at a time; "
                    f"{len(missing_uploads)} uploaded queries remain"
                ),
            )
            for query_id in upload_query_ids:
                upload_rows = uploads[uploads["query_id"] == query_id]
                if upload_rows.empty:
                    continue
                predicted = worker_model.predict_seek_for_upload(upload_rows.iloc[-1].to_dict())
                worker_state.log_query_prediction(query_id, predicted)
                predicted_count += 1
            self._increment_status(predicted=predicted_count)

        queue = worker_catalog.queue_dataframe(worker_state)
        uploads = worker_state.read("uploads.csv")
        if not uploads.empty:
            predictions = worker_state.query_predictions()
            upload_rows = []
            for _, row in uploads.iterrows():
                upload_rows.append({
                    "query_id": row["query_id"],
                    "idx": "",
                    "predicted_seek": predictions.get(str(row["query_id"]), ""),
                })
            queue = pd.concat([queue, pd.DataFrame(upload_rows)], ignore_index=True)

        ranked_count = 0
        skipped_count = 0
        failed_count = 0
        gallery_version = worker_state.gallery_version()
        for _, row in queue.iterrows():
            if ranked_count >= self.rank_batch_size:
                break
            query_id = str(row["query_id"])
            seek_code = str(row.get("predicted_seek", ""))
            if not seek_code:
                skipped_count += 1
                continue
            if worker_state.cached_rank(query_id, seek_code, self.top_k, gallery_version=gallery_version) is not None:
                skipped_count += 1
                continue
            did_work = True
            try:
                self._set_status(
                    state="ranking",
                    message=f"caching top-{self.top_k} pre-rank for {query_id}",
                )
                if query_id.startswith("dataset:"):
                    ranking = worker_model.rank_dataset_query(int(row["idx"]), seek_code, top_k=self.top_k)
                else:
                    upload_rows = uploads[uploads["query_id"] == query_id]
                    if upload_rows.empty:
                        skipped_count += 1
                        continue
                    ranking = worker_model.rank_upload_query(upload_rows.iloc[-1].to_dict(), seek_code, top_k=self.top_k)
                worker_state.log_rank_cache(query_id, seek_code, self.top_k, ranking)
                ranked_count += 1
            except Exception:
                failed_count += 1
            self._set_status(skipped=skipped_count)
            if ranked_count:
                self._increment_status(ranked=1)
                ranked_count = 0
            if failed_count:
                self._increment_status(failed=failed_count)
                failed_count = 0
        return did_work


@st.cache_resource
def get_query_precompute_worker():
    return QueryPrecomputeWorker(top_k=QUERY_REVIEW_RANK_CACHE_TOP_K)


def badge_html(label, badge_class):
    return f"<span class='ui-badge {badge_class}'>{label}</span>"


def status_badge_html(status):
    status = str(status or "pending")
    label = STATUS_LABELS.get(status, status)
    css_class = {
        "matched": "badge-matched",
        "uncertain": "badge-uncertain",
        "rejected": "badge-rejected",
        "pending": "badge-pending",
    }.get(status, "badge-pending")
    return badge_html(label, css_class)


def gallery_correction_badge_html(is_corrected):
    if is_corrected:
        return badge_html("corrected", "badge-corrected")
    return badge_html("original", "badge-original")


def table_status_label(status):
    status = str(status or "pending")
    return STATUS_LABELS.get(status, status)


def page_query_params(table_id):
    return {f"{table_id}_page": str(st.session_state.get(f"{table_id}_page_index", 0))}


def consume_page_query_param(table_id):
    value = st.query_params.get(f"{table_id}_page")
    if value is None:
        return None
    try:
        page_index = max(0, int(value))
    except (TypeError, ValueError):
        return None
    st.session_state[f"{table_id}_page_index"] = page_index
    return page_index


def consume_sort_query_params():
    params = st.query_params
    table_id = params.get("sort_table")
    sort_col = params.get("sort_col")
    sort_dir = params.get("sort_dir")
    if table_id and sort_col and sort_dir in {"asc", "desc"}:
        page_index = consume_page_query_param(table_id)
        st.session_state[f"{table_id}_sort_col"] = sort_col
        st.session_state[f"{table_id}_sort_dir"] = sort_dir
        if page_index is None and f"{table_id}_page_index" not in st.session_state:
            st.session_state[f"{table_id}_page_index"] = 0


def sorted_dataframe(df, table_id, allowed_columns):
    sort_col = st.session_state.get(f"{table_id}_sort_col")
    sort_dir = st.session_state.get(f"{table_id}_sort_dir", "asc")
    if sort_col not in allowed_columns or sort_col not in df.columns:
        return df
    result = df.copy()
    values = result[sort_col].astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    parsed_time = pd.to_datetime(values, errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(result) * 0.8)):
        result["_sort_value"] = numeric
    elif parsed_time.notna().sum() >= max(1, int(len(result) * 0.8)):
        result["_sort_value"] = parsed_time
    else:
        result["_sort_value"] = values.str.lower()
    return (
        result.sort_values("_sort_value", ascending=sort_dir == "asc", kind="mergesort")
        .drop(columns=["_sort_value"])
        .reset_index(drop=True)
    )


def random_sighting_group_sort(df):
    if "encounter_id" not in df.columns:
        return df
    result = df.copy()
    result["_sighting_order"] = result["encounter_id"].astype(str).map(
        lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()
    )
    secondary = "image_id" if "image_id" in result.columns else "subject_id"
    return (
        result.sort_values(["_sighting_order", "encounter_id", secondary], kind="mergesort")
        .drop(columns=["_sighting_order"])
        .reset_index(drop=True)
    )


def sort_header(label, table_id, column, view):
    current_col = st.session_state.get(f"{table_id}_sort_col")
    current_dir = st.session_state.get(f"{table_id}_sort_dir", "asc")
    next_dir = "desc" if current_col == column and current_dir == "asc" else "asc"
    arrow = " ↑" if current_col == column and current_dir == "asc" else ""
    arrow = " ↓" if current_col == column and current_dir == "desc" else arrow
    params = {
        "view": view,
        "sort_table": table_id,
        "sort_col": column,
        "sort_dir": next_dir,
    }
    params.update(page_query_params(table_id))
    href = "?" + urlencode(params)
    return f"<a class='sort-link' target='_self' href='{href}'>{html.escape(label + arrow)}</a>"


def style_badge_column(df, column, color_map):
    def style_cell(value):
        color = color_map.get(str(value), "#eeeeee")
        return f"background-color: {color}; font-weight: 650; color: #1f2328;"

    return df.style.map(style_cell, subset=[column])


def selected_dataframe_rows(event):
    if event is None:
        return []
    try:
        return list(event.selection.rows)
    except AttributeError:
        pass
    try:
        return list(event["selection"]["rows"])
    except (KeyError, TypeError):
        return []


def display_image(path, caption=None, use_container_width=True):
    image = catalog.open_image(path)
    if image is not None:
        st.image(image, caption=caption, use_container_width=use_container_width)


def image_data_uri(image):
    if image is None:
        return ""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def display_images_same_height(paths):
    body = catalog.open_image(paths.get("body"))
    left = catalog.open_image(paths.get("left_ear"))
    right = catalog.open_image(paths.get("right_ear"))

    crop_height = 280
    images = [("body crop", body)]
    if left is not None:
        images.append(("left ear", left))
    if right is not None:
        images.append(("right ear", right))
    if len(images) == 3:
        columns = "2fr 1fr 1fr"
    elif len(images) == 2:
        columns = "2fr 1fr"
    else:
        columns = "1fr"

    cards = []
    for label, image in images:
        uri = image_data_uri(image)
        if uri:
            image_html = (
                f"<img src='{uri}' style='height:{crop_height}px; max-width:100%; "
                "width:auto; object-fit:contain; display:block; margin:0 auto;'>"
            )
        else:
            image_html = (
                f"<div style='height:{crop_height}px; display:flex; align-items:center; "
                "justify-content:center; color:#777; border:1px solid #eee;'>missing</div>"
            )
        cards.append(
            "<div style='min-width:0;'>"
            f"{image_html}<div style='font-size:0.78rem; color:#666; text-align:center; margin-top:0.25rem;'>{label}</div>"
            "</div>"
        )
    st.markdown(
        f"<div style='display:grid; grid-template-columns: {columns}; "
        "gap:0.75rem; align-items:start;'>"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def readable_time(value):
    if value is None or str(value).strip() == "" or str(value).lower() == "nan":
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %-d, %Y %H:%M")


def display_query_images(meta, use_display=True):
    paths = meta["paths"]
    if use_display:
        if "display_paths" in meta:
            paths = meta["display_paths"]
        elif meta.get("idx") is not None:
            with st.spinner("Preparing RGB inspection crops..."):
                row = catalog.row(meta["idx"])
                paths = catalog.display_image_paths(row)
    display_images_same_height(paths)


def display_gallery_crop_triplet(idx, title=None):
    if title:
        st.caption(title)
    row = catalog.row(idx)
    paths = catalog.display_image_paths(row)
    display_images_same_height(paths)


def display_gallery_image_grid(rows):
    for _, row in rows.iterrows():
        display_gallery_crop_triplet(
            int(row["idx"]),
            title=f"image {row['subject_id']} · elephant {row['ele_id']} · sighting {row['encounter_id']}",
        )


def gallery_action_href(action, value):
    params = {"view": "Gallery", action: str(value)}
    params.update(page_query_params("gallery"))
    return "?" + urlencode(params)


def render_compact_gallery_table(page_gallery):
    headers = [
        ("Image ID", "subject_id"),
        ("Elephant ID", "ele_id"),
        ("Sighting ID", "encounter_id"),
        ("Image Time", "picture_time"),
        ("State", "corrected_seek"),
        ("Image-level SEEK", "subject_seek"),
        ("Sighting-level SEEK", "ele_seek"),
        ("Corrected SEEK", "corrected_seek"),
    ]
    rows = [
        "<div class='gallery-row gallery-header'>"
        + "".join(
            f"<div class='gallery-cell'>{sort_header(label, 'gallery', column, 'Gallery')}</div>"
            for label, column in headers
        )
        + "</div>"
    ]
    for _, row in page_gallery.iterrows():
        cells = [
            f"<a class='gallery-link' target='_self' href='{gallery_action_href('gallery_open', row['idx'])}'>{html.escape(str(row['subject_id']))}</a>",
            f"<a class='gallery-link' target='_self' href='{gallery_action_href('gallery_ele', row['ele_id'])}'>{html.escape(str(row['ele_id']))}</a>",
            f"<a class='gallery-link' target='_self' href='{gallery_action_href('gallery_sighting', row['encounter_id'])}'>{html.escape(str(row['encounter_id']))}</a>",
            html.escape(readable_time(row["picture_time"])),
            gallery_correction_badge_html(bool(row["corrected_seek"])),
            f"<span class='seek-cell'>{html.escape(str(row['subject_seek']))}</span>",
            f"<span class='seek-cell'>{html.escape(str(row['ele_seek']))}</span>",
            f"<span class='seek-cell'>{html.escape(str(row['corrected_seek'] or '-'))}</span>",
        ]
        rows.append(
            "<div class='gallery-row'>"
            + "".join(f"<div class='gallery-cell'>{cell}</div>" for cell in cells)
            + "</div>"
        )
    st.markdown("<div class='gallery-table'>" + "".join(rows) + "</div>", unsafe_allow_html=True)


def consume_gallery_query_params():
    params = st.query_params
    handled = False
    consume_page_query_param("gallery")
    if "gallery_open" in params:
        selected_idx = int(params["gallery_open"])
        st.session_state["gallery_selected_idx"] = selected_idx
        handled = True
    if "gallery_ele" in params:
        st.session_state["gallery_ele_filter"] = str(params["gallery_ele"])
        st.session_state["gallery_subject_filter"] = ""
        st.session_state["gallery_sighting_filter"] = ""
        st.session_state["gallery_selected_idx"] = None
        st.session_state["gallery_page_index"] = 0
        handled = True
    if "gallery_sighting" in params:
        st.session_state["gallery_sighting_filter"] = str(params["gallery_sighting"])
        st.session_state["gallery_subject_filter"] = ""
        st.session_state["gallery_ele_filter"] = ""
        st.session_state["gallery_selected_idx"] = None
        st.session_state["gallery_page_index"] = 0
        handled = True
    if handled:
        st.query_params.clear()


def query_action_href(query_id, table_id=None):
    params = {"view": "Query Review", "query_open": str(query_id)}
    if table_id in {"review", "queue"}:
        page = str(st.session_state.get(f"{table_id}_page_index", 0))
        params["review_page"] = page
        params["queue_page"] = page
    elif table_id:
        params.update(page_query_params(table_id))
    return "?" + urlencode(params)


def query_sighting_href(view, sighting_id):
    return "?" + urlencode({"view": view, "query_sighting": str(sighting_id)})


def render_compact_query_table(page_queue, table_id, view):
    headers = [
        ("Image ID", "image_id"),
        ("Sighting ID", "encounter_id"),
        ("Status", "status"),
        ("Matched Elephant", "selected_ele_id"),
        ("Predicted SEEK", "predicted_seek"),
        ("Corrected SEEK", "corrected_seek"),
    ]
    grid = "1fr 0.9fr 0.72fr 0.85fr 1.8fr 1.8fr"
    rows = [
        f"<div class='gallery-row gallery-header' style='grid-template-columns: {grid};'>"
        + "".join(
            f"<div class='gallery-cell'>{sort_header(label, table_id, column, view)}</div>"
            for label, column in headers
        )
        + "</div>"
    ]
    for _, row in page_queue.iterrows():
        selected = row["selected_ele_id"] if row["selected_ele_id"] else "-"
        sighting = row["encounter_id"] if row["encounter_id"] else "-"
        query_href = query_action_href(row["query_id"], table_id=table_id)
        sighting_cell = html.escape(str(sighting))
        if sighting != "-":
            sighting_cell = (
                f"<a class='gallery-link' target='_self' "
                f"href='{query_sighting_href(view, sighting)}'>{sighting_cell}</a>"
            )
        cells = [
            f"<a class='gallery-link' target='_self' href='{query_href}'>{html.escape(str(row['image_id']))}</a>",
            sighting_cell,
            status_badge_html(row["status"]),
            html.escape(str(selected)),
            f"<span class='seek-cell'>{html.escape(str(row['predicted_seek'] or '-'))}</span>",
            f"<span class='seek-cell'>{html.escape(str(row['corrected_seek'] or '-'))}</span>",
        ]
        rows.append(
            f"<div class='gallery-row' style='grid-template-columns: {grid};'>"
            + "".join(f"<div class='gallery-cell'>{cell}</div>" for cell in cells)
            + "</div>"
        )
    st.markdown("<div class='gallery-table'>" + "".join(rows) + "</div>", unsafe_allow_html=True)


def consume_query_query_params():
    params = st.query_params
    handled = False
    consume_page_query_param("review")
    consume_page_query_param("queue")
    if "query_open" in params:
        st.session_state["review_query_id"] = str(params["query_open"])
        handled = True
    if "select_identity" in params:
        query_id = str(params.get("query_open", st.session_state.get("review_query_id", "")))
        if query_id:
            st.session_state[f"pending_identity_ele_id:{query_id}"] = str(params["select_identity"])
        handled = True
    if "decision_action" in params:
        query_id = str(params.get("query_open", st.session_state.get("review_query_id", "")))
        action = str(params["decision_action"])
        if query_id and action in {"uncertain", "rejected"}:
            st.session_state[f"pending_decision_action:{query_id}"] = action
        handled = True
    if "choose_identity" in params:
        query_id = str(params.get("query_open", st.session_state.get("review_query_id", "")))
        if query_id:
            st.session_state[f"pending_choose_identity:{query_id}"] = str(params["choose_identity"])
        handled = True
    if "query_sighting" in params:
        sighting = str(params["query_sighting"])
        view = str(params.get("view", st.session_state.get("active_view", "Query Review")))
        if view == "Query Queue":
            st.session_state["queue_sighting"] = sighting
            st.session_state["queue_subject"] = ""
            st.session_state["queue_image"] = ""
            st.session_state["queue_page_index"] = 0
        else:
            st.session_state["review_sighting"] = sighting
            st.session_state["review_subject"] = ""
            st.session_state["review_image"] = ""
            st.session_state["review_page_index"] = 0
        handled = True
    if handled:
        st.query_params.clear()


def render_seek_editor(initial_seek, key_prefix):
    try:
        attrs = seek_to_attrs(initial_seek)
    except Exception:
        attrs = seek_to_attrs("B00T__E0000-0000X00S00")

    with st.expander("SEEK editor", expanded=True):
        short_labels = {
            "right_tusk": "R tusk",
            "left_tusk": "L tusk",
            "right_extreme": "R ext",
            "left_extreme": "L ext",
            "ear_special": "ear sp.",
            "body_special": "body sp.",
        }
        edited = {}
        for row_i in range(2):
            row_attributes = ATTRIBUTES[row_i * 8:(row_i + 1) * 8]
            cols = st.columns(8)
            for i, attr in enumerate(row_attributes):
                with cols[i]:
                    default_index = attr.choices.index(attrs[attr.name]) if attrs[attr.name] in attr.choices else 0
                    label = short_labels.get(attr.name, attr.name.replace("_", " "))
                    st.markdown(
                        f"<div class='seek-editor-label'>{html.escape(label)}</div>",
                        unsafe_allow_html=True,
                    )
                    edited[attr.name] = st.selectbox(
                        attr.name,
                        attr.choices,
                        index=default_index,
                        key=f"{key_prefix}_{attr.name}",
                        label_visibility="collapsed",
                    )
        categorical_code = attrs_to_seek(edited)
        raw_code = st.text_input(
            "SEEK code",
            value=categorical_code,
            key=f"{key_prefix}_raw",
            label_visibility="collapsed",
        )
        if is_valid_seek(raw_code):
            return normalize_seek(raw_code), True
        st.error("Invalid SEEK code")
        return categorical_code, False


def dataset_query_meta(query_id):
    idx = int(query_id.split(":", 1)[1])
    return catalog.metadata_for_idx(idx, state=state, item_type="query")


def upload_query_rows():
    return cached_upload_query_rows(
        state,
        state_file_fingerprint(QUEUE_TABLE_STATE_FILES),
    )


def all_queue_rows():
    queue = cached_queue_dataframe(
        catalog,
        state,
        state_file_fingerprint(QUEUE_TABLE_STATE_FILES),
    )
    uploads = upload_query_rows()
    if not uploads.empty:
        queue = pd.concat([uploads, queue], ignore_index=True)
    if not queue.empty:
        queue = queue.copy()
        queue["predicted_seek"] = queue.apply(
            lambda row: st.session_state.get(f"predicted_seek:{row['query_id']}", row.get("predicted_seek", "")),
            axis=1,
        )
    return queue


def selected_query_metadata(query_id):
    if query_id.startswith("dataset:"):
        return dataset_query_meta(query_id)

    uploads = state.read("uploads.csv")
    row = uploads[uploads["query_id"] == query_id].iloc[-1]
    return {
        "idx": None,
        "query_id": row["query_id"],
        "subject_id": "",
        "ele_id": "",
        "encounter_id": "",
        "subject_seek": "",
        "ele_seek": "",
        "current_ele_seek": "",
        "current_query_seek": state.latest_query_seek(row["query_id"]),
        "picture_time": row["timestamp"],
        "season": "upload",
        "paths": {
            "raw": row["raw_image_path"],
            "body": row["body_image_path"],
            "left_ear": row["left_ear_path"],
            "right_ear": row["right_ear_path"],
        },
        "display_paths": {
            "raw": row["raw_image_path"],
            "body": str(Path(row["raw_image_path"]).parent / "display" / "body_rgb.jpg"),
            "left_ear": str(Path(row["raw_image_path"]).parent / "display" / "left_ear_rgb.jpg"),
            "right_ear": str(Path(row["raw_image_path"]).parent / "display" / "right_ear_rgb.jpg"),
        },
        "upload_row": row.to_dict(),
    }


def predict_seek(query_id, meta):
    cache_key = f"predicted_seek:{query_id}"
    if cache_key in st.session_state:
        model.remember_query_prediction(query_id, st.session_state[cache_key])
        return st.session_state[cache_key]
    cached = state.query_predictions().get(query_id)
    if cached:
        st.session_state[cache_key] = cached
        model.remember_query_prediction(query_id, cached)
        return cached
    with st.spinner("Loading model and predicting SEEK..."):
        if query_id.startswith("dataset:"):
            predicted = model.predict_seek_for_dataset_idx(meta["idx"])
        else:
            predicted = model.predict_seek_for_upload(meta["upload_row"])
    state.log_query_prediction(query_id, predicted)
    st.session_state[cache_key] = predicted
    return predicted


def cached_query_prediction(query_id):
    cache_key = f"predicted_seek:{query_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    cached = state.query_predictions().get(query_id, "")
    if cached:
        st.session_state[cache_key] = cached
        model.remember_query_prediction(query_id, cached)
    return cached


def run_ranking(query_id, meta, seek_code, top_k):
    cached = state.cached_rank(query_id, seek_code, top_k)
    if cached is not None:
        st.session_state[f"ranking:{query_id}"] = cached
        return cached
    with st.spinner("Reranking gallery..."):
        if query_id.startswith("dataset:"):
            ranking = model.rank_dataset_query(meta["idx"], seek_code, top_k=top_k)
        else:
            ranking = model.rank_upload_query(meta["upload_row"], seek_code, top_k=top_k)
    state.log_ranking(query_id, seek_code, top_k, ranking)
    state.log_rank_cache(query_id, seek_code, top_k, ranking)
    st.session_state[f"ranking:{query_id}"] = ranking
    return ranking


def get_cached_prerank(query_id, predicted_seek, top_k):
    if not predicted_seek:
        return []
    cache_key = f"prerank:{query_id}:{predicted_seek}:{top_k}:{state.gallery_version()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    cached = state.cached_rank(query_id, predicted_seek, top_k)
    if cached is not None:
        st.session_state[cache_key] = cached
        return cached
    return []


def affected_gallery_idxs(meta, scope_code):
    gallery = cached_gallery_dataframe(
        catalog,
        state,
        state_file_fingerprint(GALLERY_TABLE_STATE_FILES),
    )
    if scope_code == "image":
        affected = gallery[gallery["subject_id"].astype(str) == str(meta["subject_id"])]
    elif scope_code == "elephant":
        affected = gallery[gallery["ele_id"].astype(str) == str(meta["ele_id"])]
    elif scope_code == "sighting":
        affected = gallery[gallery["encounter_id"].astype(str) == str(meta["encounter_id"])]
    else:
        affected = gallery[
            (gallery["ele_id"].astype(str) == str(meta["ele_id"]))
            & (gallery["encounter_id"].astype(str) == str(meta["encounter_id"]))
        ]
    return affected["idx"].astype(int).tolist()


def gallery_ele_href(ele_id):
    return "?" + urlencode({"view": "Gallery", "gallery_ele": str(ele_id)})


def gallery_sighting_href(sighting_id):
    return "?" + urlencode({"view": "Gallery", "gallery_sighting": str(sighting_id)})


def candidate_select_href(query_id, ele_id):
    params = {"view": "Query Review", "query_open": str(query_id), "select_identity": str(ele_id)}
    params["review_page"] = str(st.session_state.get("review_page_index", 0))
    params["queue_page"] = str(st.session_state.get("queue_page_index", st.session_state.get("review_page_index", 0)))
    return "?" + urlencode(params)


def decision_action_href(query_id, action, elephant_id=None):
    params = {"view": "Query Review", "query_open": str(query_id)}
    if action == "choose":
        params["choose_identity"] = str(elephant_id or "")
    else:
        params["decision_action"] = action
    params["review_page"] = str(st.session_state.get("review_page_index", 0))
    params["queue_page"] = str(st.session_state.get("queue_page_index", st.session_state.get("review_page_index", 0)))
    return "?" + urlencode(params)


def highlighted_seek(query_seek, candidate_seek):
    try:
        query_seek = normalize_seek(query_seek)
    except Exception:
        query_seek = str(query_seek or "")
    try:
        candidate_seek = normalize_seek(candidate_seek)
    except Exception:
        candidate_seek = str(candidate_seek or "")
    length = max(len(query_seek), len(candidate_seek))
    pieces = []
    for i in range(length):
        query_char = query_seek[i] if i < len(query_seek) else ""
        candidate_char = candidate_seek[i] if i < len(candidate_seek) else ""
        if query_char == candidate_char:
            css_class = "seek-match"
        elif query_char == "_" or candidate_char == "_":
            css_class = "seek-unknown"
        else:
            css_class = "seek-diff"
        pieces.append(f"<span class='{css_class}'>{html.escape(candidate_char or ' ')}</span>")
    return "".join(pieces)


def find_candidate_by_ele_id(ranking, ele_id):
    for candidate in ranking:
        if str(candidate.get("ele_id")) == str(ele_id):
            return candidate
    return None


def render_candidate(candidate, query_id):
    row = catalog.row(candidate["idx"])
    paths = catalog.display_image_paths(row)
    elephant_href = gallery_ele_href(candidate["ele_id"])
    sighting_href = gallery_sighting_href(candidate["encounter_id"])
    select_href = candidate_select_href(query_id, candidate["ele_id"])
    st.markdown("<div class='candidate-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='candidate-chips'>"
        f"<div class='candidate-chip'>#{html.escape(str(candidate['rank']))}"
        f"<small>{float(candidate['score']):.3f}</small></div>"
        f"<a class='candidate-chip candidate-chip-link' target='_blank' href='{elephant_href}'>"
        f"elephant {html.escape(str(candidate['ele_id']))}"
        f"<small>{html.escape(str(candidate['subject_id']))}</small></a>"
        f"<a class='candidate-chip candidate-chip-link' target='_blank' href='{sighting_href}'>"
        f"sighting {html.escape(str(candidate['encounter_id']))}</a>"
        f"<a class='candidate-chip candidate-select-chip' target='_self' href='{select_href}'>Select</a>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='seek-compare'>{highlighted_seek(st.session_state.get(f'active_query_seek:{query_id}', ''), candidate['current_ele_seek'])}</div>",
        unsafe_allow_html=True,
    )
    image_cols = st.columns([0.08, 0.92])
    with image_cols[0]:
        if st.button("⤢", key=f"compare_{query_id}_{candidate['idx']}", help="Make bigger"):
            st.session_state[f"compare:{query_id}"] = candidate
    with image_cols[1]:
        display_images_same_height(paths)
    st.markdown("</div>", unsafe_allow_html=True)


@st.dialog("Large comparison", width="large")
def render_comparison_dialog(query_meta, candidate):
    row = catalog.row(candidate["idx"])
    candidate_paths = catalog.display_image_paths(row)
    query_paths = query_meta.get("display_paths")
    if query_paths is None and query_meta.get("idx") is not None:
        query_paths = catalog.display_image_paths(catalog.row(query_meta["idx"]))
    if query_paths is None:
        query_paths = query_meta["paths"]
    cols = st.columns(2)
    with cols[0]:
        st.markdown("<div class='modal-label'>query</div>", unsafe_allow_html=True)
        display_images_same_height(query_paths)
    with cols[1]:
        st.markdown(
            f"<div class='modal-label'>candidate elephant {html.escape(str(candidate['ele_id']))}</div>",
            unsafe_allow_html=True,
        )
        display_images_same_height(candidate_paths)
    if st.button("Close comparison"):
        st.session_state.pop(f"compare:{query_meta['query_id']}", None)
        st.rerun()


def query_review_page():
    consume_query_query_params()
    queue = all_queue_rows()
    if queue.empty:
        st.info("No queries available.")
        return
    if "review_page_index" not in st.session_state:
        st.session_state["review_page_index"] = 0

    with st.container(border=True):
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        statuses = sorted(queue["status"].unique().tolist())
        filter_cols = st.columns([1, 1, 1, 1])
        selected_statuses = filter_cols[0].multiselect("Status", statuses, default=statuses, key="review_status")
        subject_filter = filter_cols[1].text_input("Subject ID", placeholder="exact", key="review_subject")
        sighting_filter = filter_cols[2].text_input("Sighting ID", placeholder="exact", key="review_sighting")
        query_search = filter_cols[3].text_input("Image ID", placeholder="exact", key="review_image")

    filter_state = (tuple(selected_statuses), subject_filter, sighting_filter, query_search)
    last_filter_state = st.session_state.get("review_last_filter_state")
    if last_filter_state is None:
        st.session_state["review_last_filter_state"] = filter_state
    elif last_filter_state != filter_state:
        st.session_state["review_last_filter_state"] = filter_state
        st.session_state["review_page_index"] = 0

    if selected_statuses:
        filtered_queue = queue[queue["status"].isin(selected_statuses)].reset_index(drop=True)
    else:
        filtered_queue = queue.reset_index(drop=True)
    if subject_filter:
        filtered_queue = filtered_queue[
            filtered_queue["subject_id"].astype(str) == subject_filter.strip()
        ].reset_index(drop=True)
    if sighting_filter:
        filtered_queue = filtered_queue[
            filtered_queue["encounter_id"].astype(str) == sighting_filter.strip()
        ].reset_index(drop=True)
    if query_search:
        filtered_queue = filtered_queue[
            filtered_queue["image_id"].astype(str) == query_search.strip()
        ].reset_index(drop=True)

    if filtered_queue.empty:
        st.info("No queries match the current filters.")
        return
    filtered_queue = sorted_dataframe(
        filtered_queue,
        "review",
        {
            "image_id", "encounter_id", "status",
            "selected_ele_id", "predicted_seek", "corrected_seek",
        },
    )

    review_page_size = 25
    page_count = max(1, math.ceil(len(filtered_queue) / review_page_size))
    st.session_state["review_page_index"] = min(st.session_state["review_page_index"], page_count - 1)
    start = st.session_state["review_page_index"] * review_page_size
    end = min(start + review_page_size, len(filtered_queue))
    page_queue = filtered_queue.iloc[start:end].reset_index(drop=True)
    with st.container(border=True):
        table_cols = st.columns([1, 1.1, 0.45, 0.35, 0.45])
        table_cols[0].markdown("<div class='section-title'>Database</div>", unsafe_allow_html=True)
        table_cols[1].caption(f"{start + 1}-{end} of {len(filtered_queue)}")
        with table_cols[2]:
            if st.button("‹", disabled=st.session_state["review_page_index"] == 0, key="review_prev"):
                st.session_state["review_page_index"] -= 1
                st.rerun()
        table_cols[3].caption(f"{st.session_state['review_page_index'] + 1}/{page_count}")
        with table_cols[4]:
            if st.button("›", disabled=st.session_state["review_page_index"] >= page_count - 1, key="review_next"):
                st.session_state["review_page_index"] += 1
                st.rerun()
        render_compact_query_table(page_queue, "review", "Query Review")

    selected = st.session_state.get("review_query_id", str(page_queue.iloc[0]["query_id"]))
    if selected not in set(filtered_queue["query_id"].astype(str)):
        selected = str(page_queue.iloc[0]["query_id"])
        st.session_state["review_query_id"] = selected

    if selected not in set(queue["query_id"].astype(str)):
        selected = str(queue.iloc[0]["query_id"])
        st.session_state["review_query_id"] = selected
    meta = selected_query_metadata(selected)
    predicted = cached_query_prediction(selected)
    current_seek = meta.get("current_query_seek") or predicted or "B00T__E0000-0000X00S00"
    candidate_limit_key = f"candidate_limit:{selected}"
    if candidate_limit_key not in st.session_state:
        st.session_state[candidate_limit_key] = 10
    candidate_limit = min(int(st.session_state[candidate_limit_key]), QUERY_REVIEW_RANK_CACHE_TOP_K)
    prerank = get_cached_prerank(selected, predicted, QUERY_REVIEW_RANK_CACHE_TOP_K)
    ranking = st.session_state.get(f"ranking:{selected}", [])
    ranking_source = "Rerank from edited SEEK"
    if not ranking:
        ranking = prerank
        ranking_source = "Pre-rank from predicted SEEK"

    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                f"<div class='section-title'>Query {html.escape(str(meta['subject_id'] or selected))}</div>",
                unsafe_allow_html=True,
            )
            status = queue[queue["query_id"] == selected].iloc[0]["status"]
            st.markdown(status_badge_html(status), unsafe_allow_html=True)
            display_query_images(meta)

            edited_seek, valid = render_seek_editor(current_seek, key_prefix=f"query_seek_{selected}")
            with st.expander("Cheat sheet", expanded=False):
                st.caption(f"true elephant ID: `{meta['ele_id'] or '-'}`")
                st.caption(f"image-level SEEK: `{meta['subject_seek'] or '-'}`")
                st.caption(f"sighting-level SEEK: `{meta['ele_seek'] or '-'}`")
            sighting_id = str(meta["encounter_id"] or "")
            sighting_available = bool(sighting_id) and len(
                queue[queue["encounter_id"].astype(str) == sighting_id]
            ) > 1
            scope_options = ["image-level", "sighting-level"] if sighting_available else ["image-level"]
            default_scope = "sighting-level" if sighting_available else "image-level"
            st.markdown("<div class='section-title'>Correction</div>", unsafe_allow_html=True)
            correction_cols = st.columns([1.45, 0.55])
            with correction_cols[0]:
                query_correction_scope = st.segmented_control(
                    "Correction scope",
                    scope_options,
                    default=default_scope,
                    key=f"query_scope_{selected}",
                    label_visibility="collapsed",
                )
            with correction_cols[1]:
                apply_correction = st.button(
                    "Apply correction",
                    type="primary",
                    disabled=not valid,
                    key=f"apply_query_correction_{selected}",
                    use_container_width=True,
                )
            if not sighting_available:
                st.caption("sighting-level unavailable for this query")
            query_scope_code = {
                "image-level": "image",
                "sighting-level": "elephant_sighting",
            }[query_correction_scope]

            if apply_correction:
                targets = [selected]
                if query_scope_code == "elephant_sighting":
                    targets = queue[
                        queue["encounter_id"].astype(str) == sighting_id
                    ]["query_id"].astype(str).tolist()
                for target_query_id in targets:
                    target_meta = selected_query_metadata(target_query_id)
                    state.log_seek_correction(
                        item_type="query",
                        query_id=target_query_id,
                        subject_id=target_meta["subject_id"],
                        ele_id=target_meta["ele_id"],
                        encounter_id=target_meta["encounter_id"],
                        original_subject_seek=target_meta["subject_seek"],
                        original_ele_seek=target_meta["ele_seek"],
                        predicted_seek=state.query_predictions().get(target_query_id, ""),
                        corrected_seek=edited_seek,
                        scope=query_scope_code,
                    )
                    if target_query_id.startswith("dataset:"):
                        model.warm_dataset_query_features(target_meta["idx"], edited_seek)
                    else:
                        model.warm_upload_query_features(target_meta["upload_row"], edited_seek)
                run_ranking(selected, meta, edited_seek, QUERY_REVIEW_RANK_CACHE_TOP_K)
                st.success("Applied query SEEK correction.")

            identity_input_key = f"identity_ele_id:{selected}"
            pending_identity = st.session_state.pop(f"pending_identity_ele_id:{selected}", None)
            if pending_identity is not None:
                st.session_state[identity_input_key] = pending_identity
            elif identity_input_key not in st.session_state:
                current_decision = state.query_statuses().get(selected, {})
                st.session_state[identity_input_key] = current_decision.get("selected_ele_id", "")
            identity_ele_id = st.session_state.get(identity_input_key, "").strip()
            pending_action = st.session_state.pop(f"pending_decision_action:{selected}", None)
            pending_choose = st.session_state.pop(f"pending_choose_identity:{selected}", None)
            if pending_action:
                state.log_decision(selected, meta["subject_id"], pending_action)
                st.success("Marked uncertain." if pending_action == "uncertain" else "Rejected.")
                st.rerun()
            if pending_choose:
                matched = find_candidate_by_ele_id(ranking, pending_choose)
                state.log_decision(
                    query_id=selected,
                    subject_id=meta["subject_id"],
                    decision_type="matched",
                    selected_ele_id=pending_choose,
                    selected_subject_id=matched.get("subject_id", "") if matched else "",
                )
                st.success(f"Matched to elephant {pending_choose}")
                st.rerun()

            st.markdown("<div class='section-title'>Decision</div>", unsafe_allow_html=True)
            identity_cols = st.columns(4, gap="small")
            with identity_cols[0]:
                st.markdown(
                    f"<a class='decision-action decision-uncertain' target='_self' "
                    f"href='{decision_action_href(selected, 'uncertain')}'>Uncertain</a>",
                    unsafe_allow_html=True,
                )
            with identity_cols[1]:
                st.markdown(
                    f"<a class='decision-action decision-reject' target='_self' "
                    f"href='{decision_action_href(selected, 'rejected')}'>Reject</a>",
                    unsafe_allow_html=True,
                )
            identity_ele_id = identity_cols[2].text_input(
                "Elephant ID",
                key=identity_input_key,
                placeholder="Elephant ID",
                label_visibility="collapsed",
            ).strip()
            choose_class = "decision-action decision-choose" if identity_ele_id else "decision-action decision-disabled"
            choose_href = decision_action_href(selected, "choose", identity_ele_id) if identity_ele_id else "#"
            with identity_cols[3]:
                st.markdown(
                    f"<a class='{choose_class}' target='_self' href='{choose_href}'>Choose</a>",
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Candidates</div>", unsafe_allow_html=True)
            st.caption(ranking_source)
            if not ranking:
                st.info("No cached pre-rank for this query yet. Precompute it from Query Queue or upload ingestion.")
            st.session_state[f"active_query_seek:{selected}"] = edited_seek
            for candidate in ranking[:candidate_limit]:
                render_candidate(candidate, selected)
            if candidate_limit < min(len(ranking), QUERY_REVIEW_RANK_CACHE_TOP_K) and st.button(
                "Show 10 more",
                key=f"show_more_{selected}",
            ):
                st.session_state[candidate_limit_key] = candidate_limit + 10
                st.rerun()
            compare_candidate = st.session_state.get(f"compare:{selected}")
            if compare_candidate:
                render_comparison_dialog(meta, compare_candidate)


def query_queue_page():
    consume_query_query_params()
    if "queue_page_index" not in st.session_state:
        st.session_state["queue_page_index"] = 0
    with st.container(border=True):
        st.markdown("<div class='section-title'>Upload</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
        if uploaded is not None and st.button("Add upload"):
            row = preprocess_upload(uploaded, state)
            st.success(f"Added {row['query_id']} and queued background precompute.")

    queue = all_queue_rows()
    with st.container(border=True):
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        statuses = sorted(queue["status"].unique().tolist())
        filter_cols = st.columns([1, 1, 1, 1])
        status_filter = filter_cols[0].multiselect("Status", statuses, default=statuses, key="queue_status")
        subject_filter = filter_cols[1].text_input("Subject ID", placeholder="exact", key="queue_subject")
        sighting_filter = filter_cols[2].text_input("Sighting ID", placeholder="exact", key="queue_sighting")
        query_search = filter_cols[3].text_input("Image ID", placeholder="exact", key="queue_image")
    filter_state = (tuple(status_filter), subject_filter, sighting_filter, query_search)
    last_filter_state = st.session_state.get("queue_last_filter_state")
    if last_filter_state is None:
        st.session_state["queue_last_filter_state"] = filter_state
    elif last_filter_state != filter_state:
        st.session_state["queue_last_filter_state"] = filter_state
        st.session_state["queue_page_index"] = 0
    if status_filter:
        filtered = queue[queue["status"].isin(status_filter)].reset_index(drop=True)
    else:
        filtered = queue.reset_index(drop=True)
    if subject_filter:
        filtered = filtered[filtered["subject_id"].astype(str) == subject_filter.strip()].reset_index(drop=True)
    if sighting_filter:
        filtered = filtered[filtered["encounter_id"].astype(str) == sighting_filter.strip()].reset_index(drop=True)
    if query_search:
        filtered = filtered[
            filtered["image_id"].astype(str) == query_search.strip()
        ].reset_index(drop=True)
    if filtered.empty:
        st.info("No queries match the current filters.")
        return
    if st.session_state.get("queue_sort_col"):
        filtered = sorted_dataframe(
            filtered,
            "queue",
            {
                "image_id", "encounter_id", "status",
                "selected_ele_id", "predicted_seek", "corrected_seek",
            },
        )
    else:
        filtered = random_sighting_group_sort(filtered)
    page_size = 25
    page_count = max(1, math.ceil(len(filtered) / page_size))
    st.session_state["queue_page_index"] = min(st.session_state["queue_page_index"], page_count - 1)
    start = st.session_state["queue_page_index"] * page_size
    end = min(start + page_size, len(filtered))
    page_filtered = filtered.iloc[start:end].reset_index(drop=True)
    with st.container(border=True):
        table_cols = st.columns([1, 1.1, 0.45, 0.35, 0.45])
        table_cols[0].markdown("<div class='section-title'>Database</div>", unsafe_allow_html=True)
        table_cols[1].caption(f"{start + 1}-{end} of {len(filtered)}")
        with table_cols[2]:
            if st.button("‹", disabled=st.session_state["queue_page_index"] == 0, key="queue_prev"):
                st.session_state["queue_page_index"] -= 1
                st.rerun()
        table_cols[3].caption(f"{st.session_state['queue_page_index'] + 1}/{page_count}")
        with table_cols[4]:
            if st.button("›", disabled=st.session_state["queue_page_index"] >= page_count - 1, key="queue_next"):
                st.session_state["queue_page_index"] += 1
                st.rerun()
        render_compact_query_table(page_filtered, "queue", "Query Queue")


def gallery_page():
    consume_gallery_query_params()
    gallery = cached_gallery_dataframe(
        catalog,
        state,
        state_file_fingerprint(GALLERY_TABLE_STATE_FILES),
    )
    if "gallery_page_index" not in st.session_state:
        st.session_state["gallery_page_index"] = 0
    if "gallery_selected_idx" not in st.session_state:
        st.session_state["gallery_selected_idx"] = None

    with st.container(border=True):
        st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
        filter_cols = st.columns([1, 1, 1])
        ele_filter = filter_cols[0].text_input(
            "Elephant ID",
            key="gallery_ele_filter",
            placeholder="exact",
        ).strip()
        subject_filter = filter_cols[1].text_input(
            "Image ID",
            key="gallery_subject_filter",
            placeholder="exact",
        ).strip()
        sighting_filter = filter_cols[2].text_input(
            "Sighting ID",
            key="gallery_sighting_filter",
            placeholder="exact",
        ).strip()
        page_size = 25
    filter_state = (ele_filter, subject_filter, sighting_filter)
    last_filter_state = st.session_state.get("gallery_last_filter_state")
    if last_filter_state is None:
        st.session_state["gallery_last_filter_state"] = filter_state
    elif last_filter_state != filter_state:
        st.session_state["gallery_last_filter_state"] = filter_state
        st.session_state["gallery_selected_idx"] = None
        st.session_state["gallery_page_index"] = 0

    if ele_filter:
        gallery = gallery[gallery["ele_id"].astype(str) == ele_filter]
    if subject_filter:
        gallery = gallery[gallery["subject_id"].astype(str) == subject_filter]
    if sighting_filter:
        gallery = gallery[gallery["encounter_id"].astype(str) == sighting_filter]

    if gallery.empty:
        st.info("No gallery images match the current filters.")
        return
    gallery = gallery.reset_index(drop=True)
    gallery = sorted_dataframe(
        gallery,
        "gallery",
        {
            "subject_id", "ele_id", "encounter_id", "picture_time",
            "corrected_seek", "subject_seek", "ele_seek",
        },
    )
    page_count = max(1, math.ceil(len(gallery) / page_size))
    st.session_state["gallery_page_index"] = min(st.session_state["gallery_page_index"], page_count - 1)

    start = st.session_state["gallery_page_index"] * page_size
    end = min(start + page_size, len(gallery))
    page_gallery = gallery.iloc[start:end].reset_index(drop=True)

    with st.container(border=True):
        table_cols = st.columns([1, 1.1, 0.45, 0.35, 0.45])
        table_cols[0].markdown("<div class='section-title'>Database</div>", unsafe_allow_html=True)
        table_cols[1].caption(f"{start + 1}-{end} of {len(gallery)}")
        with table_cols[2]:
            if st.button("‹", disabled=st.session_state["gallery_page_index"] == 0, key="gallery_prev"):
                st.session_state["gallery_page_index"] -= 1
                st.rerun()
        table_cols[3].caption(f"{st.session_state['gallery_page_index'] + 1}/{page_count}")
        with table_cols[4]:
            if st.button("›", disabled=st.session_state["gallery_page_index"] >= page_count - 1, key="gallery_next"):
                st.session_state["gallery_page_index"] += 1
                st.rerun()
        render_compact_gallery_table(page_gallery)

    selected_idx = st.session_state.get("gallery_selected_idx")
    selected_in_filter = (
        selected_idx is not None
        and not gallery[gallery["idx"].astype(int) == int(selected_idx)].empty
    )
    with st.container(border=True):
        st.markdown("<div class='section-title'>Correction</div>", unsafe_allow_html=True)
        if selected_in_filter:
            meta = catalog.metadata_for_idx(selected_idx, state=state, item_type="gallery")
            st.markdown(
                gallery_correction_badge_html(bool(meta["corrected_seek"])),
                unsafe_allow_html=True,
            )
            display_query_images(meta, use_display=True)
            st.caption(f"image-level SEEK: `{meta['subject_seek']}`")
            st.caption(f"sighting-level SEEK: `{meta['ele_seek']}`")
            st.caption(f"Corrected SEEK: `{meta['corrected_seek'] or '-'}`")
            edited_seek, valid = render_seek_editor(meta["current_ele_seek"], key_prefix=f"gallery_seek_{selected_idx}")
            correction_scope = st.segmented_control(
                "Apply correction",
                [
                    "this image only",
                    "all images of this elephant",
                    "all images in this sighting",
                    "all images of this elephant in this sighting",
                ],
                default="all images of this elephant in this sighting",
                key=f"gallery_scope_{selected_idx}",
            )
            scope_code = {
                "this image only": "image",
                "all images of this elephant": "elephant",
                "all images in this sighting": "sighting",
                "all images of this elephant in this sighting": "elephant_sighting",
            }[correction_scope]
            cols = st.columns(2)
            with cols[0]:
                if st.button("Save gallery SEEK", disabled=not valid):
                    affected_idxs = affected_gallery_idxs(meta, scope_code)
                    state.log_seek_correction(
                        item_type="gallery",
                        query_id="",
                        subject_id=meta["subject_id"],
                        ele_id=meta["ele_id"],
                        encounter_id=meta["encounter_id"],
                        original_subject_seek=meta["subject_seek"],
                        original_ele_seek=meta["ele_seek"],
                        predicted_seek="",
                        corrected_seek=edited_seek,
                        scope=scope_code,
                    )
                    model.update_gallery_cache_for_idxs(affected_idxs)
                    st.success("Saved gallery SEEK correction.")
                    st.rerun()
            with cols[1]:
                if st.button("Delete from UI gallery"):
                    state.hide_gallery_image(meta["subject_id"], meta["encounter_id"])
                    model.refresh_gallery_cache(force=True)
                    st.warning("Removed from UI gallery and retrieval.")
                    st.rerun()
        elif ele_filter or sighting_filter or subject_filter:
            display_gallery_image_grid(gallery)
        else:
            st.info("Click an Image ID to inspect crops.")


def main():
    st.title("Elephant Re-ID Expert Interface")
    st.caption(f"Projector checkpoint: `{DEFAULT_EXPERIMENT_DIR}`")
    query_worker = get_query_precompute_worker()
    query_worker.start()
    worker_status = query_worker.snapshot()
    st.caption(f"Background query cache: {worker_status['state']} · {worker_status['message']}")
    pages = ["Query Review", "Query Queue", "Gallery"]
    query_view = st.query_params.get("view")
    if query_view in pages:
        st.session_state["active_view"] = query_view
    consume_sort_query_params()
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Query Review"
    page = st.segmented_control(
        "View",
        pages,
        default=st.session_state["active_view"],
        key="active_view",
        label_visibility="collapsed",
    )
    page = page or st.session_state["active_view"]
    if page == "Query Review":
        query_review_page()
    elif page == "Query Queue":
        query_queue_page()
    else:
        gallery_page()


if __name__ == "__main__":
    main()
