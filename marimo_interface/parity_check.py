from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "user_interface", ROOT / "marimo_interface"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from marimo_interface.config import DEFAULT_EXPERIMENT_DIR  # noqa: E402
from marimo_interface.data import DataCatalog as MarimoCatalog  # noqa: E402
from marimo_interface.model_service import ModelService as MarimoModelService  # noqa: E402
from marimo_interface.sqlite_state import SQLiteState  # noqa: E402
from user_interface.data import DataCatalog as StreamlitCatalog  # noqa: E402
from user_interface.model_service import ModelService as StreamlitModelService  # noqa: E402
from user_interface.state import UIState  # noqa: E402


def compare_dataframes():
    streamlit_state = UIState()
    streamlit_catalog = StreamlitCatalog()
    marimo_state = SQLiteState()
    marimo_catalog = MarimoCatalog()

    streamlit_queue = streamlit_catalog.queue_dataframe(streamlit_state)
    marimo_queue = marimo_catalog.queue_dataframe(marimo_state)
    streamlit_gallery = streamlit_catalog.gallery_dataframe(streamlit_state)
    marimo_gallery = marimo_catalog.gallery_dataframe(marimo_state)

    checks = {
        "train_indices_equal": streamlit_catalog.train_indices == marimo_catalog.train_indices,
        "test_indices_equal": streamlit_catalog.test_indices == marimo_catalog.test_indices,
        "queue_row_count_equal": len(streamlit_queue) == len(marimo_queue),
        "gallery_row_count_equal": len(streamlit_gallery) == len(marimo_gallery),
        "queue_first_25_ids_equal": (
            streamlit_queue["query_id"].head(25).astype(str).tolist()
            == marimo_queue["query_id"].head(25).astype(str).tolist()
        ),
        "gallery_first_25_subjects_equal": (
            streamlit_gallery["subject_id"].head(25).astype(str).tolist()
            == marimo_gallery["subject_id"].head(25).astype(str).tolist()
        ),
        "queue_columns_equal": list(streamlit_queue.columns) == list(marimo_queue.columns),
        "gallery_columns_equal": list(streamlit_gallery.columns) == list(marimo_gallery.columns),
    }
    summary = {
        "streamlit_queue_rows": len(streamlit_queue),
        "marimo_queue_rows": len(marimo_queue),
        "streamlit_gallery_rows": len(streamlit_gallery),
        "marimo_gallery_rows": len(marimo_gallery),
        "checks": checks,
    }
    return summary


def compare_model(idx: int, top_k: int):
    streamlit_state = UIState()
    streamlit_catalog = StreamlitCatalog()
    marimo_state = SQLiteState()
    marimo_catalog = MarimoCatalog()
    streamlit_model = StreamlitModelService(streamlit_catalog, streamlit_state, experiment_dir=DEFAULT_EXPERIMENT_DIR)
    marimo_model = MarimoModelService(marimo_catalog, marimo_state, experiment_dir=DEFAULT_EXPERIMENT_DIR)

    streamlit_seek = streamlit_model.predict_seek_for_dataset_idx(idx)
    marimo_seek = marimo_model.predict_seek_for_dataset_idx(idx)
    streamlit_rank = streamlit_model.rank_dataset_query(idx, streamlit_seek, top_k=top_k)
    marimo_rank = marimo_model.rank_dataset_query(idx, marimo_seek, top_k=top_k)

    return {
        "idx": idx,
        "top_k": top_k,
        "predicted_seek_equal": streamlit_seek == marimo_seek,
        "streamlit_predicted_seek": streamlit_seek,
        "marimo_predicted_seek": marimo_seek,
        "rank_subjects_equal": (
            [str(row["subject_id"]) for row in streamlit_rank]
            == [str(row["subject_id"]) for row in marimo_rank]
        ),
        "rank_elephants_equal": (
            [str(row["ele_id"]) for row in streamlit_rank]
            == [str(row["ele_id"]) for row in marimo_rank]
        ),
        "streamlit_rank": streamlit_rank,
        "marimo_rank": marimo_rank,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Streamlit and marimo UI behavior.")
    parser.add_argument("--model", action="store_true", help="Also run expensive model inference/ranking parity.")
    parser.add_argument("--idx", type=int, default=None, help="Dataset idx for --model. Defaults to first test idx.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    summary = compare_dataframes()
    if args.model:
        idx = args.idx
        if idx is None:
            idx = MarimoCatalog().test_indices[0]
        summary["model"] = compare_model(idx, args.top_k)
    print(json.dumps(summary, indent=2))
    failed = []
    for name, passed in summary["checks"].items():
        if not passed:
            failed.append(name)
    if args.model:
        model_checks = summary["model"]
        if not model_checks["predicted_seek_equal"]:
            failed.append("model.predicted_seek_equal")
        if not model_checks["rank_subjects_equal"]:
            failed.append("model.rank_subjects_equal")
        if not model_checks["rank_elephants_equal"]:
            failed.append("model.rank_elephants_equal")
    if failed:
        raise SystemExit("Failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()

