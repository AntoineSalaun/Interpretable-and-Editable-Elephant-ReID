from __future__ import annotations

import argparse
import warnings

try:
    from .config import DEFAULT_EXPERIMENT_DIR, DEFAULT_TOP_K
    from .data import DataCatalog
    from .model_service import ModelService
    from .state import UIState
except ImportError:
    from config import DEFAULT_EXPERIMENT_DIR, DEFAULT_TOP_K
    from data import DataCatalog
    from model_service import ModelService
    from state import UIState


def precompute(limit=None, top_k=DEFAULT_TOP_K):
    warnings.filterwarnings("ignore", category=FutureWarning)
    state = UIState()
    catalog = DataCatalog()
    model = ModelService(catalog, state, experiment_dir=DEFAULT_EXPERIMENT_DIR)
    uploads = state.read("uploads.csv")
    predictions = state.query_predictions()

    done = 0
    skipped = 0
    failed = 0
    for query_id, seek_code in predictions.items():
        if state.cached_rank(query_id, seek_code, top_k) is not None:
            skipped += 1
            continue
        try:
            if query_id.startswith("dataset:"):
                idx = int(query_id.split(":", 1)[1])
                ranking = model.rank_dataset_query(idx, seek_code, top_k=top_k)
            else:
                upload_rows = uploads[uploads["query_id"] == query_id]
                if upload_rows.empty:
                    skipped += 1
                    continue
                ranking = model.rank_upload_query(upload_rows.iloc[-1].to_dict(), seek_code, top_k=top_k)
            state.log_rank_cache(query_id, seek_code, top_k, ranking)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"failed query_id={query_id}: {exc}", flush=True)
        if limit is not None and done >= limit:
            break
        if (done + skipped + failed) % 10 == 0:
            print(f"ranked={done} skipped={skipped} failed={failed}", flush=True)
    print(f"ranked={done} skipped={skipped} failed={failed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Precompute cached query pre-ranks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    precompute(limit=args.limit, top_k=args.top_k)


if __name__ == "__main__":
    main()
