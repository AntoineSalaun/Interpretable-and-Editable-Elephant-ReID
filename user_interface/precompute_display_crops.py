from __future__ import annotations

import argparse
import warnings
from pathlib import Path

try:
    from .data import DataCatalog
    from .state import UIState
except ImportError:
    from data import DataCatalog
    from state import UIState


def precompute(split, limit=None, force=False):
    catalog = DataCatalog()
    state = UIState()
    if split == "gallery":
        rows = catalog.gallery_dataframe(state)
    elif split == "query":
        rows = catalog.queue_dataframe(state)
    else:
        rows = catalog.df.copy().assign(idx=range(len(catalog.df)))

    done = 0
    skipped = 0
    failed = 0
    for _, row in rows.iterrows():
        idx = int(row["idx"])
        output_dir = catalog.display_crop_dir_for_idx(idx)
        if force or not (output_dir / "body_rgb.jpg").exists():
            try:
                catalog.display_image_paths(catalog.row(idx), force=force)
                done += 1
            except Exception as exc:
                failed += 1
                print(f"failed idx={idx}: {exc}")
        else:
            skipped += 1
        if limit is not None and done >= limit:
            break
        if (done + skipped + failed) % 100 == 0:
            print(f"processed={done} skipped={skipped} failed={failed}")
    print(f"processed={done} skipped={skipped} failed={failed}")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="Precompute RGB display crops for the Streamlit UI.")
    parser.add_argument("--split", choices=["gallery", "query", "all"], default="gallery")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    precompute(args.split, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
