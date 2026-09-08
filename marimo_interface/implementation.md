# Marimo Interface Implementation Notes

## Scope

- Marimo app and support code live only under `marimo_interface/`.
- Mutable state lives under `marimo_interface/state/`.
- The Streamlit app and `user_interface/state/` CSVs are not modified.
- The app reads the same Mara catalog, train split, test split, SEEK utilities, and
  CHAIR/projector checkpoint defaults as the Streamlit interface.

## Implemented

- [x] Marimo notebook at `marimo_interface/app.py`.
- [x] SQLite state database at `marimo_interface/state/marimo_ui.sqlite3`.
- [x] Query Review, Query Queue, and Gallery tabs.
- [x] Categorical SEEK editing in all three tabs.
- [x] Native marimo tables for fast paging, sorting, and row selection.
- [x] Lazy selected-row crop rendering; full tables do not render image grids.
- [x] Initial navigation avoids YOLO display-crop generation and uses existing
  model-facing crop paths when display crops are not already cached.
- [x] Query Queue defaults to the same deterministic grouped-by-sighting ordering
  as the Streamlit interface.
- [x] Filter text inputs update immediately instead of waiting for debounced
  submission.
- [x] Batched concept-head prediction for visible query cache warming.
- [x] Matrix-multiply ranking through normalized `torch.matmul(q, g.T)`.
- [x] Persistent caches for concept-head predictions, backbone embeddings,
  SEEK projections, projected edited embeddings, rank caches, and display crops.
- [x] Upload ingestion with the same body/ear preprocessing behavior as Streamlit.
- [x] Gallery hide/delete as UI-only exclusion.
- [x] Parity script comparing Streamlit and marimo data/view behavior.

## Cache Semantics

- Display crops are generated under `marimo_interface/state/display_crops/`.
- Upload crops are generated under `marimo_interface/state/uploads/`.
- Concept-head predictions are stored in `query_predictions` and `feature_cache`.
- Backbone embeddings are stored once per dataset or upload image in
  `feature_cache`.
- Projected SEEK vectors are stored by SEEK code in `seek_projection_cache`.
- Final edited retrieval embeddings are stored by `(image_key, seek_code)` in
  `projected_embedding_cache`.
- Top-k rankings are stored by `(query_id, seek_code, top_k, gallery_version)` in
  `rank_cache`.
- Gallery SEEK edits and UI deletes change `gallery_version`, invalidating stale
  rank-cache lookups without deleting old rows.

## Running

```bash
python -m marimo run marimo_interface/app.py --no-sandbox
```

## Verification

```bash
python -m py_compile marimo_interface/*.py
uvx marimo check marimo_interface/app.py
python marimo_interface/parity_check.py
```

Optional model parity:

```bash
python marimo_interface/parity_check.py --model --top-k 5
```

The optional model parity check loads the GPU-heavy CHAIR/projector stack and may
take a long time on a cold cache.

## Browser Verification

- Chromium/Playwright smoke-tested the marimo app at `http://127.0.0.1:2718`.
- Query Review, Query Queue, and Gallery tabs render without marimo runtime errors.
- Query Queue exact Image ID filter updates immediately; `5` selects query `5`.
- Gallery exact Elephant ID filter updates immediately; `175` selects elephant `175`.
- Query Queue starts on the same grouped sighting batch as Streamlit, with query
  `34873` from sighting `4897`.
- Streamlit was smoke-tested separately at `http://127.0.0.1:8502`; Gallery and
  Query Queue render the same core database columns and split sizes.
