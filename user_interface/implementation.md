# UI MVP Implementation Notes

## Current Scope

- Streamlit app under `user_interface/`.
- CSV state under `user_interface/state/`.
- Debug query queue from `weights/test_indicies_gold.txt`.
- Gallery from `weights/train_indices_gold.txt`.
- CHAIR projector retrieval, defaulting to
  `../experiments/exp_Heatmap_three_100cor`.

## Progress

- [x] Simplified `features.md` around concrete model constraints and user answers.
- [x] Create Streamlit scaffold.
- [x] Add CSV state helpers.
- [x] Add dataset/query/gallery loading helpers.
- [x] Add SEEK editor helpers.
- [x] Add model retrieval wrapper.
- [x] Add upload preprocessing path using Mara-style body crop + YOLO ear crops.
- [x] Add UI-local catalog copy under `user_interface/state/catalog.csv`.
- [x] Add inspection-only RGB display crop cache.
- [x] Add gallery row selection and status/correction badges.
- [x] Replace Streamlit tabs with a single active view to avoid computing hidden pages.
- [x] Add explicit query SEEK prediction button instead of predicting on page load.
- [x] Paginate gallery table rendering after filters.
- [x] Compact image inspection to body crop plus left/right ear crops.
- [x] Compact SEEK editor to two rows of selectors.
- [x] Replace gallery dataframe with a 25-row clickable table.
- [x] Add exact elephant, image, and sighting filters.
- [x] Add image/elephant/elephant-sighting correction scopes.
- [x] Add raw-color display crop precompute script.
- [x] Compact gallery rows into an HTML table so 25 rows fit without page scroll.
- [x] Replace main page radio with segmented page selection.
- [x] Split Gallery into top controls, table, and corrections/inspection sections.
- [x] Apply compact 25-row table and page controls to Query Queue.
- [x] Remove ground-truth SEEK and duplicated identifiers from Query Queue.
- [x] Add Query Review cheat sheet toggle for debug-only ground-truth SEEK.
- [x] Add subject ID and sighting ID filters to Query Queue on one line.
- [x] Cache query concept-head predictions in `query_predictions.csv`.
- [x] Auto-compute missing predicted SEEK for visible Query Queue rows.
- [x] Restore Gallery table to compact Query Queue-style HTML layout with same-tab
  click handling for image, elephant, and sighting IDs.
- [x] Render inspection/ranking images from raw-color display crops with equal
  display height.
- [x] Verify syntax and non-Streamlit helper imports.
- [x] Verify Streamlit app startup.
- [x] Split Query Review into the same section style as the other pages.
- [x] Reuse the Query Queue compact database table and pager in Query Review.
- [x] Add sortable clickable headers to Gallery, Query Queue, and Query Review
  database tables.
- [x] Replace Gallery iframe clicks with same-tab query-parameter links.
- [x] Make Query Review predict SEEK by default and remove the manual predict
  button.
- [x] Compact Query Review SEEK editor labels and raw-code input.
- [x] Replace Query Review save/rerank controls with one apply-correction action.
- [x] Render candidates as full-width visual cards with metadata chips and
  highlighted SEEK character agreement.
- [x] Cache 100 Query Review candidates while revealing them 10 at a time.
- [x] Move large candidate comparison into a modal dialog.
- [x] Make candidate `Select` fill the Decision elephant ID field while candidate
  elephant/sighting links open Gallery filters separately.
- [x] Default Query Queue ordering groups images by sighting with deterministic
  random sighting order.

## Implementation Decisions

- The UI will never write to `data/` or `methods/`.
- "Delete" means "hide from UI retrieval" via CSV state.
- Gallery SEEK correction is encounter-level.
- Query SEEK correction is image-level.
- Ranking logs store every rerank as JSON inside a CSV row.
- Missing ear crops are omitted visually but represented as zero tensors for the model.
- The model service uses direct corrected SEEK one-hot vectors for projector inference,
  matching the expert-in-the-loop behavior where correction replaces the concept-head
  prediction.
- Gallery embeddings are cached in the Streamlit model service and refreshed after
  gallery SEEK edits or UI deletions.
- Upload preprocessing preserves the raw photo and stores derived model-facing crops
  under `user_interface/state/uploads/`.
- Uploaded images without a supplied Mara-style bbox use the full image as the body
  crop. Ear crops are detected from that body crop with `weights/ear_YOLOv5_n.pt`.
- The UI reads from `user_interface/state/catalog.csv`, which is copied from the
  original Mara catalog only when the UI copy is missing.
- Visual inspection uses raw-color display crops. These are cached separately from
  the model-facing normalized crops and are not used for inference.
- Model inference never uses display crops. Dataset inference uses the existing
  dataset tensors, and upload inference uses the once-created model-facing upload
  crops.
- Gallery inspection is driven by row selection in the gallery table.
- Query statuses and gallery SEEK corrections are surfaced as compact badges.
- Only the active view is executed. This avoids loading/predicting the model while
  browsing the gallery.
- Query Review predicts SEEK by default for the selected query. Query Queue still
  caches visible predictions in batches.
- Gallery tables are paginated to keep browser rendering bounded.
- Gallery table uses `subject_id` as `Image ID`; raw `idx` remains internal.
- Gallery table filters are exact matches, not substring matches.
- Gallery correction priority is latest matching correction in log order. Supported
  scopes are image, elephant, and elephant_sighting.
- `subject-SEEK` is displayed as `Image-level SEEK`.
- `ele-SEEK` is displayed as `Sighting-level SEEK`.
- `current_ele_seek` is displayed as `Corrected SEEK` only when a correction exists.
- Display crops can be precomputed with:
  `python -m user_interface.precompute_display_crops --split gallery`
- Gallery page size is fixed at 25 rows to keep the table predictable.
- Gallery row clicks use lightweight same-tab query-parameter links instead of an
  iframe component.
- Database table headers toggle ascending/descending sort before pagination.
- Query Queue and Query Review show `Image ID`, `Subject ID`, `Sighting ID`,
  status, matched decision, predicted SEEK, and corrected SEEK. Dataset `idx` and
  internal `query_id` are not shown.
- Empty Query Queue status filter means show all statuses.
- Query Queue filters are on one line: status, subject ID, sighting ID, image ID.
- Ground-truth query `subject-SEEK` and `ele-SEEK` are hidden behind the Query Review
  `Cheat sheet` expander, which opens client-side without a Streamlit rerun.
- The background query worker fills missing concept-head predictions and stores them
  in `user_interface/state/query_predictions.csv`.
- Query Review uses the same compact query database table as Query Queue, so image
  selection, columns, row density, and pager behavior stay consistent.
- Query Review correction scope is image-level or sighting-level. Sighting-level is
  only offered when multiple query images share the selected sighting.
- Query Review labels the SEEK correction section `Correction` and places the
  `Apply correction` button on the same row as the image-level/sighting-level scope
  selector.
- Query Review has a compact `Decision` row: Uncertain, Reject, Elephant ID, and
  Choose. Decision controls use solid button styling with white text.
- Candidate `Select` buttons fill the identity field. Candidate elephant and
  sighting links open Gallery filters in a new tab so the active Query Review is
  preserved.
- Candidate metadata controls use one fixed four-column chip row so rank,
  elephant, sighting, and Select stay aligned and equally sized.
- Query Queue and Query Review Image ID filters are exact matches.
- SEEK strings are canonicalized to the 22-character `SEEK.__str__` form before
  display/comparison so raw Mara trailing characters do not misalign candidate
  highlights against the editor value.
- Query Review displays 10 candidates by default and increases the rank depth in
  chunks of 10.
- Query Review rank cache depth is 100. The model service currently returns a full
  ranking list per call, so first-time uncached reranks are not true streaming.
- Large comparison uses a modal dialog with query and candidate crop triplets shown
  side by side.
- Query Queue defaults to grouping rows by sighting ID with deterministic random
  sighting ordering unless the user selects an explicit table sort.
- Query Queue, Query Review, and Gallery cache their assembled table dataframes
  using CSV file fingerprints, so page navigation and row selection avoid rebuilding
  the full tables unless backing state changes.
- Same-tab row and sort links preserve table page indices; only real filter changes
  reset a table to page 1.

## Known Risks

- Model loading is GPU-heavy and may make Streamlit startup slow on CPU-only machines.
- First upload preprocessing call may be slow because it lazy-loads the YOLOv5 ear
  detector.
- The first inspection of a dataset image may be slow because raw-color display ear
  crops are generated lazily.
- Increasing gallery page size above 500 may make Streamlit table rendering slower.
- Precomputing all gallery RGB crops may take a long time because YOLO ear detection
  runs once per image without an existing cache.

## Verification

- `python -m py_compile user_interface/*.py` passes.
- `python -m streamlit --version` reports Streamlit 1.40.2.
- Temporary Streamlit server starts and returns HTTP 200 / health `ok`.
- Direct `import user_interface.app` succeeds; Streamlit bare-mode warnings are
  expected outside `streamlit run`.
- Dataset smoke test loads 16,506 query rows from the gold test split.
- Gallery smoke test loads 19,112 gallery rows from the gold train split.
- SEEK normalization works on the current Mara code format with the extra trailing
  character.
- `ModelService` imports without eagerly loading model checkpoints.
