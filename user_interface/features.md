# Elephant Re-Identification Expert Interface MVP

## Objective

Build a lightweight Streamlit interface for expert elephant re-identification.

The MVP should let an expert:

1. Review a query image.
2. Inspect the model-predicted SEEK code.
3. Correct the SEEK code using valid categorical values.
4. Re-rank gallery images with the CHAIR projector model.
5. Assign the query to an existing elephant ID, mark it uncertain, or reject it.

The model is frozen. The interface must not train models and must not modify files in
`data/` or `methods/`.

All mutable UI state lives in `user_interface/state/` as CSV files. On first run,
the app copies the Mara image dictionary into `user_interface/state/catalog.csv` and
uses that UI-local catalog thereafter.

## Model Assumptions

- The main retrieval model is the CHAIR projector.
- Default checkpoint: the best 100% sighting-correction heatmap run,
  `../experiments/exp_Heatmap_three_100cor`.
- The model uses:
  - full/body image crop,
  - left ear crop when available,
  - right ear crop when available,
  - a 63-dimensional SEEK one-hot vector.
- `sighting SEEK` and `ele-SEEK` mean the same thing in this UI.
- The gallery uses the train split from `weights/train_indices_gold.txt`.
- The debug query queue uses the test split from `weights/test_indicies_gold.txt`.
- Gallery retrieval SEEK is `ele-SEEK`, unless an expert sighting-level correction exists.
- `subject-SEEK` should still be visible as reference metadata.
- Query SEEK corrections are per image.
- Gallery SEEK corrections may apply to one image, all images of an elephant, or all
  images of an elephant in one sighting.
- If a gallery image is deleted in the UI, it is excluded from all UI views and retrieval,
  but the original data file is not touched.
- Model-facing crops may be normalized/preprocessed. Inspection views should prefer
  visual-only RGB crops cached under `user_interface/state/display_crops/`.

## MVP Pages

### Query Review

This is the main screen.

The expert can choose one query from the queue and see:

- body crop,
- left ear crop if available,
- right ear crop if available,
- subject-SEEK,
- ele-SEEK / sighting-SEEK,
- model-predicted SEEK after the expert asks for prediction,
- current editable SEEK.

The SEEK editor should expose all 16 valid categorical attributes using select boxes.
It should also show a raw SEEK string field for expert users. Saving is allowed only
when the SEEK code is valid.

After editing SEEK, the expert clicks `Re-rank`. The UI recomputes ranking with the
corrected SEEK replacing the concept-head prediction.

The right side shows top image matches. Duplicates from the same elephant are allowed.
Each candidate shows:

- rank,
- elephant ID,
- subject ID,
- similarity score,
- image,
- ear crops when available,
- subject-SEEK,
- current gallery ele-SEEK.

Candidate actions:

- open large side-by-side comparison,
- select candidate elephant ID,
- delete candidate/gallery image from UI retrieval.

Final decisions:

- `matched`: assign query to the selected elephant ID.
- `uncertain`: defer identity decision.
- `rejected`: reject the query image.

Matched queries remain visible in the query queue with their status.

### Query Queue

For debugging, the queue is populated from all raw images in the test split.

The queue shows:

- status,
- subject ID filter and sighting ID filter,
- predicted SEEK if it has been computed in the current session,
- corrected SEEK if an expert saved a query correction.

Predicted SEEK should be computed automatically for visible query rows and cached
under UI state so it is not recomputed on every rerun.

The debug test split still contains ground-truth metadata, but the queue should not
display ground-truth image-level or sighting-level SEEK. Query Review may show those
values only behind a small `Cheat sheet` toggle.

The MVP also accepts uploaded `.jpg`, `.jpeg`, and `.png` images. Uploaded images are
stored under `user_interface/state/uploads/`. The UI preprocessing path should produce
the model-facing body crop and ear crops before model inference. The raw uploaded
photo remains preserved as the source image. For uploaded images without an external
Mara-style elephant bbox, the body crop is the full image resized to 224. Ear crops
are detected with `weights/ear_YOLOv5_n.pt`. If an ear crop is missing, the UI simply
does not show it.

### Gallery Browser

The gallery is the train split.

The expert can:

- search by elephant ID,
- search by subject ID,
- filter hidden/deleted images out,
- click a table row to inspect images and ear crops,
- edit gallery ele-SEEK per encounter,
- delete images from the UI gallery/retrieval.

The gallery should show a visible correction badge for sightings whose SEEK code has
been edited. Query views should show a visible status badge for `pending`, `matched`,
`uncertain`, or `rejected`.

The gallery table is paginated after filtering so the browser does not render the
entire train split on every interaction. The default page shows 25 rows. Filters for
elephant ID, image ID, and sighting ID are exact-match filters.

Gallery column labels:

- `Image ID`: the image/subject identifier used to open crops.
- `Elephant ID`: click to filter exactly to that elephant.
- `Sighting ID`: click to filter exactly to that sighting.
- `Image Time`: human-readable capture time.
- `Image-level SEEK`: the catalog `subject-SEEK`.
- `Sighting-level SEEK`: the catalog `ele-SEEK`.
- `Corrected SEEK`: the active expert correction, if any.

When an elephant or sighting filter is active and no specific image is selected, the
gallery shows the crop triplets for every image in the filtered set.

This page should stay simple; the MVP does not need complex gallery management.

## Logging

Use CSV logs in `user_interface/state/`.

Required tables:

- `seek_corrections.csv`: query/gallery corrections.
- `ranking_logs.csv`: every ranking run, including the full Top-K list.
- `decisions.csv`: every final decision.
- `hidden_gallery_images.csv`: gallery images excluded from UI/retrieval.
- `uploads.csv`: uploaded image metadata.
- `catalog.csv`: UI-local copy of the original Mara image dictionary.

Raw-color display crops are cached under `user_interface/state/display_crops/`.

The UI does not need to log every click.

## Retrieval Interface

The UI-facing retrieval service should provide:

```python
predict_seek(query)
rank(query, seek_code, top_k)
refresh_gallery_cache()
```

The service may reuse code from `methods/`, but it must not modify files in `methods/`.

Gallery embeddings should be cached so reranking one query does not recompute the
whole gallery each time. When a gallery sighting SEEK correction or gallery deletion
happens, the cache should be refreshed.

## Out of Scope

- training or active learning,
- multi-user support,
- authentication,
- cloud deployment,
- consensus annotation workflows,
- new elephant ID creation.

For now, potentially new elephants should be marked `uncertain`.
