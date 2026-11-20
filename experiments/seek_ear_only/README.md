# Ear-only correction comparison for Section 4.2

The latest report presents **two comparisons**, both using per-image accuracy:
(1) eight corrections for random (**37.9%**), greedy oracle (**77.3%**), ear-only
(**72.6%**), and margin-ratio (**72.3%**); and (2) no correction (**15.3%**), full
correction (**80.8%**), ear-only (**72.6%**), and margin-ratio at eight (**72.3%**).
Ear-only versus margin-ratio differs by +0.2 pp (paired 95% CI: -0.2 to +0.7 pp),
which does not establish a statistically significant difference. Ear-only recovers
87.4% of the full-correction gain over no correction in the second comparison.
Run `python experiments/seek_ear_only/comparison_texts.py` to reproduce both
tables, the paired intervals, and the two one-decimal LaTeX paragraphs in
[`results/comparisons/`](results/comparisons/report.md). The main
[`results/report.md`](results/report.md) and `results/section_4_2.tex` present
these two comparisons; the sections below retain prior revisions for provenance.

The current requested metric is **top-1 retrieval accuracy across images**, with
each query image equally weighted. Reproduce it from the saved predictions with
`python experiments/seek_ear_only/image_weighted.py`. Results, paired sighting-bootstrap
intervals, configuration, command and the one-decimal LaTeX paragraph are in
[`results/image_weighted/`](results/image_weighted/report.md): eight random corrections
**37.9%** (mean over all 20 original runs, seeds 42--61), ear-only **72.6%**,
margin-ratio at eight corrections **72.3%**, and full correction **80.8%**.
Ear-only recovers **80.8%** of the improvement from eight random corrections to
full correction. The no-correction row has been replaced in the current table;
gains and recovered fractions now use random-at-eight as the baseline. The paired
sighting bootstrap additionally resamples the 20 random runs to include seed
uncertainty. No model or policy was retrained or retuned. The sections below document the earlier
sighting-averaged analysis retained for provenance.
The main [`results/report.md`](results/report.md) and
[`results/section_4_2.tex`](results/section_4_2.tex) now present this current
comparison with one decimal place. `python experiments/seek_ear_only/report.py`
refreshes those documents from the current saved statistics.

This experiment imports the preceding intervention-prioritisation pipeline from
`experiments/seek_intervention_top20/run_prioritization.py`. The trained
`Heatmap_three_20cor` checkpoint (`u63g7ldu`), all gallery and test indices, fixed
`sighting_0.2` gallery correction, visual features, learned mixing coefficient,
layer normalization, cosine similarity, and gallery-image ranking are reused.
No training, checkpoint selection, temperature fitting, or policy tuning occurs.

The completed run gives sighting-averaged Recall@1 of **16.5754% / 75.2589% /
82.5456%** for no / ear-only / full correction. Ear-only recovers **88.9545%**
of the full-correction improvement, while remaining 7.2867 percentage points
below full correction (paired 95% CI: 6.3305 to 8.2763 points below). See
[`results/legacy_sighting_averaged/report.md`](results/legacy_sighting_averaged/report.md)
for the historical contrasts and
[`results/legacy_sighting_averaged/section_4_2.tex`](results/legacy_sighting_averaged/section_4_2.tex)
for the historical draft.
All 46 saved policy traces from the original and raw-margin runs match both
endpoint hit vectors exactly, across all 16,506 images and 1,269 sightings.

The added **iterative margin-ratio at budget 8** row gives **74.1123%** Recall@1,
a **57.5369 pp** improvement over no correction (paired 95% CI: **55.6521 to
59.4624 pp**), recovering **87.2165%** of the full-correction gain. Relative to
ear-only at the same budget, its difference is **-1.1466 pp** (95% CI: **-1.9395
to -0.3990 pp**). These are sighting-averaged outcomes from the saved sequential
policy, with scores recomputed after each intervention. The policy chooses eight
distinct attributes per sighting and applies them to every image; every image
therefore has eight corrected attributes, with selection shared within a sighting.
The frozen lambda, temperature, checkpoint, gallery, and annotation source are
unchanged. `add_margin.py` checks query order, both endpoints, and exact agreement
with the independently verified raw-margin rerun before reusing budget 8.

## Correction granularity found in the implementation

There is a substantive distinction between the requested image-level correction
and the preceding experiment's correction source:

- `EleHandler.__getitem__` encodes `subject-SEEK` into batch element 4 (expert
  **image-level** annotation) and `ele-SEEK` into element 5 (expert
  **sighting-level** annotation).
- `collect_cache` preserves them separately as `subject_seeks` and
  `sighting_seeks`.
- `run_single_policy` and the verified raw-margin rerun both pass
  `query.sighting_seeks` into `copy_attribute_from_expert`. That function copies
  the selected complete categorical slice row by row. It does not average expert
  labels. The annotation source is nevertheless sighting-level `ele-SEEK`.
- The per-image `subject-SEEK` can contain an occluded `_` where `ele-SEEK`
  contains the sighting's known value. The two sources cannot be interchanged
  while claiming the same full-correction endpoint.

The runner therefore requires an explicit `--correction-source` (`sighting`,
`image`, or `both`). It always audits the actual preceding `ele-SEEK` endpoints
first. This separates a correction-source discrepancy from a numerical failure.
The baseline is shared if both sources are requested.

For the reported comparison, the user confirmed **preserve prior `ele-SEEK`
corrections**. Thus the final command uses `--correction-source sighting`;
image-level `subject-SEEK` is not used for the reported correction conditions.

The eight attributes are discovered by matching `[LR]_(hole|tear)_[12]` against
`SEEK.attribute_names` and verified against `SEEK.mappings`; no fixed column
indices are assumed. They are `R_tear_1`, `R_hole_1`, `R_tear_2`, `R_hole_2`,
`L_tear_1`, `L_hole_1`, `L_tear_2`, and `L_hole_2`. `ear_special` and the two
`extreme` attributes are not included. All categorical values, including `_`,
are copied exactly as encoded by the existing loader; no unknown is masked,
discarded, imputed, or replaced by another annotation source. Nonselected
attributes remain the original per-image hard concept-head predictions.

## Aggregation and uncertainty

The order is: hard per-image predictions; copy selected expert categorical
slices; project the resulting per-image SEEK vectors; apply the existing branch
layer normalization and learned visual/SEEK fusion; cosine retrieval against
every fixed gallery image; test the top gallery image's identity. Embeddings,
expert codes, and similarities are never averaged across a sighting before
retrieval. The preceding adaptive policy averages concept probabilities for
selection, but this predetermined heuristic needs no selection distribution.

The primary metric requested here averages image hits within each query sighting,
then averages those sighting means equally. This matches the preceding supporting
first-concept analysis. The preceding main curve instead averages all images
equally, so its printed values are not the sighting-weighted endpoints. The audit
checks exact per-image hits and the original image-weighted values at tolerance
`1e-6` percentage points, and reports both endpoints reaggregated by sighting.

Paired percentile 95% bootstrap intervals use 10,000 draws, seed 2026. Each draw
samples all query sightings with replacement and applies those same sampled
sightings to every condition. The intervals describe differences in the equal
sighting mean, conditional on the checkpoint and fixed gallery. They do not
include training or gallery uncertainty, and unlike the preceding main curve's
identity bootstrap, do not cluster multiple sightings of one identity together.
All three pairwise contrasts are exported. The fraction recovered is
`100 * (R_ear - R_none) / (R_full - R_none)`; a nonpositive denominator produces
JSON `null` with an explicit explanation. Reviewed-attribute counts 0, 8, and 16
are counts, not measured annotation time.

## Execution and artifacts

Run from the repository root in the existing CUDA environment:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg \
  python -u experiments/seek_ear_only/run.py --correction-source sighting \
  --train-indices /data/vision/beery/scratch/antoine/sae_reid/data/mara/splits/train_indices_gold.txt \
  --test-indices /data/vision/beery/scratch/antoine/sae_reid/data/mara/splits/test_indicies_gold.txt \
  --preprocessed-root /data/vision/beery/scratch/antoine/sae_reid_archive/20260925_170036/removed_from_workspace/data/mara/mara_preprocessed
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg \
  python experiments/seek_ear_only/add_margin.py
python experiments/seek_ear_only/report.py
```

The original split symlinks and absolute crop paths became stale after an archive
move. Both relocated split files are byte-identical to the archived originals;
`results/relocation_audit.json` records the paths and matching hashes. The crop
override replaces only the path prefix, preserving all relative suffixes and the
archived original image files. Neither metadata nor source data is edited.
The final exact command is also written to `results/reproduce.sh`.
A source-hash-validated feature cache avoids repeated backbone inference;
removing `results/feature_cache.pt` forces fresh inference. `--prepare-only` runs
only feature extraction and endpoint auditing.

Outputs in `experiments/seek_ear_only/results/` include configuration and input
SHA-256 hashes, endpoint audit, query/gallery order and gallery correction mask,
per-image hits, per-sighting hit counts/means, `summary.csv/json`,
`paired_differences.csv`, the factual `report.md`, and `section_4_2.tex`.
`margin8_decisions.csv` saves all eight ordered choices for each sighting, and
`margin8_audit.json` records the trace/input checks and hashes. All four conditions
share the same 10,000 bootstrap sighting samples. The original three conditions'
means and confidence intervals are checked to remain unchanged when adding the row.
Per-sighting Recall@1 fields are fractions in [0, 1]; summary recalls are
percentages and differences/intervals are percentage points.
Result files follow the repository's existing ignored
results-directory convention; source and this README are versionable.

Validation:

```bash
python -m unittest experiments.seek_ear_only.test_run -v
python experiments/seek_ear_only/validate_results.py
```
