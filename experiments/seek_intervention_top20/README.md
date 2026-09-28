# SEEK interventions: fixed initial top-20 entropy and 95% intervals

## Final paper figure

The final comparison now displays only Random, Margin-ratio, and Oracle. The
entropy policies below document the completed exploratory experiment; their raw
results are retained, but are omitted by the plotter's default display. This does
not remove the concept-entropy term from the margin score.

A subsequent full-data rerun in `../seek_margin_rescaled/` replaces the retrieval
log-probability ratio by the raw identity-similarity gap and rescales its weight
to beta=8.455150030741938. All selections, retrieval hits, and confidence intervals
matched this experiment exactly; the final paper figure now uses that verified
rerun. This directory retains the original experiment for provenance.

The compact final figure, revised LaTeX subsection, supporting ear-concept evidence,
and code/error-bar explanation are in
[`figures/priorization/`](../../figures/priorization/README.md). Reproduce them with:

```bash
python figures/priorization/export.py
```

`plot_results.py` defaults to three curves, larger labels, and no title or footer;
use `--include-entropy` to display the two exploratory entropy curves again.
It retains the full confidence-interval table in the source results, while a
separate export directory receives only the displayed policies' intervals.

## Original experiment

This experiment is isolated from `../seek_intervention_prioritization`. Its runner
is a snapshot of that runner with shortlist scoring and evaluation traces added.
It reuses the original unchanged policy classes, model checkpoint, and uncertainty
helpers. It never writes results into the previous experiment's directory.

## Question

Does restricting expected entropy reduction to initially plausible identities make
early corrections more useful for ReID? Full entropy assigns each probability a
contribution `-p log p`; it does not assign equal importance to all ranks. However,
many small tail probabilities can contribute substantial total entropy. The
top-20 variant tests whether excluding that tail improves correction selection.
This is a hypothesis, not an assumed explanation of the previous outcome.

## Shared retrieval and corrections

- Same `Heatmap_three_20cor` checkpoint and fixed `G_S(20)` gallery as before.
- Sixteen categorical SEEK attributes; one human correction costs one unit.
- Attribute probabilities are averages of per-image concept-head softmax outputs
  across a sighting. No expert attribute values enter deployable policy selection.
- A selected attribute is replaced with its expert sighting value in every image
  of the sighting. Earlier corrections remain fixed and cannot be selected again.
- Recompute intervention scores after every correction. Evaluate budgets 0..16.
- Query embeddings combine the visual branch and the projected current SEEK
  representation with the checkpoint's learned mixing coefficient.

## Fixed initial top-20 entropy

Let `a_y(x)` be the maximum similarity to any gallery image of identity `y`.
For each query image, before ANY corrections, select:

```text
A_x = the top min(20, number of gallery identities) identities by a_y(x, budget=0).
```

These are identity candidates, not 20 individual gallery photographs. `A_x` is
computed without query identity labels and frozen for all 16 correction decisions.
Different images in a sighting may have different shortlists; their entropies are
averaged, preserving the original sighting aggregation. Scores are still computed
against the full gallery after every real or hypothetical correction.

For a current or hypothetical corrected state `z`, renormalize within the SAME
initial shortlist:

```text
p_A(y | x,z) = exp(a_y(x,z)/tau) / sum_{j in A_x} exp(a_j(x,z)/tau), y in A_x
H_A(s,z) = mean_{x in s} [-sum_{y in A_x} p_A(y | x,z) log p_A(y | x,z)]
S_i(s,z) = H_A(s,z) - sum_v p(c_i=v | s) H_A(s,z with i corrected to v)
i* = argmax over remaining attributes of S_i(s,z)
```

There is no additional concept-entropy term. Only after selecting `i*` is its
expert value revealed and applied. An outside identity becoming highly ranked
does not change shortlist membership. Initial shortlist identities are exported
in `trace_entropy_42.npz`; its true-identity coverage is measured only as a
post-hoc diagnostic, not used for selection.

This variant can miss useful information when the true identity is initially
outside the top 20. Therefore final Recall@1 always searches the FULL gallery,
never just the shortlist, and the all-identities entropy policy is rerun as a
control. Initial shortlist coverage does not cap final full-gallery Recall@1.

## Policies and calibration

The full run evaluates Random (20 independent policy seeds, 42..61), fixed-initial
top-20 expected entropy, original all-identities expected entropy, Margin-ratio,
and Oracle. Margin remains:

```text
S_i = H(C_i | s) + lambda * [sum_v p(c_i=v | s) M(s with i=v) - M(s)]
M(s) = mean over images of log(top identity probability / second probability)
```

Retrieval temperature is fitted using train-gallery leave-one-out identity NLL.
Margin lambda is retuned on a sighting holdout constructed from training indices
only, using the unchanged logarithmic grid and mean validation Recall@1 at budgets
1..4. Both parameters are frozen before test evaluation. No test-based tuning is
introduced. Random, margin, and oracle decisions have their original semantics.

## Error bars: exactly what is estimated?

The y-axis is image-weighted ReID Recall@1 (%), using fused visual and SEEK
embeddings. An image counts as correct when the highest-scoring gallery image
has the correct identity. Temperature changes uncertainty scores, not the ranking
used to measure Recall@1 at a fixed corrected state.

Every policy saves each test image's hit/miss at every budget, including budget 0.
This makes the plotted mean and uncertainty use the same image weighting.

The main plot shows pointwise 95% percentile bootstrap intervals (4,000 resamples,
seed 2026):

1. Sample whole test identities with replacement. Keep all sightings and images
   of a sampled identity together; calculate image-weighted Recall@1 in the sample.
2. For Random, independently resample its 20 policy runs with replacement and
   average their outcomes on that identity sample. This includes both test-sample
   and finite-policy-seed uncertainty. Deterministic policies have one trajectory.
3. Take the 2.5th and 97.5th percentiles at each correction budget.

Identity resamples are shared across policies and budgets. This respects repeated
images/sightings of the same subject. Repeating seeds for deterministic margin or
entropy would not provide uncertainty, so their intervals use identity resampling.
`confidence_intervals.csv` also includes seed-only 95% intervals for the mean
Random curve, conditional on the fixed test sample. These differ from a standard
deviation band or an interval covering 95% of individual random runs.

Intervals are conditional on this trained checkpoint and fixed gallery; they do
not capture training randomness or gallery sampling. They are pointwise, not a
simultaneous confidence band across budgets. Overlap alone is not a formal paired
test of policy differences. All policies share budget-0 and budget-16 outcomes,
but these endpoints can still have nonzero test-identity uncertainty.

## Run and outputs

```bash
python -m unittest discover -s experiments/seek_intervention_top20/tests -v
sbatch experiments/seek_intervention_top20/run_prioritization.sbatch
python experiments/seek_intervention_top20/plot_results.py
```

Full output goes into `results/`; the smoke test goes into `results_smoke/`.
W&B uses project `CBM-ReID`, a separate `seek_intervention_top20` group, and logs
the final confidence-interval plot/table and initial shortlist coverage.

- `prioritization_results.csv/json`: curves, configuration, calibration and checks.
- `confidence_intervals.csv`: image-weighted means and both interval definitions.
- `intervention_prioritization.png/pdf`: three final curves by default; five with
  `--include-entropy`. The original full-run exports contained five curves.
- `query_metadata.npz`: query ordering, identity labels and sighting IDs.
- `trace_<policy>_<seed>.npz`: image-by-budget hit matrices; entropy shortlist audit.
- `per_sighting_interventions.json`: chosen concepts, scores and diagnostics.
- `run_metadata.json`: W&B URL, Slurm job ID and effective command arguments.

Per-run traces and the partial results CSV are saved as each policy seed finishes.
The full JSON and CI plot are written after all policies complete. Logs are in
`bash/slurm_logs/seek_top20.<jobid>.out` relative to the repository root.

The smoke test uses three sightings, a 512-image gallery, all budgets and three
random seeds, with previously calibrated temperature/lambda supplied to keep the
test short. The full Slurm command recalibrates normally. CPU tests cover frozen
shortlists, outside-leader exclusion, temperature, full-support equivalence,
expected-entropy selection, identity clustering and seed variability. Runtime
checks verify trace means against the original metric, common endpoints, and
complete cumulative correction at budget 16.

## Previous run: ear concepts among the first five

Across the previous margin run's 1,269 sightings, 6,078 of the 6,345 corrections in
positions 1..5 are ear-hole/tear attributes: **95.79196%**. The denominator counts
five decisions per sighting, not the five most frequent attribute types.
