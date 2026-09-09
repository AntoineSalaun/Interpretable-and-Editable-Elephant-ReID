# Prioritized SEEK Interventions

This isolated experiment asks which SEEK attribute an expert should correct
first for a query sighting.  It produces a Recall@1-vs-correction-budget curve
for Random, Expected entropy reduction, Margin-ratio, and Oracle policies.

## Reference Heatmap Setup

The relevant paper heatmap path in this checkout is the CHAIR correction policy
pipeline:

- Figure data: `figures/correction_policies/CHAIR_vs_cor.csv`
- W&B exporter: `figures/correction_policies/export_projector_correction_runs.py`
- Heatmap plotter: `figures/correction_policies/heatmap.py`
- Original cluster commands: `bash/projector.sh`

The sighting-scope heatmap columns are ordered by `correction_at_training`:
`sighting_0.0`, `sighting_0.1`, `sighting_0.2`, ... .  The third column is
therefore the run:

- W&B run: `Heatmap_three_20cor`
- W&B id: `u63g7ldu`
- W&B URL:
  `https://wandb.ai/antoinesalaun-massachusetts-institute-of-technology/CBM-ReID/runs/u63g7ldu`
- W&B project/entity:
  `antoinesalaun-massachusetts-institute-of-technology/CBM-ReID`
- W&B group: `Projector_heatmap_three_alpha_learned_min10`
- Training correction: `sighting_0.2`
- Dataset/subset: `mara`, `2+encounters`
- Split files: `weights/train_indices_gold.txt`,
  `weights/test_indicies_gold.txt`
- Concept backbone: `backbone_for_concepts_three_gold`
- Retrieval backbone: `backbone_normalized_gold`
- Concept head: `concept_head_three_gold`
- Concept-head architecture: `three`
- Projector policy: `sequential`
- Projector architecture: `small`
- Projector lr/wd: `1e-4`, `1e-6`
- Layer norm: `True`
- Alpha: learned from initial `0.5`; best alpha
  `0.41754093766212463`
- Best checkpoint epoch: `65`
- Batch size: `64`
- Checkpoint files: `projector.pt`, `backbone_w.pt`, `alpha.pt`

The local checkpoint source is expected at
`../experiments/exp_Heatmap_three_20cor`.  If it is missing, the runner can
download the W&B run files into
`experiments/seek_intervention_prioritization/cache/checkpoints/Heatmap_three_20cor`.

The new experiment keeps the gallery fixed at `sighting_0.2` by default, i.e.
the `G_S(20)` gallery condition, and changes only the query-sighting
intervention procedure.  The current local CHAIR heatmap export mostly contains
diagonal correction cells, so exact reference comparisons are available for
those exported endpoints and documented as such in the result JSON.

No calibrated retrieval temperature was found in the existing heatmap code.  If
`--temperature` is not provided, the runner fits one scalar temperature on the
training gallery with leave-one-out identity NLL and freezes it before test
query evaluation.

## What Runs

For every query sighting, the experiment starts from hard concept-head SEEK
predictions.  At each budget, each policy chooses one uncorrected SEEK
attribute.  The expert sighting-level value for that attribute is then copied to
all query images in that sighting.  The policy is recomputed after every actual
correction.

Deployable policies receive only:

- current query SEEK representations,
- model concept logits/probabilities,
- the corrected-attribute mask,
- a label-free retrieval uncertainty scorer.

They do not receive query identity labels or expert SEEK values.  The
`greedy_oracle` implementation is separate and explicitly receives the expert
SEEK and true identity scorer.

## Learned Policy Scores

Expected entropy reduction remains a downstream retrieval-uncertainty rule:

```text
S_i = H(Y | s) - E_v[H(Y | s, c_i = v)].
```

It does not add a separate concept-entropy term; concept uncertainty enters only
through the expectation over the sighting-level distribution `p(c_i | s)`.

The margin policy is now CooP-style and explicitly combines concept uncertainty
with downstream retrieval importance:

```text
S_i = H(C_i | s) + margin_lambda * I_i_margin
I_i_margin = E_v[M(s with c_i = v)] - M(s)
```

`M(s)` is the average image-level log ratio between the largest and
second-largest normalized identity probabilities for the query sighting.  The
expert value remains hidden until the highest-scoring attribute is selected.

`margin_lambda` is selected on a deterministic validation split carved only from
the training/gallery indices.  The default grid is:

```text
0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100
```

The validation objective is mean Recall@1 over non-zero budgets `1..K`, with
`K=4` by default.  The chosen value is frozen before test-query evaluation and
stored in the result JSON under `margin_lambda`.

## Commands

Smoke test on a few query sightings:

```bash
python experiments/seek_intervention_prioritization/run_prioritization.py \
  --smoke \
  --max-sightings 5 \
  --max-gallery-images 1024 \
  --temperature 1.0 \
  --wandb-mode disabled
```

The smoke path deliberately uses a reduced gallery but forces the selected
query identities into that gallery so oracle diagnostics remain meaningful.  The
full run omits `--max-sightings` and `--max-gallery-images`.  For faster local
debugging, `--max-validation-sightings` can limit only the margin-lambda tuning
split.

Full run:

```bash
python experiments/seek_intervention_prioritization/run_prioritization.py \
  --download-checkpoint \
  --wandb-mode online
```

Cluster run:

```bash
sbatch experiments/seek_intervention_prioritization/run_prioritization.sbatch
```

Replot existing results:

```bash
python experiments/seek_intervention_prioritization/plot_results.py
```

## Outputs

The runner writes:

- `results/prioritization_results.csv`
- `results/prioritization_results.json`
- `results/per_sighting_interventions.json`
- `results/intervention_prioritization.pdf`
- `results/intervention_prioritization.png`

The CSV contains at least `policy`, `seed`, `budget`, `recall_at_1`,
`recall_at_5`, and `mrr`.  Recall values are stored as percentages to match the
existing W&B summary convention; `mrr` is stored as a fraction.  Margin rows also
contain the frozen `margin_lambda`, and per-sighting intervention records include
`margin_concept_uncertainty`, `margin_importance`, and
`margin_combined_score`.

## Validation

Unit tests:

```bash
python -m pytest experiments/seek_intervention_prioritization/tests
```

Runtime checks:

- Budget 0 Recall@1 is identical across policies.
- Budget 16 Recall@1 is identical across policies.
- Entropy and margin policies have no target labels or expert SEEK in their
  policy signatures.
- Tests verify entropy/margin selections do not change when labels are
  shuffled.
- The result JSON records temperature calibration and reference heatmap
  comparison status.
