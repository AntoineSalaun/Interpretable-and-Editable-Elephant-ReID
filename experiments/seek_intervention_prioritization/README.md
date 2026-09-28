# Prioritized SEEK Interventions

This isolated experiment asks which SEEK attribute an expert should correct
first for a query sighting when retrieval uses the existing CHAIR/SEEK-CBM
model.  It evaluates four policies:

- Random
- Expected retrieval entropy reduction
- Margin-ratio, CooP-style
- Greedy oracle

The output is a curve of Recall@1 versus the number of corrected SEEK
attributes per query sighting.

## Fixed Reference Model

The experiment uses the third sighting-level correction column from the paper
heatmap, corresponding to `G_S(20)`:

- W&B run: `Heatmap_three_20cor`
- W&B id: `u63g7ldu`
- W&B group: `Projector_heatmap_three_alpha_learned_min10`
- Dataset/subset: `mara`, `2+encounters`
- Training correction: `sighting_0.2`
- Local checkpoint: `../experiments/exp_Heatmap_three_20cor`
- Checkpoint files: `projector.pt`, `backbone_w.pt`, `alpha.pt`

The gallery is fixed at `sighting_0.2` for the whole experiment.  Policies only
change the query sighting SEEK vector by revealing additional expert-corrected
attributes.  Existing paper experiments and training code are not modified.

For each query image, the retrieval embedding is:

```text
z(x, c) = (1 - alpha) * f_image(x) + alpha * f_projector(c)
```

where `c` is the current query SEEK vector.  At budget 0, `c` is the hard
concept-head prediction.  At budget 16, every SEEK attribute has been replaced
by the true sighting-level SEEK value.

## Y-Axis

The plotted y-axis is gallery-entry Recall@1 in percent.  For each query image,
we rank gallery images by CHAIR similarity and check whether the top-ranked
gallery image has the same elephant identity as the query:

```text
R@1 = mean_x 1[label(top_gallery(x)) = label(x)]
```

This is a ReID metric, but it is not pure visual ReID: it is measured on the
fused visual-plus-SEEK CHAIR embedding above.  `R@5` and MRR are saved in the
CSV/JSON but not plotted by default.

## Sighting-Level Concept Distribution

Each SEEK attribute is categorical.  For an attribute `i`, the concept head
produces image-level logits over its allowed values.  The policy uses the
sighting-level distribution:

```text
p(c_i = v | s) = mean_{x in s} softmax(logits_i(x))[v]
```

This aggregation is label-free and does not use expert values.

## Retrieval Uncertainty And Temperature

Policy uncertainty is computed over elephant identities, not over individual
gallery images.  For a query image `x`, gallery-image similarities are max
pooled by identity:

```text
a_y(x) = max_{g: label(g)=y} sim(z_x, z_g)
```

The normalized identity probability is temperature scaled:

```text
pi_y(x; tau) = softmax_y(a_y(x) / tau)
```

`tau` controls how sharp the identity distribution is.  A small temperature
makes the policy more confident; a large temperature makes retrieval uncertainty
flatter.  If `--temperature` is not supplied, the runner fits one scalar
temperature on non-test data only: the training gallery is used in a
leave-one-out identity NLL objective, and calibration rows with no finite
same-identity leave-one-out target are skipped.  The chosen temperature is then
frozen for all validation and test policy decisions.

The helper `IdentityScorePooler` caches the gallery-label to identity mapping so
counterfactual uncertainty scoring does not repeatedly rebuild identity pools.

## Policy 1: Random

Random chooses one remaining uncorrected SEEK attribute uniformly.  It is run
with multiple seeds.  The plot shows its mean and a one-standard-deviation band.

## Policy 2: Expected Retrieval Entropy Reduction

Let:

```text
H(Y | s) = mean_{x in s} -sum_y pi_y(x) log pi_y(x)
```

For a candidate SEEK attribute `i`, the policy evaluates every possible value
`v` by temporarily setting the whole sighting to `c_i=v`, rerunning retrieval
uncertainty, and averaging by the model probability `p(c_i=v | s)`:

```text
S_i_entropy =
    H(Y | s)
    - sum_v p(c_i=v | s) H(Y | s, c_i=v)
```

The selected intervention is:

```text
i* = argmax_i S_i_entropy
```

This policy does not add concept entropy as a separate bonus.  Concept
uncertainty enters only through the expectation over `p(c_i | s)`.  Unit tests
check that this policy selects the attribute with the largest expected
downstream entropy reduction and that it does not score concept uncertainty when
retrieval entropy is unchanged.

## Policy 3: Margin-Ratio

The margin policy follows the CooP-style separation between concept uncertainty
and downstream importance.

For concept uncertainty:

```text
U_i(s) = H(C_i | s) = -sum_v p(c_i=v | s) log p(c_i=v | s)
```

For retrieval margin, let `pi_(1)(x)` and `pi_(2)(x)` be the largest and
second-largest identity probabilities for query image `x`:

```text
M(x) = log(pi_(1)(x) + eps) - log(pi_(2)(x) + eps)
M(s) = mean_{x in s} M(x)
```

The expected downstream margin importance of correcting attribute `i` is:

```text
I_i_margin = sum_v p(c_i=v | s) M(s with c_i=v) - M(s)
```

The final score is:

```text
S_i_margin = U_i(s) + margin_lambda * I_i_margin
```

`margin_lambda` is tuned on a validation split carved only from the training
indices.  The default grid is:

```text
0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100
```

The validation objective is early-intervention performance:

```text
mean_{b=1..K} R@1(b), with K=4
```

The selected `margin_lambda` is frozen before test evaluation.  Per-decision
records include `margin_lambda`, `margin_concept_uncertainty`,
`margin_importance`, and `margin_combined_score`.

## Policy 4: Greedy Oracle

The oracle is intentionally separated from deployable policies.  It receives
the true sighting-level SEEK vector and a truth-aware retrieval scorer.  At each
step it tries each remaining true correction and chooses the attribute with the
best immediate true identity retrieval score.  It is an upper bound, not a
deployable strategy.

## Intervention Loop

For each policy and each query sighting:

1. Start from hard concept-head SEEK predictions.
2. Score every uncorrected attribute using that policy.
3. Select one attribute without seeing its expert value.
4. Reveal only that selected attribute's true sighting-level value.
5. Copy it to every query image in the sighting.
6. Rerun retrieval and recompute all scores before the next budget.

Human cost is fixed to `q_i=1` for all attributes.  The runner checks that all
policies have identical Recall@1 at budget 0 and budget 16.

## Outputs

The full run writes:

- `results/prioritization_results.csv`
- `results/prioritization_results.json`
- `results/per_sighting_interventions.json`
- `results/intervention_prioritization.pdf`
- `results/intervention_prioritization.png`

The JSON stores the exact config, fitted temperature, selected margin lambda,
endpoint checks, checkpoint location, and all per-budget rows.

## Commands

Small calibrated smoke:

```bash
python experiments/seek_intervention_prioritization/run_prioritization.py \
  --smoke \
  --max-sightings 2 \
  --max-gallery-images 128 \
  --max-validation-sightings 1 \
  --max-budget 4 \
  --random-rollouts 1 \
  --wandb-mode disabled \
  --output-dir experiments/seek_intervention_prioritization/results_temperature_entropy_smoke_v2 \
  --no-plot
```

Full local run:

```bash
python experiments/seek_intervention_prioritization/run_prioritization.py \
  --download-checkpoint \
  --wandb-mode online
```

Cluster run:

```bash
sbatch experiments/seek_intervention_prioritization/run_prioritization.sbatch
```

Replot an existing run:

```bash
python experiments/seek_intervention_prioritization/plot_results.py \
  --results experiments/seek_intervention_prioritization/results/prioritization_results.csv \
  --pdf experiments/seek_intervention_prioritization/results/intervention_prioritization.pdf \
  --png experiments/seek_intervention_prioritization/results/intervention_prioritization.png
```

## Validation

Current local checks:

- `python -m py_compile experiments/seek_intervention_prioritization/*.py experiments/seek_intervention_prioritization/tests/*.py`
- Direct execution of all test functions in `tests/`
- Calibrated smoke run without `--temperature`

`pytest` is not installed in the current environment, so the direct test-function
runner is used here.  The tests cover identity max pooling, temperature effects,
finite temperature calibration, gallery-entry Recall@K semantics, entropy
policy behavior, margin score decomposition, and oracle separation.
