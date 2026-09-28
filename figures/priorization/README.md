# Final intervention-prioritization figure

The final paper comparison contains Random, Margin-ratio, and greedy Oracle.
Retrieval-entropy policies were exploratory alternatives and are omitted from the
figure. Their saved results remain available to support the brief comparison in
the text. Concept entropy is still the uncertainty term of the margin policy.

## Verified raw-margin rescaling

The final plot now uses the independent full-data raw-gap rerun in
`experiments/seek_margin_rescaled/results`. Its score is
`H(C_i | s) + beta * expected raw similarity-gap increase`, where
`beta = 0.3 / 0.03548133373260498 = 8.455150030741938`. No retrieval probability or
temperature is needed to select an attribute in this implementation. Concept
entropy is unchanged and unnormalized. Original softmax metrics are calculated
separately only to audit the equivalence and retain diagnostic fields.

The rerun matched all 20,304 selections across 1,269 sightings, all 16,506 images'
hit/miss outcomes at all 17 budgets, and every plotted confidence bound exactly.
None of the 8,025,032 evaluated current/counterfactual image rows triggered the
top-two probability floor. The largest score difference was 5.97e-7, from floating
point arithmetic. See `rescaling_audit.json` for full-precision results and the
W&B link, and `experiments/seek_margin_rescaled/README.md` for code and commands.

`export.py` uses this rerun when its audit is present and confirms matching
decisions and outcomes before reusing the original controlled ear-concept
evidence. The sections below retain the original parameterization for provenance.

## Files and reproduction

Run from the repository root:

```bash
python figures/priorization/export.py
```

This regenerates `intervention_prioritization.pdf` and `.png`, the three displayed
policies' `confidence_intervals.csv`, and the supporting `first_concept_effects.csv`
and `evidence.json`. It reads the verified rescaled run for the plot and completed
Slurm run 1817395 in `experiments/seek_intervention_top20/results` for the original
controlled concept evidence; no model inference is rerun.
The large intervention JSON is loaded once to verify the ear-concept claims.

`prioritization.tex` is the revised paper subsection, and `figure.tex` includes the
PDF and caption. The opening paragraph's prose is retained; underscores, citation
key and percent signs use valid LaTeX syntax. Include these files in the manuscript
with `amsmath`, `graphicx`, and its existing bibliography setup.

The plot is 5.6 x 4 inches, with 16-point axis labels and 13-point tick and legend
text. It has no title or footer. Its labels are exactly `Recall@1 (%)` and
`# corrected attributes per query`. PDF is
vector output; PNG is 300 dpi. The source plotter is
`experiments/seek_intervention_top20/plot_results.py`; its default display is now
the three final policies. `--include-entropy` reconstructs the exploratory curves.

## What the code computes

`MarginRatioPolicy.select` in
`experiments/seek_intervention_prioritization/policies.py` averages each attribute's
concept-head probabilities over a sighting. For every uncorrected attribute, it
tries all possible values on the current corrected SEEK state, calls the retrieval
scorer, and computes:

```text
U_i = -sum_v p_i(v) log p_i(v)
M(s) = mean_x [log(max(pi_top1(x), epsilon)) - log(max(pi_top2(x), epsilon))]
I_i = sum_v p_i(v) M(s with attribute i set to v) - M(s)
S_i = U_i + lambda * I_i
```

The numerical implementation floors probabilities at epsilon before taking logs.
Uncertainty and importance are separate additive terms, not a product. The chosen
attribute maximizes S_i. `run_single_policy` reveals its expert value only after
selection, writes that attribute into all images of the sighting, and excludes it
from future selection. All previous corrections persist. No identity label or
expert attribute value is given to the deployable selection scorer.

Gallery-image similarities are max-pooled by identity, then transformed with
softmax(scores / tau). This expresses relative support over candidate identities.
The probability rank is the similarity rank. The fitted tau is 0.0354813337;
lambda is 0.3, selected on validation Recall@1 over budgets 1--4. Both were frozen
before testing. The fixed gallery and learned model are identical for all policies.
The target remains image-weighted retrieval Recall@1, not concept accuracy.

## How the error bars are computed

There are 16,506 query images from 1,269 sightings and 531 identities. For each
budget, `trace_<policy>_<seed>.npz` stores one hit/miss per query image.
`cluster_totals` groups these hits by identity using `query_metadata.npz`.
`bootstrap_curves` performs 4,000 bootstrap replicates:

1. Draw 531 identities with replacement. If an identity is drawn twice, all its
   images and sightings count twice. Keeping the whole identity together respects
   the dependence among repeated observations of the same subject.
2. For Random, also draw 20 runs with replacement from seeds 42..61, independently
   of the identity draw. Average their hit counts. Margin and Oracle each have one
   deterministic trajectory; changing a selection seed would not add variation.
3. Compute Recall@1 as total hits divided by total images in that resample, times
   100. Larger identities therefore retain their image-based contribution.
4. Use the 2.5th and 97.5th percentiles as the lower and upper error-bar endpoints.

The central line is the original-sample Recall@1, averaged over 20 seeds for
Random. It is not the bootstrap mean. Identity resamples are shared across
policies and budgets (seed 2026); the independent seed-resampling RNG uses 2027.
The lightly shaded bands and capped bars show the same interval.

These are pointwise confidence intervals for mean performance, conditional on
the trained model and fixed gallery. They are neither standard deviations nor
intervals containing 95% of individual random runs. They do not include training
or gallery-selection variation, and are not simultaneous bands across all budgets.
Shared endpoint results can still have nonzero uncertainty across test identities.
Do not use overlap alone as a paired test of policy differences.

The source results CSV also contains seed-only intervals, which hold the test set
fixed and resample Random runs. The final plot uses the combined identity/seed
intervals (`ci95_low`, `ci95_high`), not the seed-only columns.

## Evidence for the paper text

`export.py` checks the current full run's margin trajectories, rather than assuming
the earlier run's concept frequencies remain identical. Of the first five
corrections per sighting, 6,078/6,345 are among the eight `L/R` x `hole/tear` x
`1/2` attributes: 95.79196%. This is a count of decisions, not a percentage of the
five most frequent concept types.

The oracle's first-step diagnostics record the effect of correcting every concept
on the same initial state of every sighting. `export.py` averages those outcomes
equally across sightings and compares each concept with the exact uniform average
of all 16 possible first corrections. It verifies the saved margin first-step
outcomes against the corresponding oracle counterfactuals. All eight ear concepts
have higher average first-correction Recall@1 than any of the other eight concepts.
L_tear_1 gains 1.96 percentage points and R_hole_1 gains 1.75 points relative to a
random first correction. These are not gains relative to no correction.

This concept comparison uses equal sighting weights; the main plot uses equal
image weights. Its ordering is descriptive for this model and data, not a claim
that all pairwise differences are statistically significant. Selection frequency
alone is insufficient: five-category ear concepts can also have higher raw entropy
than two- or three-category concepts. The counterfactual comparison supplies direct
evidence of retrieval benefit in addition to the observed selection preference.

The text calls the margin measure simpler because it depends on the top-two gap
rather than the entire candidate distribution. Better efficiency refers to human
correction budget: mean Recall@1 over budgets 1--4 is 29.73% for margin versus
19.34% for full entropy and 23.58% for top-20 entropy. This is not a runtime claim.
