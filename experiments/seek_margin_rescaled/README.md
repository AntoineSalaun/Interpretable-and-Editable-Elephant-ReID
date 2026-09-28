# Raw-margin rescaling check

This reruns margin intervention selection using raw identity-similarity gaps:

```text
D(x) = highest identity similarity - second-highest identity similarity
D(s) = mean of D(x) across images in the sighting
I_i = sum_v p_i(v | s) D(s with i=v) - D(s)
S_i = H(C_i | s) + beta * I_i
```

Concept entropy remains unnormalized. Identity similarities still use the maximum
over gallery images of each identity. Each hypothetical correction recomputes its
own top two identities. Expert values remain hidden until the attribute is chosen,
and previous corrections stay fixed.

For an unclipped softmax, log(pi_1/pi_2) = (a_1-a_2)/tau. Thus the original
validation-selected lambda=0.3 and frozen tau=0.03548133373260498 correspond to
beta=lambda/tau=8.455150030741938. This is a change of units, not a newly tuned
hyperparameter. Keeping 0.3 on raw gaps would change the selection rule.

## Completed full-data check

The rerun finished in 340.5 seconds. All 20,304 intervention choices match the
original recorded choices, and all image-level hit/miss outcomes match at every
budget for all 16,506 queries. The maximum difference in the plotted 95% confidence
bounds is zero. The audit found zero clipped top-two probabilities across
8,025,032 scored image rows, with a minimum second probability of 7.91e-10
(above the 1e-12 floor). Maximum policy-score discrepancy was 5.97e-7, with no
selection differences. See `results/rescaling_audit.json` and W&B run `m1ty69xg`.

## Run

```bash
python -m unittest discover -s experiments/seek_margin_rescaled/tests -v
python -u experiments/seek_margin_rescaled/run.py
```

To regenerate only the plot from these completed results:

```bash
python experiments/seek_intervention_top20/plot_results.py \
  --results-dir experiments/seek_margin_rescaled/results \
  --output-dir figures/priorization
```

The script uses the same frozen checkpoint, full train gallery and test queries
as `../seek_intervention_top20/results`. It does not recalibrate or tune on test
data. Results are written to this directory's `results/`. W&B uses the separate
group `seek_margin_rescaled`. Optional `--max-sightings N --output-dir ...` performs
a subset check against the full gallery without replacing the final paper plot.

`policy.py` implements raw-gap selection. It batches all remaining attribute-value
counterfactuals for a sighting; the model is in evaluation mode, and normalization
is per image, matching the original runner. The scorer computes raw gaps directly
from identity similarities. It separately evaluates the old probability-based
margin on the same counterfactuals solely for auditing; those probabilities and
the old temperature do not affect raw-policy selection.

## Verification and outputs

- Five CPU tests verify softmax cancellation, score equivalence after rescaling,
  the non-equivalence when a probability floor is active, a single identity, and
  batched counterfactual scores against individual calls to the original evaluator.
- The full run counts any top-two probability clipping and records the largest
  differences between raw-gap and log-probability scores.
- Every raw decision is compared with the old formula on the same current state,
  and with the original run's saved intervention sequence. Small numerical ties
  can change selections even when the algebra is equivalent.
- Each image's hit/miss is compared at all 17 budgets with the original run.
  `rescaling_audit.json` reports differences instead of assuming they are zero.
- Random's 20 traces and the Oracle trace are reused on the identical query set;
  only the raw-margin policy needs new inference.
- The same 4,000 identity-cluster bootstrap replicates regenerate the confidence
  intervals. `max_ci_difference_pp` compares them with the original figure.

The final full-run PDF/PNG and three-policy interval CSV are exported to
`figures/priorization/`, with a separate `rescaling_audit.json`. No original model
checkpoint or original experiment result is replaced. `margin_decisions.csv`
contains the raw importance, concept uncertainty, beta, combined score and both
reference choices for each decision. `raw_margin_results.csv` saves intermediate
budget progress; `trace_margin_42.npz` stores the newly computed image outcomes.

Metrics named `retrieval_entropy` or `uncertainty_temperature` in diagnostic rows
come from the reused evaluator and are not inputs to the raw selection rule.
The existing LaTeX describes the original log-ratio parameterization; raw-gap
notation should use beta=8.45515 rather than retaining lambda=0.3.
