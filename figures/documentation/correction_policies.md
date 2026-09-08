# Correction Policies

This note defines the active SEEK correction policies used by `methods/run_experiment.py`.

## Mara SEEK Codes

For Mara, the raw archive metadata does not contain a true elephant-level SEEK code. The SEEK-bearing source used by the Mara notebook gives a `seek_code` per `individual_sighting_id`.

- `subject-SEEK` is the image-level code. It includes visibility and occlusion, so missing ears stay masked.
- `ele-SEEK` is the unmasked sighting-level code copied from the raw `seek_code` before image-level masking. It is not a canonical elephant-level oracle.
- The fallback for an uncorrected sample is the hard concept-head prediction, produced with `SEEK.closest_valid_one_hot`.

## Active Treatments

There are only two active correction treatments for Mara:

- `image_p`: sample images with probability/fraction `p`; selected images use `subject-SEEK`.
- `sighting_p`: sample `encounter_id` values with probability/fraction `p`; every image in a selected sighting uses `ele-SEEK`.

`hard` or `None` means no correction: use the hard concept-head prediction.

`oracle` or `oracle_correction` is kept as a convenience alias for 100% sighting correction: every image uses `ele-SEEK`.

`elephant_p` is intentionally unsupported for Mara. We do not have a raw elephant-level SEEK target, so applying a correction consistently across all images of an elephant would require constructing an aggregate code first. That aggregation choice is archived separately and is not part of the active correction policy.

## Defaults

Old bare numeric policies still parse. Their meaning depends on the call site:

- Projector training: bare `p` means `sighting_p`.
- Retrieval gallery: bare `p` means `sighting_p`.
- Retrieval query: bare `p` means `image_p`.

The old `correct_or_hard` name is kept as an alias for `image_0.5` so older commented commands still parse, but new commands should prefer explicit `image_p` or `sighting_p`.

## Gallery And Query Convention

Gallery and projector-training corrections use `sighting_p` by default, because that is the only grouped ground-truth SEEK treatment available for Mara.

Queries may use either treatment:

- use `image_p` when each image is corrected independently with its visible/occlusion-aware `subject-SEEK`;
- use `sighting_p` when whole sightings should be corrected together with the unmasked sighting-level `ele-SEEK`.

For example, `sighting_0.5` selects 50% of `encounter_id` values; all images in those selected sightings receive their `ele-SEEK`.

## Heatmap Convention

The projector heatmap commands in `bash/projector.sh` should train CHAIR with `correction_at_training=sighting_p` for `p = 0.0, 0.1, ..., 1.0`, plus `oracle_correction`.

After training, `Projector.test` exports the usual `Pr/Test Recall@k with X% Correction correction` W&B summary metrics. For those percentage metrics, gallery correction uses `sighting_p` and query correction uses `image_p`. `ORACLE` remains a separate sanity-check condition using `oracle_correction` on both gallery and query.

The metric names are unchanged so existing plot exporters can still read the runs, but the policy details are also written to W&B summary fields such as `training_correction_policy`, `gallery_correction_policy`, and `query_correction_policy`.
