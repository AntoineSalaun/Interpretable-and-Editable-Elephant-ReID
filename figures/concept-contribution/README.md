# Concept Contribution Figures

These scripts generate the concept-grounding figures for correction levels
25%, 50%, 75%, and 100%.

Run:

```bash
python figures/concept-contribution/export_enriched_embeddings.py
python figures/concept-contribution/compute_metrics.py
python figures/concept-contribution/plot.py
```

The exporter restores the projector checkpoints from the W&B t-SNE runs, then
recomputes tensors with stable dataset indices and edited concept codes. Gallery
training concepts use `sighting_p`; query concepts use `image_p`.

The default concept head is the three-head architecture:
`backbone_for_concepts_three_gold` and `concept_head_three_gold`.

The identity probes use simple linear logistic classifiers on grouped SEEK
concept blocks. The eight groups are sex, age, tusks, tear 1, tear 2, hole 1,
hole 2, and extreme/special features.
