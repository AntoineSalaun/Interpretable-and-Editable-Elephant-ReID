# Dataset distributions

Regenerate the separate PDF and 300 dpi PNG panels:

```bash
python figures/dataset_distribution/plot.py
```

`figure.tex` supplies the two-minipage layout. Use the same inclusion width as
the performance panels to preserve matching apparent font sizes.

The plots reproduce cells 6 and 9 of
`data/mara/mara_dataset_processing.ipynb`, retaining its 160 and 40 bins and
its exact blue (`#1F77B4`). Both frequency axes are logarithmic. Fonts match
the performance PDFs: DejaVu Serif, 22 pt x-labels, 20 pt y-labels, 16 pt ticks,
on a 6 × 4 inch canvas. Both panels have equal page dimensions and margins.

Panel A includes every individual sighting in the original table, including
sightings without an identified elephant. Panel B counts distinct sightings
for each identified elephant, as in the original notebook.

The original input was recovered from Git history into the ignored local file
`data/mara/25-10-30_individual_sightings.tsv`. To recover it in a fresh checkout:

```bash
git show '9cb90207b^:data_processing/mara/25-10-30_individual_sightings.tsv' > data/mara/25-10-30_individual_sightings.tsv
```

An alternate location can be supplied with `--source /path/to/table.tsv`.
