# Journal figure exports

All journal variants share DejaVu Serif, 20 pt x-axis labels, 18 pt y-axis labels,
16 pt ticks, and 15 pt legends. Canvas widths compensate for the different LaTeX
inclusion widths, so these sizes match across the manuscript. The axis labels
are about 10% smaller than those of the dataset histograms at 0.49 linewidth.
The original figure PDFs are retained; outputs end in `_journal.pdf` with PNG
previews alongside them.

| Figure | PDF relative to `figures/` | LaTeX width |
| --- | --- | --- |
| Performance (9) | `performance_plot/performance_sighting_journal.pdf` | `0.75\linewidth` |
| Annotated heatmap (11) | `correction_policies/heatmap_sighting_with_numbers_journal.pdf` | `0.70\linewidth` |
| Learned fusion alpha | `alpha/alpha_vs_cor_journal.pdf` | `0.75\linewidth` |
| Additional identity information | `concept-contribution/leave-one-out/leave_one_out_journal.pdf` | `\linewidth` |
| Intervention prioritization | `priorization/intervention_prioritization_journal.pdf` | `0.75\linewidth` |

Run from the repository root:

```bash
python figures/performance_plot/plot.py --journal
python figures/correction_policies/heatmap.py --journal
python figures/alpha/alpha_vs_cor.py --journal
python figures/concept-contribution/plot.py --journal
python figures/priorization/plot_journal.py
```

Line-plot legends and the bar-chart legend sit inside the axes. Correction axes
have labels every 20%. The heatmap retains all 10% cells and all numeric cell
annotations, including the special sighting-correction row `S`. The bar chart
retains negative contributions and places category labels below the plotting
area. Prioritization is rendered directly from the exported
`priorization/confidence_intervals.csv`, preserving all estimates and bounds.

The existing heatmap PDF differs from the current CSV in its 0%-training
column; that CSV was exported while the replacement run was still incomplete.
To keep this a formatting-only update, the journal heatmap reproduces all 132
displayed values and original RGB cell fills extracted directly from the
existing vector PDF. `correction_policies/heatmap_sighting_published.json`
records that visual snapshot and the source PDF's SHA-256. Its numbers are
the rounded annotations, not reconstructed full-precision experimental metrics.
The other journal figures use their existing numerical source tables.

Shared sizing and typography are defined in `journal_style.py`. PDFs retain the
full canvas because a tight crop would change their effective LaTeX font scale.
Use `journal_figures.tex` as inclusion snippets for the existing figure environments.
