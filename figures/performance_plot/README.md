# Performance figures

Generate the journal sighting figure from the repository root:

```bash
python figures/performance_plot/plot.py --journal
```

This writes `performance_sighting_journal.pdf` and a PNG preview, leaving the
existing PDFs intact. The journal layout places the legend inside the upper-left
of the plot, uses 20% x-tick intervals, adds light horizontal grid lines, and
extends the plotting area to the top margin. Data, colors, and axis ranges match
the original panel.

Include the journal PDF at `width=0.75\linewidth` (see `journal.tex`). Its
9.1837 × 5.6 inch canvas uses the same scaling as the 6-inch dataset histograms
at `0.49\linewidth`. The revised 20 pt x-label and 18 pt y-label are about 10%
smaller than the histogram labels; the 16 pt ticks match their printed size:

```text
0.75 / (6 × 0.75 / 0.49) = 0.49 / 6
```

Both use DejaVu Serif. The PDF uses the full canvas rather than a tight crop
to preserve this scaling. If the histograms are placed at `0.5\linewidth`,
their text is about 2% larger, which is visually very similar.

Run `python figures/performance_plot/plot.py` without `--journal` to regenerate
the three original performance PDFs with their existing layout.
