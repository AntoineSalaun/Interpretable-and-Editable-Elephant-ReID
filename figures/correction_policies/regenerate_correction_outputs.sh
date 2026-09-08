#!/bin/bash
set -euo pipefail

cd /data/vision/beery/scratch/antoine/CBM_reid

python figures/correction_policies/export_projector_correction_runs.py
python figures/correction_policies/heatmap.py
python figures/performance_plot/export_seek_retrieval_runs.py
python figures/performance_plot/export_elephantbook_runs.py
python figures/performance_plot/export_miewid_runs.py
python figures/performance_plot/plot.py
python figures/alpha/alpha_vs_cor.py
