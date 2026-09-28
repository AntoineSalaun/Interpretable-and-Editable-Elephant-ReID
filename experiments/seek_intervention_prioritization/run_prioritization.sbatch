#!/bin/bash
#SBATCH --job-name=seek-priority
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --time=2-0:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/seek_prioritization.%j.out
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

echo "Started at $(date)"
echo "Host: $(hostname)"
python --version
nvidia-smi || true

python -u experiments/seek_intervention_prioritization/run_prioritization.py \
  --download-checkpoint \
  --wandb-mode online \
  --output-dir experiments/seek_intervention_prioritization/results

echo "Finished at $(date)"
