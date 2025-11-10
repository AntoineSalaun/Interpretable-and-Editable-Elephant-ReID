#!/bin/bash
#SBATCH --job-name=cbm-reid-test
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery

#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Use absolute paths for logs
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/test.%j.out

# Set the working directory explicitly (adjust to your repo root)
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

# Ensure log directory exists (harmless if it already does)
mkdir -p /data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs

echo "== mara2+_MD_finetuning 400 EPOCHS BACKBONE FINETUNING =="
date

python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py --epochs 400 --experiment mara2+_MD_finetuning
