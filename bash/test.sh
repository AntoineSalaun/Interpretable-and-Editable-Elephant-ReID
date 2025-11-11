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

echo "== TRAINING CONCEPT HEAD ON FULL MARA WITH NO PRETRAINING(200 EPOCHS) =="
date

#python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py --experiment baseline_2_2 --epochs 50 --dataset mara --code MARA2+_50ep_MDfinetuning
python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py  --epochs 200 --experiment train_concept_head_on_full_mara