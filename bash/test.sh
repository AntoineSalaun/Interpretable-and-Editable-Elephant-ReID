#!/bin/bash
#SBATCH -o CBM_reid/bash/slurm_logs/test.sh.log-%j
#SBATCH --time=1-12:00:00
#SBATCH --partition=vision-beery
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main

# Train baseline
python CBM_reid/methods/exp_baselines.py --experiment exp_1_1 --epochs 5
