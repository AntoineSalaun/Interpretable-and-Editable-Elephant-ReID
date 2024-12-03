#!/bin/bash
#SBATCH -o CBM_reid/bash/slurm_logs/test.sh.log-%j
#SBATCH --time=1-12:00:00
#SBATCH --partition=vision-beery
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --qos=vision-beery-main

# Train baseline
python CBM_reid/methods/background_experiments.py