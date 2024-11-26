#!/bin/bash
#SBATCH -o CBM_reid/bash/slurm_logs/test.sh.log-%j
#SBATCH --time=22:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main

# Train baseline
python /data/vision/beery/scratch/antoine/CBM_reid/methods/testing_stuff.py