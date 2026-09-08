#!/bin/bash
#SBATCH --job-name=cbm-reid-backbone-normalized
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
##SBATCH --requeue

#SBATCH --time=7-0:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/backbone_normalized_gold.%j.out
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

REPO_ROOT=/data/vision/beery/scratch/antoine/CBM_reid
RUN_NAME=TCB_mara_MDgold_lr1e-6_wd1e-5_e250_normalized_gold

mkdir -p "${REPO_ROOT}/bash/slurm_logs" "${REPO_ROOT}/experiments/wandb" "${REPO_ROOT}/experiments/wandb-cache"

export MPLCONFIGDIR=/tmp
export WANDB_DIR="${REPO_ROOT}/experiments/wandb"
export WANDB_CACHE_DIR="${REPO_ROOT}/experiments/wandb-cache"

python "${REPO_ROOT}/methods/run_experiment.py" \
    --pipeline_name FinetuningTripleCropBackbone \
    --epochs 250 \
    --dataset mara \
    --subset 2+encounters \
    --backbone_pretraining MegaDescriptor_complete_gold,MegaDescriptor_left_gold,MegaDescriptor_right_gold \
    --backbone_lr 1e-6 \
    --backbone_wd 1e-5 \
    --backbone_layer_norm True \
    --print_every 5 \
    --batch_size 64 \
    --save_best True \
    --wandb_group TripleCropBackbone_Mara_MDgold_normalized \
    --wandb_name "${RUN_NAME}"

cp "${REPO_ROOT}/experiments/exp_${RUN_NAME}/backbone_w.pt" "${REPO_ROOT}/weights/backbone_normalized_gold.pt"
