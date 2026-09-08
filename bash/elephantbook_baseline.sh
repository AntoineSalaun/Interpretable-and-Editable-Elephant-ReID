#!/bin/bash
#SBATCH --job-name=cbm-reid-elephantbook
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
##SBATCH --requeue

#SBATCH --time=7-0:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/elephantbook.%j.out
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

REPO_ROOT=/data/vision/beery/scratch/antoine/CBM_reid

mkdir -p "${REPO_ROOT}/bash/slurm_logs" "${REPO_ROOT}/experiments/wandb" "${REPO_ROOT}/experiments/wandb-cache"

export MPLCONFIGDIR=/tmp
export WANDB_DIR="${REPO_ROOT}/experiments/wandb"
export WANDB_CACHE_DIR="${REPO_ROOT}/experiments/wandb-cache"

# ElephantBook-style fusion baseline with body-only MegaDescriptor:
# - no CurveRank computation; use only the complete/body crop MegaDescriptor as the visual confidence.
# - SEEK distance is 0 for equal attributes, 1 for different attributes, and 0.6 when either side is "_".
# - combined score = backbone_weight * backbone_confidence + (1 - backbone_weight) * SEEK_similarity.
# - each Slurm job handles one correction percentage and evaluates both gallery scopes:
#   gallery=sighting_p, query=image_p and gallery=image_p, query=image_p.
for pct in 00 10 20 30 40 50 60 70 80 90 100; do
    probability=$(awk "BEGIN { printf \"%.1f\", $pct / 100 }")
    sbatch --parsable \
        --job-name="EBbody_${pct}" \
        --partition=vision-beery \
        --qos=vision-beery-main \
        --account=vision-beery \
        --time=7-0:00:00 \
        --mem=64G \
        --gres=gpu:1 \
        --chdir="${REPO_ROOT}" \
        --output="${REPO_ROOT}/bash/slurm_logs/elephantbook_body_${pct}.%j.out" \
        --wrap="
            set -eu
            export MPLCONFIGDIR=/tmp
            export WANDB_DIR='${REPO_ROOT}/experiments/wandb'
            export WANDB_CACHE_DIR='${REPO_ROOT}/experiments/wandb-cache'

            python '${REPO_ROOT}/methods/run_experiment.py' \
                --pipeline_name ElephantBook \
                --dataset mara \
                --subset 2+encounters \
                --backbone_pretraining MegaDescriptor_complete_gold \
                --backbone_layer_norm True \
                --elephantbook_visual_with_ears False \
                --backbone_for_concepts_pretraining backbone_for_concepts_three_gold \
                --concept_head_pretraining concept_head_three_gold \
                --concept_head_architecture three \
                --gallery_correction_fn 'sighting_${probability}' \
                --query_correction_fn 'image_${probability}' \
                --elephantbook_backbone_weight 0.5 \
                --elephantbook_wildcard_distance 0.6 \
                --elephantbook_query_chunk_size 512 \
                --batch_size 64 \
                --wandb_name 'ElephantBookBody_gSighting_qImage_${pct}cor_bw0.5' \
                --wandb_group 'ElephantBook_body_mega_baseline'

            python '${REPO_ROOT}/methods/run_experiment.py' \
                --pipeline_name ElephantBook \
                --dataset mara \
                --subset 2+encounters \
                --backbone_pretraining MegaDescriptor_complete_gold \
                --backbone_layer_norm True \
                --elephantbook_visual_with_ears False \
                --backbone_for_concepts_pretraining backbone_for_concepts_three_gold \
                --concept_head_pretraining concept_head_three_gold \
                --concept_head_architecture three \
                --gallery_correction_fn 'image_${probability}' \
                --query_correction_fn 'image_${probability}' \
                --elephantbook_backbone_weight 0.5 \
                --elephantbook_wildcard_distance 0.6 \
                --elephantbook_query_chunk_size 512 \
                --batch_size 64 \
                --wandb_name 'ElephantBookBody_gImage_qImage_${pct}cor_bw0.5' \
                --wandb_group 'ElephantBook_body_mega_baseline'
        "
done
