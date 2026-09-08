#!/bin/bash
#SBATCH --job-name=cbm-reid-seek-grid
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --time=7-0:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%8
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/seek_retrieval_grid.%A_%a.out
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

mkdir -p /data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs

# Missing cells for the expanded SEEK-only table:
# gallery rows are sighting_0, sighting_0.25, sighting_0.5, sighting_0.75, sighting_1.0;
# query columns are image_0, image_0.25, image_0.5, image_0.75, image_1.0, sighting_1.0.
# Existing 0/50/100 cells are intentionally omitted.

RUN_SPECS=(
  "cosine_sim|SEEK_retrieval_cosine|dCos|00|sighting_0.0|Image25|image_0.25"
  "cosine_sim|SEEK_retrieval_cosine|dCos|00|sighting_0.0|Image75|image_0.75"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Image00|image_0.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Image25|image_0.25"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Image50|image_0.5"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Image75|image_0.75"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Image100|image_1.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|25|sighting_0.25|Sighting100|sighting_1.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|50|sighting_0.5|Image25|image_0.25"
  "cosine_sim|SEEK_retrieval_cosine|dCos|50|sighting_0.5|Image75|image_0.75"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Image00|image_0.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Image25|image_0.25"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Image50|image_0.5"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Image75|image_0.75"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Image100|image_1.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|75|sighting_0.75|Sighting100|sighting_1.0"
  "cosine_sim|SEEK_retrieval_cosine|dCos|100|sighting_1.0|Image25|image_0.25"
  "cosine_sim|SEEK_retrieval_cosine|dCos|100|sighting_1.0|Image75|image_0.75"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|00|sighting_0.0|Image25|image_0.25"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|00|sighting_0.0|Image75|image_0.75"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Image00|image_0.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Image25|image_0.25"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Image50|image_0.5"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Image75|image_0.75"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Image100|image_1.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|25|sighting_0.25|Sighting100|sighting_1.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|50|sighting_0.5|Image25|image_0.25"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|50|sighting_0.5|Image75|image_0.75"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Image00|image_0.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Image25|image_0.25"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Image50|image_0.5"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Image75|image_0.75"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Image100|image_1.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|75|sighting_0.75|Sighting100|sighting_1.0"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|100|sighting_1.0|Image25|image_0.25"
  "seek_homemade_3|SEEK_retrieval_homemade|dSeek3|100|sighting_1.0|Image75|image_0.75"
)

IFS="|" read -r distance wandb_group distance_suffix gallery_label gallery_policy query_label query_policy <<< "${RUN_SPECS[$SLURM_ARRAY_TASK_ID]}"

python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py \
  --pipeline_name Retrieval_on_SEEK_codes \
  --dataset mara \
  --subset 2+encounters \
  --backbone_for_concepts_pretraining backbone_for_concepts_three_gold \
  --concept_head_pretraining concept_head_three_gold \
  --concept_head_architecture three \
  --gallery_correction_fn "$gallery_policy" \
  --query_correction_fn "$query_policy" \
  --batch_size 64 \
  --concept_distance "$distance" \
  --wandb_name "Retrieval_gSighting${gallery_label}_q${query_label}_${distance_suffix}" \
  --wandb_group "$wandb_group"
