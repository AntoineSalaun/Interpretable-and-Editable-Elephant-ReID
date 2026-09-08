#!/bin/bash
#SBATCH --job-name=cbm-reid-seek
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
##SBATCH --requeue

#SBATCH --time=7-0:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs/seek_retrieval.%j.out
#SBATCH --chdir=/data/vision/beery/scratch/antoine/CBM_reid

set -euo pipefail

mkdir -p /data/vision/beery/scratch/antoine/CBM_reid/bash/slurm_logs

# Correction policy convention:
# - gallery correction is written as sighting_p, so selected sightings are corrected consistently.
# - query correction is written as image_p, so query images are corrected independently.

# SEEK retrieval: gallery=sighting_p, query=image_p ##############################################
# 00% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.0 --query_correction_fn image_0.0 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_00cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 10% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.1 --query_correction_fn image_0.1 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_10cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 20% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.2 --query_correction_fn image_0.2 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_20cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 30% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.3 --query_correction_fn image_0.3 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_30cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 40% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.4 --query_correction_fn image_0.4 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_40cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 50% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.5 --query_correction_fn image_0.5 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_50cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 60% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.6 --query_correction_fn image_0.6 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_60cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 70% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.7 --query_correction_fn image_0.7 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_70cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 80% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.8 --query_correction_fn image_0.8 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_80cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 90% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.9 --query_correction_fn image_0.9 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_90cor_dSeek3 --wandb_group SEEK_retrieval_homemade
# 100% seek_homemade_3 : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_1.0 --query_correction_fn image_1.0 --batch_size 64 --concept_distance seek_homemade_3 --wandb_name Retrieval_gSighting_qImage_100cor_dSeek3 --wandb_group SEEK_retrieval_homemade

# 00% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.0 --query_correction_fn image_0.0 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_00cor_dCos --wandb_group SEEK_retrieval_cosine
# 10% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.1 --query_correction_fn image_0.1 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_10cor_dCos --wandb_group SEEK_retrieval_cosine
# 20% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.2 --query_correction_fn image_0.2 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_20cor_dCos --wandb_group SEEK_retrieval_cosine
# 30% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.3 --query_correction_fn image_0.3 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_30cor_dCos --wandb_group SEEK_retrieval_cosine
# 40% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.4 --query_correction_fn image_0.4 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_40cor_dCos --wandb_group SEEK_retrieval_cosine
# 50% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.5 --query_correction_fn image_0.5 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_50cor_dCos --wandb_group SEEK_retrieval_cosine
# 60% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.6 --query_correction_fn image_0.6 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_60cor_dCos --wandb_group SEEK_retrieval_cosine
# 70% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.7 --query_correction_fn image_0.7 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_70cor_dCos --wandb_group SEEK_retrieval_cosine
# 80% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.8 --query_correction_fn image_0.8 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_80cor_dCos --wandb_group SEEK_retrieval_cosine
# 90% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_0.9 --query_correction_fn image_0.9 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_90cor_dCos --wandb_group SEEK_retrieval_cosine
# 100% cosine_sim : python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py --pipeline_name Retrieval_on_SEEK_codes --dataset mara --subset 2+encounters --gallery_correction_fn sighting_1.0 --query_correction_fn image_1.0 --batch_size 64 --concept_distance cosine_sim --wandb_name Retrieval_gSighting_qImage_100cor_dCos --wandb_group SEEK_retrieval_cosine

for distance in seek_homemade_3 cosine_sim; do
    if [[ "$distance" == "seek_homemade_3" ]]; then
        wandb_group="SEEK_retrieval_homemade"
        distance_suffix="dSeek3"
    else
        wandb_group="SEEK_retrieval_cosine"
        distance_suffix="dCos"
    fi

    for pct in 00 10 20 30 40 50 60 70 80 90 100; do
        probability=$(awk "BEGIN { printf \"%.1f\", $pct / 100 }")
        python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py \
            --pipeline_name Retrieval_on_SEEK_codes \
            --dataset mara \
            --subset 2+encounters \
            --backbone_for_concepts_pretraining backbone_for_concepts_three_gold \
            --concept_head_pretraining concept_head_three_gold \
            --concept_head_architecture three \
            --gallery_correction_fn "sighting_${probability}" \
            --query_correction_fn "image_${probability}" \
            --batch_size 64 \
            --concept_distance "$distance" \
            --wandb_name "Retrieval_gSighting_qImage_${pct}cor_${distance_suffix}" \
            --wandb_group "$wandb_group"
    done
done

# SEEK retrieval for the image-level performance plot: gallery=image_p, query=image_p.
for distance in seek_homemade_3 cosine_sim; do
    if [[ "$distance" == "seek_homemade_3" ]]; then
        wandb_group="SEEK_retrieval_homemade"
        distance_suffix="dSeek3"
    else
        wandb_group="SEEK_retrieval_cosine"
        distance_suffix="dCos"
    fi

    for pct in 00 10 20 30 40 50 60 70 80 90 100; do
        probability=$(awk "BEGIN { printf \"%.1f\", $pct / 100 }")
        python /data/vision/beery/scratch/antoine/CBM_reid/methods/run_experiment.py \
            --pipeline_name Retrieval_on_SEEK_codes \
            --dataset mara \
            --subset 2+encounters \
            --backbone_for_concepts_pretraining backbone_for_concepts_three_gold \
            --concept_head_pretraining concept_head_three_gold \
            --concept_head_architecture three \
            --gallery_correction_fn "image_${probability}" \
            --query_correction_fn "image_${probability}" \
            --batch_size 64 \
            --concept_distance "$distance" \
            --wandb_name "Retrieval_gImage_qImage_${pct}cor_${distance_suffix}" \
            --wandb_group "$wandb_group"
    done
done

# SEEK retrieval table: gallery rows use sighting-level correction; query columns use image-level
# correction plus a full sighting query column. The diagonal image cells above already exist, so
# this block launches only the off-diagonal cells needed by the table.
for distance in seek_homemade_3 cosine_sim; do
    if [[ "$distance" == "seek_homemade_3" ]]; then
        wandb_group="SEEK_retrieval_homemade"
        distance_suffix="dSeek3"
    else
        wandb_group="SEEK_retrieval_cosine"
        distance_suffix="dCos"
    fi

    for gallery in "00:sighting_0.0" "50:sighting_0.5" "100:sighting_1.0"; do
        gallery_label="${gallery%%:*}"
        gallery_policy="${gallery#*:}"

        for query in "Image00:image_0.0" "Image50:image_0.5" "Image100:image_1.0" "Sighting100:sighting_1.0"; do
            query_label="${query%%:*}"
            query_policy="${query#*:}"

            if [[ "$query_policy" == "image_${gallery_policy#sighting_}" ]]; then
                continue
            fi

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
        done
    done
done
