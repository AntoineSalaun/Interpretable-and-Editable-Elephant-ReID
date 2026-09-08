#!/bin/bash
set -euo pipefail

ROOT=/data/vision/beery/scratch/antoine/CBM_reid
LOG_DIR="${ROOT}/bash/slurm_logs"
GROUP=MiewID_Mara_stability_sweep_20260520_fix_freeze

mkdir -p "${LOG_DIR}"

submit() {
  local short_name="$1"
  local lr="$2"
  local wd="$3"
  local epochs="$4"
  local wandb_name="MiewID_stability_fix_freeze_${short_name}_lr${lr}_wd${wd}_e${epochs}"

  sbatch --parsable \
    --job-name="MiewID_${short_name}" \
    --partition=vision-beery \
    --qos=vision-beery-main \
    --account=vision-beery \
    --time=7-0:00:00 \
    --mem=64G \
    --gres=gpu:1 \
    --chdir="${ROOT}" \
    --output="${LOG_DIR}/${wandb_name}.%j.out" \
    --wrap="python ${ROOT}/methods/run_experiment.py --pipeline_name MiewID --miewid_mode finetuning --epochs ${epochs} --dataset mara --subset 2+encounters --backbone_pretraining out_of_box --backbone_lr ${lr} --backbone_wd ${wd} --print_every 5 --batch_size 64 --save_best True --wandb_group ${GROUP} --wandb_name ${wandb_name}"
}

submit "01" "1e-6" "1e-5" "75"
submit "02" "3e-7" "1e-5" "75"
submit "03" "1e-7" "1e-5" "75"
submit "04" "1e-6" "1e-4" "75"
submit "05" "3e-7" "1e-4" "75"
submit "06" "1e-7" "1e-4" "75"
