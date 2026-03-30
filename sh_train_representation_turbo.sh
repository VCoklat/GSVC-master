#!/bin/bash

#SBATCH --job-name=T_rep_turbo    # Job name
#SBATCH --output=output/T_rep_turbo_output.txt
#SBATCH --error=output/T_rep_turbo_error.txt
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg
#SBATCH --mem=16G

source ~/.bashrc
conda activate torch

# ═══════════════════════════════════════════════════════════════════════
# GSVC-Turbo Representation Training
# ═══════════════════════════════════════════════════════════════════════
# This script runs the Representation phase with optical flow pre-advection.
# To disable turbo, simply remove the --turbo flag.
#
# Available presets:
#   default               — 540p downscale, 80px flow clamp
#   conservative          — 720p downscale, 40px flow clamp, 20 RAFT iters
#   aggressive            — 480p downscale, 120px flow clamp, 8 RAFT iters
#   ablation_no_clamp     — No flow/position clamping (stability test)
#   ablation_no_downscale — Native resolution RAFT (VRAM stress test)
# ═══════════════════════════════════════════════════════════════════════

datasets=(
  "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv Beauty"
  "/home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv HoneyBee"
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

# ── Turbo Configuration ──────────────────────────────────────────────
TURBO_PRESET="default"     # Change for ablation experiments
TURBO_ENABLED=True
TURBO_VERBOSE=False

# ── Training Configuration ───────────────────────────────────────────
savdir="GaussianVideo_results_turbo"
savdir_m="GaussianVideo_models_turbo"
is_pos=False
is_ad=True
is_rm=True
loss_type="L2"

for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in 10000 20000 30000 40000 50000; do
    for iterations in 100000; do
      pos_flag=""
      ad_flag=""
      rm_flag=""
      turbo_flag=""
      verbose_flag=""

      if [ "$is_pos" = True ]; then pos_flag="--is_pos"; fi
      if [ "$is_ad" = True ]; then ad_flag="--is_ad"; fi
      if [ "$is_rm" = True ]; then rm_flag="--is_rm"; fi
      if [ "$TURBO_ENABLED" = True ]; then
        turbo_flag="--turbo --turbo_preset $TURBO_PRESET"
      fi
      if [ "$TURBO_VERBOSE" = True ]; then verbose_flag="--turbo_verbose"; fi

      srun python train_video_Represent.py \
        --loss_type $loss_type \
        --dataset $dataset_path \
        --data_name $data_name \
        --num_points $num_points \
        --iterations $iterations \
        --savdir $savdir \
        --savdir_m $savdir_m \
        $pos_flag $ad_flag $rm_flag \
        $turbo_flag $verbose_flag
    done
  done
done
