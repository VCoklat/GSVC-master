#!/bin/bash

#SBATCH --job-name=T_comp_turbo
#SBATCH --output=output/T_comp_turbo_output.txt
#SBATCH --error=output/T_comp_turbo_error.txt
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg
#SBATCH --mem=16G

source ~/.bashrc
conda activate torch

# ═══════════════════════════════════════════════════════════════════════
# GSVC-Turbo Compression Training
# ═══════════════════════════════════════════════════════════════════════

datasets=(
  "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv Beauty"
  "/home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv HoneyBee"
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

# ── Turbo Configuration ──────────────────────────────────────────────
TURBO_PRESET="default"
TURBO_ENABLED=True
TURBO_VERBOSE=False

# ── Training Configuration ───────────────────────────────────────────
savdir="GaussianVideo_results_turbo"
savdir_m="GaussianVideo_models_turbo"
loss_type="L2"

for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in 10000 20000 30000 40000 50000; do
    for iterations in 30000; do
      turbo_flag=""
      verbose_flag=""

      if [ "$TURBO_ENABLED" = True ]; then
        turbo_flag="--turbo --turbo_preset $TURBO_PRESET"
      fi
      if [ "$TURBO_VERBOSE" = True ]; then verbose_flag="--turbo_verbose"; fi

      model_path="./checkpoints/GaussianVideo_models_turbo/${data_name}/GaussianVideo_100000_${num_points}/gmodels_state_dict.pth"

      srun python train_video_Compress.py \
        --loss_type $loss_type \
        --dataset $dataset_path \
        --data_name $data_name \
        --num_points $num_points \
        --iterations $iterations \
        --savdir $savdir \
        --savdir_m $savdir_m \
        --model_path $model_path \
        $turbo_flag $verbose_flag
    done
  done
done
