#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=extract_skipped_latents
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_skipped_latents_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

VIDEO_PATH=/scratch-shared/scur0531/skipped_frames_v0.0.0
$PYTHON new_stuff/extract_skipped_latents.py --video $VIDEO_PATH \
        --quiet \
        --mu_only \
        --save-dir ./latent_actions_skipped \
        --model adaworld

