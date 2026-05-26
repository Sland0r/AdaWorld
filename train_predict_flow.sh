#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_flow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs_flex/train_predict_flow_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python -u new_stuff/train_predict_flow.py \
    --epochs 50 \
    --base_channels 64 \
    --batch_size 264 \
    --lr 1e-3 \
    --dump_dir 3 \
    --model adaworld \
    --seed 42 \
    #--baseline