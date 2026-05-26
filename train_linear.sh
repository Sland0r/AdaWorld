#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_linear_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld


python new_stuff/train_linear.py \
    --epochs 50 \
    --batch_size 256 \
    --action_hidden_layers 1 \
    --game_hidden_layers 1 \
    --dataset olafworld \
    --dump-dir 2 \