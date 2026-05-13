#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=extract_p2p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/extract_p2p_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

$PYTHON new_stuff/extract_latent_actions.py \
    --p2p-dir /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset \
    --quiet \
    --mu_only \
    --batch-size 8 \
    --save-dir ./latent_actions_dump_2 \
    --model olafworld
