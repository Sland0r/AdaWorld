#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=color_dump
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/color_dump_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python new_stuff/build_color_dump.py \
    --src-root /home/scur0531/random_actions_data/dataset/retro_act_v0.0.0_random \
    --dst-root /scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data \
    --block-size 32
