#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=optic_flow_dump
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/optic_flow_dump_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python new_stuff/build_optic_flow_dump.py \
    --src-root /home/scur0531/random_actions_data/dataset/retro_act_v0.0.0_random \
    --dst-root /scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data \
    --frame-step 1 \
    --pair-stride 1 \
    --seed 42