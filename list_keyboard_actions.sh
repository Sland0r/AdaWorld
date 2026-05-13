#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=list_keys
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs_proto/list_keyboard_actions_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python new_stuff/list_keyboard_actions.py
