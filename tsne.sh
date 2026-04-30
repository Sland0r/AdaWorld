#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=tsne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/tsne_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

$PYTHON new_stuff/plot_tsne.py