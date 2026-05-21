#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=analyze_skipped
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/analyze_skipped_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

echo "Running analysis on skipped latents..."
$PYTHON analyze_skipped.py --dataset olafworld

# echo "Running analysis on OlafWorld skipped latents..."
# $PYTHON analyze_skipped.py --dataset olafworld
