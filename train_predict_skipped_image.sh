#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_skipp_image
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_skipp_image_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# echo "Running image on adaworld"
# $PYTHON new_stuff/train_predict_skipped_image.py --model adaworld
echo "Running image on adaworld (baseline)"
$PYTHON new_stuff/train_predict_skipped_image.py --model adaworld --baseline

# echo "Running image on olafworld"
# $PYTHON new_stuff/train_predict_skipped_image.py --model olafworld
echo "Running image on olafworld (baseline)"
$PYTHON new_stuff/train_predict_skipped_image.py --model olafworld --baseline
