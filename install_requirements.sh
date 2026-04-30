#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=install_reqirements
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_reqirements_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

# create env
conda create -n adaworld python=3.10 -y

# explicit python path
PYTHON=~/.conda/envs/adaworld/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# upgrade pip first (ensures pre-built wheels are found)
$PYTHON -m pip install --upgrade pip

# install torch for H100 (cu118) first
$PYTHON -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# install remaining requirements
$PYTHON -m pip install -r requirements.txt

