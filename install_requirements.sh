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

# activate env
source activate adaworld

# install requirements
pip install -r requirements.txt

# needed for H100
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
