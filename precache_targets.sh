#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=precache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/precache_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python -u new_stuff/precache_targets.py
