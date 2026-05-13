#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=print_proto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:01:00
#SBATCH --output=logs_proto/print_proto_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

python new_stuff/print_proto.py
