#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/extract_%A.out

module purge
module load 2025

tar -xvf /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset/batch_00005.tar -C /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset/dataset/