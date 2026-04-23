#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=download_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_dataset_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate adaworld

DOWNLOAD_DIR="rtx"
DATASET_TRANSFORMS=(
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
)

for tuple in "${DATASET_TRANSFORMS[@]}"; do
  # Extract strings from the tuple
  strings=($tuple)
  DATASET=${strings[0]}
  gsutil -m cp -r gs://gresearch/robotics/${DATASET} ${DOWNLOAD_DIR}/
done
