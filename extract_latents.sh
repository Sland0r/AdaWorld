#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=extract_latents
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/extract_latents_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate adaworld

VIDEO_PATH=/home/scur0531/random_actions_data/dataset/retro_act_v0.0.0
python new_stuff/extract_latent_actions.py --video $VIDEO_PATH \
	--quiet \
	--mu_only \
	--save-dir ./latent_actions_dump


# PATH CAN BE EITHER MP4 OR FOLDER OF FRAMES

# # Extract from two image files
# python extract_latent_actions.py --frame1 frame_a.png --frame2 frame_b.png

# # Extract from a video (consecutive frame pairs)
# python extract_latent_actions.py --video path/to/video.mp4

# # Extract from a video starting at a specific frame
# python extract_latent_actions.py --video path/to/video.mp4 --start-frame 50

# # Extract from every pair in a directory of frames (sorted alphabetically)
# python extract_latent_actions.py --frame-dir path/to/frames/

# # Save outputs to disk
# python extract_latent_actions.py --video path/to/video.mp4 --save-dir outputs/latent_actions/

# # Use a specific checkpoint
# python extract_latent_actions.py --frame1 a.png --frame2 b.png --lam-ckpt /path/to/lam.ckpt