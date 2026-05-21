#!/usr/bin/env python3
"""
Pre-cache targets from scratch-shared into fast-loading formats.

For small per-sample data (color histograms, undersampled blocks):
  -> Pack into single .pt tensor files (~0.9 GB / ~3.4 GB).

For large spatial data (difference, optical flow):
  -> Build sorted file-list caches (.txt) so training scripts skip the
     expensive recursive glob + sort on every run.

Run once, then all train_predict_* scripts will auto-detect the caches.

Usage:
    python new_stuff/precache_targets.py [--cache_dir CACHE_DIR]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


def precache_color(src_dir, cache_dir):
    """Pack color histograms and blocks into single .pt files."""
    hist_cache = os.path.join(cache_dir, "color_histograms.pt")
    block_cache = os.path.join(cache_dir, "color_blocks.pt")

    if os.path.exists(hist_cache) and os.path.exists(block_cache):
        print(f"[color] Caches already exist, skipping:\n  {hist_cache}\n  {block_cache}")
        return

    print(f"[color] Globbing {src_dir} ...")
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.npz"), recursive=True))
    print(f"[color] Found {len(files)} files in {time.time() - t0:.1f}s")

    if not files:
        print("[color] No files found, skipping.")
        return

    histograms = []
    blocks = []
    for f in tqdm(files, desc="[color] Loading"):
        try:
            data = np.load(f)
            histograms.append(torch.from_numpy(data["histogram"]).float())
            blocks.append(torch.from_numpy(data["blocks"].reshape(-1).astype(np.float32) / 255.0))
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    hist_tensor = torch.stack(histograms)
    block_tensor = torch.stack(blocks)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(hist_tensor, hist_cache)
    torch.save(block_tensor, block_cache)
    print(f"[color] Saved {hist_tensor.shape} histograms -> {hist_cache}")
    print(f"[color] Saved {block_tensor.shape} blocks    -> {block_cache}")


def precache_filelist(src_dir, cache_path, extension):
    """Build a sorted file-list for large spatial data (avoids repeated glob+sort)."""
    if os.path.exists(cache_path):
        print(f"[filelist] Cache already exists, skipping: {cache_path}")
        return

    print(f"[filelist] Globbing {src_dir} for *.{extension} ...")
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(src_dir, "**", f"*.{extension}"), recursive=True))
    print(f"[filelist] Found {len(files)} files in {time.time() - t0:.1f}s")

    if not files:
        print("[filelist] No files found, skipping.")
        return

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        for f in files:
            fh.write(f + "\n")
    print(f"[filelist] Saved {len(files)} paths -> {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache targets from scratch-shared")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/_cache",
        help="Directory to store cache files",
    )
    args = parser.parse_args()

    color_src = "/scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data"
    diff_src = "/scratch-shared/FoMo-Atomic-Actions/difference_dump/random_actions_data"
    flow_src = "/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data"

    # 1) Color histograms + blocks -> single .pt files (~0.9 GB + ~3.4 GB)
    precache_color(color_src, args.cache_dir)

    # 2) Difference file list (too large for full tensor cache: ~110 GB)
    precache_filelist(diff_src, os.path.join(args.cache_dir, "difference_files.txt"), "npy")

    # 3) Optical flow file list (too large for full tensor cache: ~147 GB)
    precache_filelist(flow_src, os.path.join(args.cache_dir, "flow_files.txt"), "npy")

    print("\nDone! Training scripts will auto-detect these caches.")


if __name__ == "__main__":
    main()
