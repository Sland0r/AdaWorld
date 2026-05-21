#!/usr/bin/env python3
"""
Pack per-sample spatial targets into sharded .pt files for faster training I/O.

Targets supported:
  - difference dump (.npy, H x W x 3, int16)
  - optical flow dump (.npy, H x W x 2, float32)

Each shard stores:
  - z: latent actions for the shard
  - target: target tensor for the shard in CHW layout
"""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_TARGET_ROOTS = {
    "difference": "/scratch-shared/FoMo-Atomic-Actions/difference_dump/random_actions_data",
    "flow": "/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data",
}

DEFAULT_CACHE_FILES = {
    "difference": "difference_files.txt",
    "flow": "flow_files.txt",
}


def load_latent_actions(base_dir: str) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True))
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found in {base_dir}")

    all_z: List[torch.Tensor] = []
    for path in files:
        data = torch.load(path, map_location="cpu")
        z = data["z_mu"]
        if z.ndim == 1:
            z = z.unsqueeze(0)
        all_z.append(z)

    if not all_z:
        raise RuntimeError(f"No usable latent actions found in {base_dir}")
    return torch.cat(all_z, dim=0)


def get_target_files(target_kind: str, target_root: str, cache_dir: str) -> List[str]:
    cache_path = os.path.join(cache_dir, DEFAULT_CACHE_FILES[target_kind])
    if os.path.exists(cache_path):
        print(f"Loading file list from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as handle:
            files = [line.strip() for line in handle if line.strip()]
        if files:
            return files
        print("Cache file is empty, falling back to glob.")

    print("Cache not found, globbing (slow)...")
    files = sorted(glob.glob(os.path.join(target_root, "**", "*.npy"), recursive=True))
    if not files:
        raise RuntimeError(f"No .npy files found in {target_root}")
    return files


def load_target_tensor(path: str) -> torch.Tensor:
    try:
        arr = np.load(path)  # H x W x C
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D target tensor at {path}, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"Empty array at {path}")
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    except Exception as e:
        raise ValueError(f"Failed to load {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sharded .pt files for difference/flow training targets")
    parser.add_argument("--target_kind", choices=["difference", "flow"], required=True)
    parser.add_argument(
        "--target_root",
        type=str,
        default=None,
        help="Root directory containing per-sample .npy targets",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/_cache",
        help="Directory containing cached file lists (difference_files.txt / flow_files.txt)",
    )
    parser.add_argument(
        "--shard_root",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/sharded_targets",
        help="Root output directory for shard files",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1024,
        help="Number of samples per output shard file",
    )
    parser.add_argument(
        "--dump_dir",
        type=int,
        choices=[1, 2],
        default=1,
        help="1=latent_actions_dump, 2=latent_actions_dump_2",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start offset in the aligned (latent,target) stream",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to shard after start_index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite shard files that already exist",
    )
    args = parser.parse_args()

    if args.samples_per_shard <= 0:
        raise ValueError("--samples_per_shard must be > 0")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    target_root = args.target_root or DEFAULT_TARGET_ROOTS[args.target_kind]
    repo_root = Path(__file__).resolve().parents[1]
    latent_root = repo_root / ("latent_actions_dump" if args.dump_dir == 1 else "latent_actions_dump_2")

    print(f"Target kind: {args.target_kind}")
    print(f"Target root: {target_root}")
    print(f"Latent root: {latent_root}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Shard root:  {args.shard_root}")

    print("Loading latent actions...")
    z = load_latent_actions(str(latent_root))
    print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

    print("Loading target file list...")
    target_files = get_target_files(args.target_kind, target_root, args.cache_dir)
    print(f"Found {len(target_files)} target files")

    aligned_total = min(z.shape[0], len(target_files))
    if aligned_total == 0:
        raise RuntimeError("No aligned samples between latent actions and targets")

    if args.start_index >= aligned_total:
        raise ValueError(f"--start_index ({args.start_index}) must be < aligned sample count ({aligned_total})")

    effective_total = aligned_total - args.start_index
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be > 0 when provided")
        effective_total = min(effective_total, args.max_samples)

    slice_start = args.start_index
    slice_end = args.start_index + effective_total
    z = z[slice_start:slice_end].contiguous()
    target_files = target_files[slice_start:slice_end]

    print(f"Aligned samples: {aligned_total}")
    print(f"Using sample window: [{slice_start}, {slice_end}) -> {effective_total} samples")

    out_dir = Path(args.shard_root) / args.target_kind / f"dump_dir_{args.dump_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    shard_count = math.ceil(effective_total / args.samples_per_shard)
    print(f"Writing {shard_count} shards with {args.samples_per_shard} samples/shard")

    first_shape = None
    first_dtype = None
    written_shards = 0
    skipped_shards = 0
    shard_sizes: List[int] = []

    t0 = time.time()
    for shard_idx in tqdm(range(shard_count), desc="Sharding"):
        rel_start = shard_idx * args.samples_per_shard
        rel_end = min((shard_idx + 1) * args.samples_per_shard, effective_total)
        global_start = slice_start + rel_start
        global_end = slice_start + rel_end
        shard_sizes.append(rel_end - rel_start)

        shard_path = out_dir / f"{args.target_kind}_shard_{shard_idx:05d}.pt"
        if shard_path.exists() and not args.overwrite:
            skipped_shards += 1
            continue

        chunk_files = target_files[rel_start:rel_end]
        valid_tensors = []
        valid_z = []
        for i, path in enumerate(chunk_files):
            try:
                tensor = load_target_tensor(path)
                valid_tensors.append(tensor)
                valid_z.append(z[rel_start + i])
            except ValueError as e:
                print(f"Skipping corrupted file: {e}")
        
        if not valid_tensors:
            print(f"Shard {shard_idx}: no valid files, skipping")
            shard_sizes[-1] = 0
            continue
        
        target_chunk = torch.stack(valid_tensors, dim=0)
        z_chunk = torch.stack(valid_z, dim=0).clone()
        shard_sizes[-1] = len(valid_tensors)

        if first_shape is None:
            first_shape = list(target_chunk.shape[1:])
            first_dtype = str(target_chunk.dtype)

        payload = {
            "z": z_chunk,
            "target": target_chunk,
            "target_kind": args.target_kind,
            "index_start": global_start,
            "index_end": global_end,
        }
        torch.save(payload, shard_path)
        written_shards += 1

    elapsed = time.time() - t0
    manifest = {
        "target_kind": args.target_kind,
        "target_root": target_root,
        "latent_root": str(latent_root),
        "cache_dir": args.cache_dir,
        "samples_per_shard": args.samples_per_shard,
        "dump_dir": args.dump_dir,
        "start_index": slice_start,
        "end_index_exclusive": slice_end,
        "num_samples": effective_total,
        "num_shards": shard_count,
        "written_shards": written_shards,
        "skipped_existing_shards": skipped_shards,
        "shard_sizes": shard_sizes,
        "latent_dim": int(z.shape[1]),
        "sample_shape_chw": first_shape,
        "sample_dtype": first_dtype,
        "elapsed_seconds": elapsed,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
