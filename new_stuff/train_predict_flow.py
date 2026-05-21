#!/usr/bin/env python3
"""
Train a CNN decoder to predict optical flow from latent actions.

Supports two data formats:
  - Sharded .pt files (recommended): /scratch-shared/FoMo-Atomic-Actions/sharded_targets/flow/dump_dir_*/
  - Legacy per-sample .npy files: optic_flow_dump/*/optic_flow_step1/*.npy
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class CNNDecoder(nn.Module):
    """Decode a latent vector into a spatial image-like output via transposed convolutions."""

    def __init__(self, z_dim, out_channels, target_h, target_w, base_channels=64):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.init_h = 8
        self.init_w = 8
        self.init_channels = base_channels * 8

        self.fc = nn.Sequential(
            nn.Linear(z_dim, self.init_channels * self.init_h * self.init_w),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(base_channels // 2, out_channels, 3, 1, 1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        return x


def load_latent_actions(base_dir):
    files = sorted(glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True))
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found in {base_dir}")

    all_z = []
    for path in files:
        data = torch.load(path, map_location="cpu")
        z = data["z_mu"]
        if z.ndim == 1:
            z = z.unsqueeze(0)
        all_z.append(z)

    if not all_z:
        raise RuntimeError(f"No usable latent actions found in {base_dir}")
    return torch.cat(all_z, dim=0)


def get_file_list(base_dir, extension, cache_dir="/scratch-shared/FoMo-Atomic-Actions/_cache"):
    cache_path = os.path.join(cache_dir, "flow_files.txt")
    if os.path.exists(cache_path):
        print(f"  Loading file list from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    print("  Cache not found, globbing (slow)...")
    files = sorted(glob.glob(os.path.join(base_dir, "**", f"*.{extension}"), recursive=True))
    if not files:
        raise RuntimeError(f"No .{extension} files found in {base_dir}")
    return files


class LazyNpyDataset(torch.utils.data.Dataset):
    """Lazily loads .npy files from a list."""

    def __init__(self, file_list, z_tensor, permute_dims=(2, 0, 1)):
        self.file_list = file_list
        self.z_tensor = z_tensor
        self.permute_dims = permute_dims

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        arr = np.load(self.file_list[idx])  # (H, W, C)
        tensor = torch.from_numpy(arr).float().permute(*self.permute_dims)
        return self.z_tensor[idx], tensor


def get_shard_dir(shard_root: str, dump_dir: int) -> str:
    return os.path.join(shard_root, "flow", f"dump_dir_{dump_dir}")


def get_shard_files(shard_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(shard_dir, "flow_shard_*.pt")))


def load_manifest(shard_dir: str) -> Optional[dict]:
    manifest_path = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_shard_sizes(manifest: Optional[dict], shard_files: Sequence[str]) -> Optional[List[int]]:
    if manifest is None:
        return None

    shard_sizes = manifest.get("shard_sizes")
    if isinstance(shard_sizes, list) and len(shard_sizes) == len(shard_files):
        if all(isinstance(size, int) and size > 0 for size in shard_sizes):
            return shard_sizes

    samples_per_shard = manifest.get("samples_per_shard")
    num_samples = manifest.get("num_samples")
    num_shards = manifest.get("num_shards")
    if not isinstance(samples_per_shard, int) or samples_per_shard <= 0:
        return None
    if not isinstance(num_samples, int) or num_samples <= 0:
        return None
    if not isinstance(num_shards, int) or num_shards != len(shard_files):
        return None

    full = len(shard_files) - 1
    last_size = num_samples - full * samples_per_shard
    if len(shard_files) == 1:
        last_size = num_samples
    if last_size <= 0:
        return None
    return [samples_per_shard] * full + [last_size]


def scan_shard_sizes(shard_files: Sequence[str]) -> List[int]:
    sizes = []
    for path in tqdm(shard_files, desc="Scanning shard sizes"):
        payload = torch.load(path, map_location="cpu")
        z_chunk = payload["z"]
        target_chunk = payload["target"]
        if z_chunk.shape[0] != target_chunk.shape[0]:
            raise RuntimeError(f"Mismatched z/target lengths in {path}")
        sizes.append(int(z_chunk.shape[0]))
    return sizes


def build_shard_splits(
    shard_files: Sequence[str],
    shard_sizes: Sequence[int],
    seed: int,
    test_ratio: float = 0.2,
) -> Tuple[List[Tuple[str, torch.Tensor]], List[Tuple[str, torch.Tensor]], int]:
    total_samples = int(sum(shard_sizes))
    if total_samples <= 1:
        raise RuntimeError("Need at least 2 samples in shards for train/test split")

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)

    n_test = max(1, int(total_samples * test_ratio))
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask[indices[:n_test]] = True
    train_mask = ~test_mask

    train_splits: List[Tuple[str, torch.Tensor]] = []
    test_splits: List[Tuple[str, torch.Tensor]] = []
    offset = 0
    for path, size in zip(shard_files, shard_sizes):
        shard_train = torch.nonzero(train_mask[offset:offset + size], as_tuple=False).squeeze(1)
        shard_test = torch.nonzero(test_mask[offset:offset + size], as_tuple=False).squeeze(1)
        if shard_train.numel() > 0:
            train_splits.append((path, shard_train))
        if shard_test.numel() > 0:
            test_splits.append((path, shard_test))
        offset += size

    train_count = int(train_mask.sum().item())
    test_count = int(test_mask.sum().item())
    print(f"Train: {train_count}, Test: {test_count}")
    return train_splits, test_splits, total_samples


def load_shard_shape_and_dim(first_shard: str) -> Tuple[int, int, int, int]:
    payload = torch.load(first_shard, map_location="cpu")
    z_chunk = payload["z"]
    target_chunk = payload["target"]
    if z_chunk.ndim != 2 or target_chunk.ndim != 4:
        raise RuntimeError(f"Unexpected tensor shapes in shard {first_shard}")
    z_dim = int(z_chunk.shape[1])
    out_channels = int(target_chunk.shape[1])
    target_h = int(target_chunk.shape[2])
    target_w = int(target_chunk.shape[3])
    return z_dim, out_channels, target_h, target_w


def train_one_epoch_sharded(model, train_splits, batch_size, device, criterion, optimizer, baseline):
    model.train()
    total_loss = 0.0
    total_batches = 0

    shard_order = list(train_splits)
    random.shuffle(shard_order)
    for shard_path, local_indices in shard_order:
        payload = torch.load(shard_path, map_location="cpu")
        z_chunk = payload["z"]
        target_chunk = payload["target"].float()
        if baseline:
            z_chunk = torch.randn_like(z_chunk)

        shuffled = local_indices[torch.randperm(local_indices.numel())]
        for start in range(0, shuffled.numel(), batch_size):
            idx = shuffled[start:start + batch_size]
            batch_z = z_chunk[idx].to(device)
            batch_target = target_chunk[idx].to(device)

            optimizer.zero_grad()
            pred = model(batch_z)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No training batches produced from sharded data")
    return total_loss / total_batches


def evaluate_sharded(model, test_splits, batch_size, device, criterion, baseline):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for shard_path, local_indices in test_splits:
            payload = torch.load(shard_path, map_location="cpu")
            z_chunk = payload["z"]
            target_chunk = payload["target"].float()
            if baseline:
                z_chunk = torch.randn_like(z_chunk)

            for start in range(0, local_indices.numel(), batch_size):
                idx = local_indices[start:start + batch_size]
                batch_z = z_chunk[idx].to(device)
                batch_target = target_chunk[idx].to(device)
                pred = model(batch_z)
                loss = criterion(pred, batch_target)
                total_loss += loss.item()
                total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No evaluation batches produced from sharded data")
    return total_loss / total_batches


def train_legacy_npy(args, device, criterion):
    repo_root = Path(__file__).resolve().parents[1]
    latent_base = str(repo_root / ("latent_actions_dump" if args.dump_dir == 1 else "latent_actions_dump_2"))
    flow_base = args.target_root

    print("Loading latent actions...")
    z = load_latent_actions(latent_base)
    print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

    if args.baseline:
        print("BASELINE MODE: replacing latent actions with random tensors")
        z = torch.randn_like(z)

    print("Loading optical flow file list...")
    flow_files = get_file_list(flow_base, "npy", cache_dir=args.cache_dir)
    print(f"Found {len(flow_files)} flow files")

    num_samples = min(z.shape[0], len(flow_files))
    z = z[:num_samples]
    flow_files = flow_files[:num_samples]
    print(f"Aligned to {num_samples} samples")

    sample = np.load(flow_files[0])
    out_channels = sample.shape[2]
    target_h = sample.shape[0]
    target_w = sample.shape[1]
    print(f"Target shape: ({out_channels}, {target_h}, {target_w})")

    test_ratio = 0.2
    n_test = max(1, int(num_samples * test_ratio))
    indices = torch.randperm(num_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_files = [flow_files[i] for i in train_idx.tolist()]
    test_files = [flow_files[i] for i in test_idx.tolist()]
    train_dataset = LazyNpyDataset(train_files, z[train_idx])
    test_dataset = LazyNpyDataset(test_files, z[test_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = CNNDecoder(z.shape[1], out_channels, target_h, target_w, base_channels=args.base_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN decoder params: {num_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc="Training")
    for _ in pbar:
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_z, batch_flow in train_loader:
            batch_z = batch_z.to(device)
            batch_flow = batch_flow.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            loss = criterion(pred, batch_flow)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        pbar.set_postfix(loss=f"{(total_loss / max(batch_count, 1)):.6f}")

    print("Evaluating...")
    model.eval()
    test_loss = 0.0
    test_batches = 0
    with torch.no_grad():
        for batch_z, batch_flow in test_loader:
            batch_z = batch_z.to(device)
            batch_flow = batch_flow.to(device)
            pred = model(batch_z)
            loss = criterion(pred, batch_flow)
            test_loss += loss.item()
            test_batches += 1
    print(f"Test MSE: {(test_loss / max(test_batches, 1)):.6f}")


def train_sharded(args, device, criterion):
    shard_dir = get_shard_dir(args.shard_root, args.dump_dir)
    shard_files = get_shard_files(shard_dir)
    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    print(f"Using sharded targets from: {shard_dir}")
    print(f"Found {len(shard_files)} shards")

    manifest = load_manifest(shard_dir)
    shard_sizes = infer_shard_sizes(manifest, shard_files)
    if shard_sizes is None:
        print("Manifest is missing or incomplete for shard sizes, scanning shard files once...")
        shard_sizes = scan_shard_sizes(shard_files)
    else:
        print("Using shard sizes from manifest.")

    z_dim, out_channels, target_h, target_w = load_shard_shape_and_dim(shard_files[0])
    print(f"Target shape: ({out_channels}, {target_h}, {target_w})")

    train_splits, test_splits, total_samples = build_shard_splits(shard_files, shard_sizes, seed=args.seed)
    print(f"Aligned to {total_samples} samples")

    model = CNNDecoder(z_dim, out_channels, target_h, target_w, base_channels=args.base_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN decoder params: {num_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc="Training")
    for _ in pbar:
        avg_loss = train_one_epoch_sharded(
            model=model,
            train_splits=train_splits,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            baseline=args.baseline,
        )
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    print("Evaluating...")
    test_loss = evaluate_sharded(
        model=model,
        test_splits=test_splits,
        batch_size=args.batch_size,
        device=device,
        criterion=criterion,
        baseline=args.baseline,
    )
    print(f"Test MSE: {test_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN decoder to predict optical flow from latent actions")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count for CNN decoder")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dump_dir", type=int, choices=[1, 2], default=1, help="1=latent_actions_dump, 2=latent_actions_dump_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Use random inputs instead of latent actions")
    parser.add_argument("--data_format", choices=["auto", "sharded", "npy"], default="auto")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for npy mode")
    parser.add_argument("--cache_dir", type=str, default="/scratch-shared/FoMo-Atomic-Actions/_cache")
    parser.add_argument(
        "--target_root",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data",
        help="Root for legacy per-sample .npy targets",
    )
    parser.add_argument(
        "--shard_root",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/sharded_targets",
        help="Root for sharded targets",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    shard_files = get_shard_files(get_shard_dir(args.shard_root, args.dump_dir))
    use_sharded = args.data_format == "sharded" or (args.data_format == "auto" and len(shard_files) > 0)

    if use_sharded:
        train_sharded(args, device, criterion)
    else:
        if args.data_format == "sharded":
            raise RuntimeError("Requested --data_format sharded, but no shard files were found")
        print("Sharded data not found, falling back to legacy .npy mode.")
        train_legacy_npy(args, device, criterion)


if __name__ == "__main__":
    main()
