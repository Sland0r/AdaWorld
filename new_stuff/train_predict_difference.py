#!/usr/bin/env python3
"""
Train a CNN decoder to predict frame differences from latent actions.
Latents: latent_actions_dump/*/latent_actions.pt -> z_mu
Targets: difference_dump/*/difference_step1/*.npy -> raw difference tensors (H, W, 3)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import os
from pathlib import Path
import numpy as np
import random
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
            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),
            # Final conv to output channels
            nn.Conv2d(base_channels // 2, out_channels, 3, 1, 1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.target_h, self.target_w),
                          mode='bilinear', align_corners=False)
        return x


def load_latent_actions(base_dir):
    """Load all latent actions from latent_actions_dump."""
    files = sorted(glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True))
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found in {base_dir}")

    all_z = []
    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
            z = data['z_mu']
            if z.ndim == 1:
                z = z.unsqueeze(0)
            all_z.append(z)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not all_z:
        raise RuntimeError(f"No usable latent actions found in {base_dir}")
    return torch.cat(all_z, dim=0)


def load_difference_tensors(base_dir, num_samples):
    """Load frame difference tensors as (N, 3, H, W)."""
    files = sorted(glob.glob(os.path.join(base_dir, "**", "*.npy"), recursive=True))
    if not files:
        raise RuntimeError(f"No .npy files found in {base_dir}")

    diffs = []
    for f in files[:num_samples]:  # Align to latent count
        try:
            diff = np.load(f)  # shape (H, W, 3) - int16
            # Convert to (3, H, W) float
            diff = torch.from_numpy(diff).float().permute(2, 0, 1)
            diffs.append(diff)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not diffs:
        raise RuntimeError(f"No usable difference tensors found in {base_dir}")
    return torch.stack(diffs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train CNN decoder to predict frame differences from latent actions")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_channels', type=int, default=64, help='Base channel count for CNN decoder')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dump_dir', type=int, choices=[1, 2], default=1, help='1=latent_actions_dump, 2=latent_actions_dump_2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve dump directory
    repo_root = Path(__file__).resolve().parents[1]
    latent_base = str(repo_root / ('latent_actions_dump' if args.dump_dir == 1 else 'latent_actions_dump_2'))
    diff_base = '/scratch-shared/FoMo-Atomic-Actions/difference_dump/random_actions_data'

    print("Loading latent actions...")
    z = load_latent_actions(latent_base)
    print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

    print("Loading difference targets...")
    diff = load_difference_tensors(diff_base, z.shape[0])
    print(f"Loaded {diff.shape[0]} difference tensors, shape={diff.shape[1:]}")

    # Ensure alignment
    if z.shape[0] != diff.shape[0]:
        min_samples = min(z.shape[0], diff.shape[0])
        z = z[:min_samples]
        diff = diff[:min_samples]
        print(f"Aligned to {min_samples} samples")

    out_channels = diff.shape[1]  # 3
    target_h = diff.shape[2]
    target_w = diff.shape[3]

    # Train/test split
    test_ratio = 0.2
    n_test = max(1, int(z.shape[0] * test_ratio))
    indices = torch.randperm(z.shape[0])
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_z, train_diff = z[train_idx], diff[train_idx]
    test_z, test_diff = z[test_idx], diff[test_idx]

    train_dataset = TensorDataset(train_z, train_diff)
    test_dataset = TensorDataset(test_z, test_diff)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = CNNDecoder(z.shape[1], out_channels, target_h, target_w,
                       base_channels=args.base_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN decoder params: {num_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc='Training')
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for batch_z, batch_diff in train_loader:
            batch_z = batch_z.to(device)
            batch_diff = batch_diff.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            loss = criterion(pred, batch_diff)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f'{avg_loss:.6f}')

    # Evaluate
    print("Evaluating...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_z, batch_diff in test_loader:
            batch_z = batch_z.to(device)
            batch_diff = batch_diff.to(device)
            pred = model(batch_z)
            loss = criterion(pred, batch_diff)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test MSE: {test_loss:.6f}")


if __name__ == '__main__':
    main()
