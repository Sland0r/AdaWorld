#!/usr/bin/env python3
"""
Train a model to predict undersampled image from latent actions.
Latents: latent_actions_dump/*/latent_actions.pt -> z_mu (N latents)
Targets: color_dump/*/color_step1/*.npz -> blocks (downscaled 32x32 image, N+1 frames, ignore first)
Note: Color dump has 1 extra frame, so we skip first frame to align with latent pairs.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm


def build_mlp(in_dim, out_dim, n_hidden, hidden_dim=256):
    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


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


def load_undersampled_images(base_dir, num_samples, cache_dir='/scratch-shared/FoMo-Atomic-Actions/_cache'):
    """Load downscaled image blocks.
    Uses pre-cached .pt file if available, otherwise falls back to individual .npz files.
    Color dump has N+1 frames, we take blocks [1:] to skip first frame."""
    cache_path = os.path.join(cache_dir, 'color_blocks.pt')
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        blocks = torch.load(cache_path, map_location='cpu')
        # Skip first frame (index 0), take next num_samples
        blocks = blocks[1:num_samples+1]
        return blocks

    # Fallback: load individual files
    print("  Cache not found, loading individual .npz files (slow)...")
    files = sorted(glob.glob(os.path.join(base_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise RuntimeError(f"No .npz files found in {base_dir}")
    
    blocks = []
    for f in files[1:num_samples+1]:  # Skip first frame, take next N
        try:
            data = np.load(f)
            block = data['blocks']  # shape (32, 32, 3) uint8
            block = block.reshape(-1).astype(np.float32) / 255.0  # Normalize and flatten
            blocks.append(torch.from_numpy(block).float())
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    if not blocks:
        raise RuntimeError(f"No usable blocks found in {base_dir}")
    return torch.stack(blocks, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Train MLP to predict undersampled image from latent actions")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dump_dir', type=int, choices=[1, 2], default=1, help='1=latent_actions_dump, 2=latent_actions_dump_2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline', action='store_true', help='Use random inputs instead of latent actions')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve dump directory
    repo_root = Path(__file__).resolve().parents[1]
    latent_base = str(repo_root / ('latent_actions_dump' if args.dump_dir == 1 else 'latent_actions_dump_2'))
    color_base = '/scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data'

    print("Loading latent actions...")
    z = load_latent_actions(latent_base)
    print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

    if args.baseline:
        print("BASELINE MODE: replacing latent actions with random tensors")
        z = torch.randn_like(z)

    print("Loading undersampled image targets...")
    image = load_undersampled_images(color_base, z.shape[0])
    print(f"Loaded {image.shape[0]} image blocks, output dim={image.shape[1]}")

    # Ensure alignment
    if z.shape[0] != image.shape[0]:
        min_samples = min(z.shape[0], image.shape[0])
        z = z[:min_samples]
        image = image[:min_samples]
        print(f"Aligned to {min_samples} samples")

    # Train/test split
    test_ratio = 0.2
    n_test = max(1, int(z.shape[0] * test_ratio))
    indices = torch.randperm(z.shape[0])
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_z, train_image = z[train_idx], image[train_idx]
    test_z, test_image = z[test_idx], image[test_idx]

    train_dataset = TensorDataset(train_z, train_image)
    test_dataset = TensorDataset(test_z, test_image)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = build_mlp(z.shape[1], image.shape[1], args.hidden_layers, args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc='Training')
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for batch_z, batch_image in train_loader:
            batch_z = batch_z.to(device)
            batch_image = batch_image.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            loss = criterion(pred, batch_image)
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
        for batch_z, batch_image in test_loader:
            batch_z = batch_z.to(device)
            batch_image = batch_image.to(device)
            pred = model(batch_z)
            loss = criterion(pred, batch_image)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test MSE: {test_loss:.6f}")


if __name__ == '__main__':
    main()
