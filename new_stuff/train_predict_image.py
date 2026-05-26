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
import cv2
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


def downscale_to_blocks(image_bgr, size=32):
    """Downscale an image to size x size and flatten."""
    blocks = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return blocks.reshape(-1).astype(np.float32) / 255.0


def load_p2p_data_image(model_name, cache_dir='/scratch-shared/FoMo-Atomic-Actions/_cache/p2p',
                        p2p_video_root='/scratch-shared/FoMo-Atomic-Actions/open-p2p-subset',
                        force_extract=False):
    """Load latent actions + undersampled image targets from P2P videos.
    Caches results in scratch-shared. Modeled after train_predict_skipped_image.py."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_z = os.path.join(cache_dir, f'{model_name}_z_mu_img.pt')
    cache_t = os.path.join(cache_dir, f'{model_name}_image_targets.pt')

    if not force_extract and os.path.exists(cache_z) and os.path.exists(cache_t):
        print(f"Loading cached P2P image data for {model_name}...")
        return torch.load(cache_z, map_location='cpu'), torch.load(cache_t, map_location='cpu')

    print(f"Extracting image targets from P2P videos for {model_name}...")
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = str(repo_root / 'latent_actions_dump_2' / model_name)
    files = sorted(glob.glob(os.path.join(base_dir, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found in {base_dir}')

    all_z = []
    all_t = []

    for f in tqdm(files, desc='Extracting P2P image targets'):
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            z = data['z_mu']
            if z.ndim == 1:
                z = z.unsqueeze(0)

            uuid = os.path.basename(os.path.dirname(f))
            video_path = os.path.join(p2p_video_root, uuid, 'video.mp4')
            if not os.path.exists(video_path):
                print(f'  Video not found for {uuid}, skipping')
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'  Cannot open video for {uuid}, skipping')
                continue

            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()

            n_pairs = len(frames) - 1
            n_use = min(n_pairs, z.shape[0])

            for i in range(n_use):
                block = downscale_to_blocks(frames[i + 1])
                all_z.append(z[i])
                all_t.append(torch.from_numpy(block).float())

            del frames
        except Exception as e:
            print(f'Skipping {f}: {e}')

    if not all_z:
        raise RuntimeError('No usable P2P image data found')

    Z = torch.stack(all_z, dim=0)
    T = torch.stack(all_t, dim=0)
    torch.save(Z, cache_z)
    torch.save(T, cache_t)
    print(f'Cached {Z.shape[0]} P2P image samples to {cache_dir}')
    return Z, T


def main():
    parser = argparse.ArgumentParser(description="Train MLP to predict undersampled image from latent actions")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dump_dir', type=int, choices=[1, 2, 3], default=1, help='1=latent_actions_dump, 2=latent_actions_dump_2, 3=latent_actions_videoflextok')
    parser.add_argument('--model', type=str, default='olafworld', choices=['adaworld', 'olafworld'],
                        help='Model subdirectory to load from (only used with --dump_dir 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline', action='store_true', help='Use random inputs instead of latent actions')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.dump_dir == 2:
        # P2P path: extract targets from P2P video frames
        z, image = load_p2p_data_image(args.model)
        print(f"Loaded {z.shape[0]} P2P latent vectors, dim={z.shape[1]}")
        print(f"Loaded {image.shape[0]} P2P image blocks, output dim={image.shape[1]}")
    elif args.dump_dir == 3:
        # VideoFlexTok path: latents from latent_actions_videoflextok, retro game targets
        repo_root = Path(__file__).resolve().parents[1]
        latent_base = str(repo_root / 'latent_actions_videoflextok')
        color_base = '/scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data'

        print("Loading VideoFlexTok latent actions...")
        z = load_latent_actions(latent_base)
        print(f"Loaded {z.shape[0]} VideoFlexTok latent vectors, dim={z.shape[1]}")

        print("Loading undersampled image targets...")
        image = load_undersampled_images(color_base, z.shape[0])
        print(f"Loaded {image.shape[0]} image blocks, output dim={image.shape[1]}")

        # Ensure alignment
        if z.shape[0] != image.shape[0]:
            min_samples = min(z.shape[0], image.shape[0])
            z = z[:min_samples]
            image = image[:min_samples]
            print(f"Aligned to {min_samples} samples")
    else:
        # Retro game path: use shared cache
        repo_root = Path(__file__).resolve().parents[1]
        latent_base = str(repo_root / 'latent_actions_dump')
        color_base = '/scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data'

        print("Loading latent actions...")
        z = load_latent_actions(latent_base)
        print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

        print("Loading undersampled image targets...")
        image = load_undersampled_images(color_base, z.shape[0])
        print(f"Loaded {image.shape[0]} image blocks, output dim={image.shape[1]}")

        # Ensure alignment
        if z.shape[0] != image.shape[0]:
            min_samples = min(z.shape[0], image.shape[0])
            z = z[:min_samples]
            image = image[:min_samples]
            print(f"Aligned to {min_samples} samples")

    if args.baseline:
        print("BASELINE MODE: replacing latent actions with random tensors")
        z = torch.randn_like(z)

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
