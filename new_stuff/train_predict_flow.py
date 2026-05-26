#!/usr/bin/env python3
"""
Train a CNN decoder to predict optical flow from latent actions.
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


FLOW_SAMPLE_PROB = 1.0 / 20.0

class CNNDecoder(nn.Module):
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

def load_flow_targets(base_dir, num_samples, cache_dir="/scratch-shared/FoMo-Atomic-Actions/_cache"):
    cache_path = os.path.join(cache_dir, "flow_targets.pt")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        targets = torch.load(cache_path, map_location="cpu")
        return targets[:num_samples]

    print("  Cache not found, loading individual .npy files (slow)...")
    files = sorted(glob.glob(os.path.join(base_dir, "**", "*.npy"), recursive=True))
    if not files:
        raise RuntimeError(f"No .npy files found in {base_dir}")
    
    targets = []
    for f in tqdm(files[:num_samples], desc="Loading .npy files"):
        try:
            arr = np.load(f)
            tensor = torch.from_numpy(arr).float().permute(2, 0, 1)
            targets.append(tensor)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not targets:
        raise RuntimeError(f"No usable flow targets found in {base_dir}")
    T = torch.stack(targets, dim=0)[:num_samples]
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(T, cache_path)
    return T


def dense_flow(prev_bgr, next_bgr):
    """Compute Farneback dense optical flow between two BGR frames."""
    prev_bgr = cv2.resize(prev_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    next_bgr = cv2.resize(next_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )
    return flow.astype(np.float32)


def load_p2p_data_flow(model_name, cache_dir='/scratch-shared/FoMo-Atomic-Actions/_cache/p2p',
                       p2p_video_root='/scratch-shared/FoMo-Atomic-Actions/open-p2p-subset',
                       force_extract=False):
    """Load latent actions + optical flow targets from P2P videos.
    Caches results in scratch-shared. Modeled after train_predict_skipped_flow.py."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_z = os.path.join(cache_dir, f'{model_name}_z_mu_flow.pt')
    cache_t = os.path.join(cache_dir, f'{model_name}_flow_targets.pt')

    if not force_extract and os.path.exists(cache_z) and os.path.exists(cache_t):
        print(f"Loading cached P2P flow data for {model_name}...")
        return torch.load(cache_z, map_location='cpu'), torch.load(cache_t, map_location='cpu')

    print(f"Extracting flow targets from P2P videos for {model_name}...")
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = str(repo_root / 'latent_actions_dump_2' / model_name)
    files = sorted(glob.glob(os.path.join(base_dir, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found in {base_dir}')

    all_z = []
    all_t = []

    for f in tqdm(files, desc='Extracting P2P flow targets'):
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

            pair_idx = 0
            prev_frame = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if prev_frame is not None:
                    if pair_idx < z.shape[0] and random.random() < FLOW_SAMPLE_PROB:
                        flow = dense_flow(prev_frame, frame)
                        t = torch.from_numpy(flow).permute(2, 0, 1).float()
                        t = F.interpolate(t.unsqueeze(0), size=(60, 80), mode='bilinear', align_corners=False).squeeze(0)
                        all_z.append(z[pair_idx])
                        all_t.append(t)
                    pair_idx += 1
                prev_frame = frame
            cap.release()
        except Exception as e:
            print(f'Skipping {f}: {e}')

    if not all_z:
        raise RuntimeError('No usable P2P flow data found')

    Z = torch.stack(all_z, dim=0)
    T = torch.stack(all_t, dim=0)
    torch.save(Z, cache_z)
    torch.save(T, cache_t)
    print(f'Cached {Z.shape[0]} P2P flow samples to {cache_dir}')
    return Z, T


def load_videoflextok_data_flow(
    latent_base,
    frames_root='/scratch-shared/scur0531/skipped_frames_v0.0.0',
    cache_dir='/scratch-shared/FoMo-Atomic-Actions/_cache',
    force_extract=False,
):
    """Load VideoFlexTok latent actions + compute optical flow targets on-the-fly.

    Each latent_actions.pt contains an 'actions' list with 'src_idx' / 'tgt_idx'
    pointing to frames in frames_root/<game>/<seed>/<episode>/frames/<idx>.jpg.
    Flow is computed with Farneback and sub-sampled at FLOW_SAMPLE_PROB.
    Results are cached so subsequent runs are instant.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_z = os.path.join(cache_dir, 'videoflextok_z_mu_flow.pt')
    cache_t = os.path.join(cache_dir, 'videoflextok_flow_targets.pt')

    if not force_extract and os.path.exists(cache_z) and os.path.exists(cache_t):
        print(f"Loading cached VideoFlexTok flow data from {cache_dir}...")
        return torch.load(cache_z, map_location='cpu'), torch.load(cache_t, map_location='cpu')

    print(f"Extracting flow targets from frames in {frames_root}...")
    files = sorted(glob.glob(os.path.join(latent_base, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found in {latent_base}')

    all_z = []
    all_t = []

    for f in tqdm(files, desc='Extracting VideoFlexTok flow targets'):
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            z = data['z_mu']
            if z.ndim == 1:
                z = z.unsqueeze(0)
            actions = data.get('actions', [])

            # Derive frames directory: latent_base/<game>/<seed>/<episode>/latent_actions.pt
            # -> frames_root/<game>/<seed>/<episode>/frames/
            rel_path = os.path.relpath(os.path.dirname(f), latent_base)
            frames_dir = os.path.join(frames_root, rel_path, 'frames')

            if not os.path.isdir(frames_dir):
                continue

            for i, meta in enumerate(actions):
                if 'src_idx' not in meta or 'tgt_idx' not in meta:
                    continue
                if i >= z.shape[0]:
                    break
                if random.random() >= FLOW_SAMPLE_PROB:
                    continue

                src_path = os.path.join(frames_dir, f"{meta['src_idx']:06d}.jpg")
                tgt_path = os.path.join(frames_dir, f"{meta['tgt_idx']:06d}.jpg")
                if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                    continue

                src_img = cv2.imread(src_path)
                tgt_img = cv2.imread(tgt_path)
                if src_img is None or tgt_img is None:
                    continue

                flow = dense_flow(src_img, tgt_img)
                t = torch.from_numpy(flow).permute(2, 0, 1).float()
                t = F.interpolate(t.unsqueeze(0), size=(60, 80), mode='bilinear', align_corners=False).squeeze(0)

                all_z.append(z[i])
                all_t.append(t)
        except Exception as e:
            print(f'Skipping {f}: {e}')

    if not all_z:
        raise RuntimeError('No usable VideoFlexTok flow data found')

    Z = torch.stack(all_z, dim=0)
    T = torch.stack(all_t, dim=0)
    torch.save(Z, cache_z)
    torch.save(T, cache_t)
    print(f'Cached {Z.shape[0]} VideoFlexTok flow samples to {cache_dir}')
    return Z, T


def main():
    parser = argparse.ArgumentParser(description="Train CNN decoder to predict optical flow from latent actions")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count for CNN decoder")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dump_dir", type=int, choices=[1, 2, 3], default=1, help="1=latent_actions_dump, 2=latent_actions_dump_2, 3=latent_actions_videoflextok")
    parser.add_argument('--model', type=str, default='olafworld', choices=['adaworld', 'olafworld'],
                        help='Model subdirectory to load from (only used with --dump_dir 2)')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Use random inputs instead of latent actions")
    parser.add_argument("--cache_dir", type=str, default="/scratch-shared/FoMo-Atomic-Actions/_cache")
    parser.add_argument(
        "--target_root",
        type=str,
        default="/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data",
        help="Root for legacy per-sample .npy targets",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="/scratch-shared/scur0531/skipped_frames_v0.0.0",
        help="Root for retro game frame data (used with --dump_dir 3)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dump_dir == 2:
        # P2P path: extract targets from P2P video frames
        z, flow_targets = load_p2p_data_flow(args.model)
        print(f"Loaded {z.shape[0]} P2P latent vectors, dim={z.shape[1]}")
        print(f"Loaded {flow_targets.shape[0]} P2P flow blocks")
        num_samples = z.shape[0]
    elif args.dump_dir == 3:
        # VideoFlexTok path: compute flow on-the-fly from skipped frames
        repo_root = Path(__file__).resolve().parents[1]
        latent_base = str(repo_root / "latent_actions_videoflextok")

        print("Loading VideoFlexTok latent actions + extracting flow targets...")
        z, flow_targets = load_videoflextok_data_flow(
            latent_base,
            frames_root=args.frames_root,
            cache_dir=args.cache_dir,
        )
        print(f"Loaded {z.shape[0]} VideoFlexTok latent vectors, dim={z.shape[1]}")
        print(f"Loaded {flow_targets.shape[0]} flow targets")
        num_samples = z.shape[0]
    else:
        # Retro game path: use shared cache
        repo_root = Path(__file__).resolve().parents[1]
        latent_base = str(repo_root / "latent_actions_dump")

        print("Loading latent actions...")
        z = load_latent_actions(latent_base)
        print(f"Loaded {z.shape[0]} latent vectors, dim={z.shape[1]}")

        print("Loading flow targets...")
        flow_targets = load_flow_targets(args.target_root, z.shape[0], cache_dir=args.cache_dir)
        print(f"Loaded {flow_targets.shape[0]} flow blocks")

        num_samples = min(z.shape[0], flow_targets.shape[0])
        z = z[:num_samples]
        flow_targets = flow_targets[:num_samples]
        print(f"Aligned to {num_samples} samples")

    if args.baseline:
        print("BASELINE MODE: replacing latent actions with random tensors")
        z = torch.randn_like(z)

    out_channels = flow_targets.shape[1]
    target_h = flow_targets.shape[2]
    target_w = flow_targets.shape[3]
    print(f"Target shape: ({out_channels}, {target_h}, {target_w})")

    test_ratio = 0.2
    n_test = max(1, int(num_samples * test_ratio))
    indices = torch.randperm(num_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_z, train_flow = z[train_idx], flow_targets[train_idx]
    test_z, test_flow = z[test_idx], flow_targets[test_idx]

    train_dataset = TensorDataset(train_z, train_flow)
    test_dataset = TensorDataset(test_z, test_flow)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = CNNDecoder(z.shape[1], out_channels, target_h, target_w, base_channels=args.base_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN decoder params: {num_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(args.epochs), desc="Training")
    for _ in pbar:
        model.train()
        total_loss = 0.0
        for batch_z, batch_flow in train_loader:
            batch_z = batch_z.to(device)
            batch_flow = batch_flow.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            loss = criterion(pred, batch_flow)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix(loss=f"{(total_loss / len(train_loader)):.6f}")

    print("Evaluating...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_z, batch_flow in test_loader:
            batch_z = batch_z.to(device)
            batch_flow = batch_flow.to(device)
            pred = model(batch_z)
            loss = criterion(pred, batch_flow)
            test_loss += loss.item()
    print(f"Test MSE: {(test_loss / len(test_loader)):.6f}")

if __name__ == '__main__':
    main()
