#!/usr/bin/env python3
import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import random

def build_mlp(in_dim, out_dim, n_hidden, hidden_dim=256):
    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

def color_histogram(image_bgr: np.ndarray, bins_per_channel: int = 256) -> np.ndarray:
    channels = cv2.split(image_bgr)
    histograms = []
    for channel in channels:
        hist = cv2.calcHist([channel], [0], None, [bins_per_channel], [0, 256])
        histograms.append(hist.reshape(-1))
    histogram = np.concatenate(histograms, axis=0).astype(np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram

def extract_target(t_img):
    return torch.from_numpy(color_histogram(t_img)).float()

def load_data(model_name, force_extract=False):
    cache_z = f"latent_actions_skipped/{model_name}_z_mu.pt"
    cache_t = f"latent_actions_skipped/{model_name}_color_targets.pt"
    if not force_extract and os.path.exists(cache_z) and os.path.exists(cache_t):
        print(f"Loading cached targets and latents for {model_name}...")
        return torch.load(cache_z, map_location='cpu'), torch.load(cache_t, map_location='cpu')

    print("Extracting targets natively from skipped frames...")
    base_dir = f"latent_actions_skipped/{model_name}"
    files = sorted(glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True))

    all_z = []
    all_t = []
    
    for f in tqdm(files, desc="Extracting targets"):
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            z = data['z_mu']
            actions = data['actions']

            # get frames dir
            parts = f.split('/')
            rel_run = '/'.join(parts[2:-1])
            frames_dir = f"/scratch-shared/scur0531/skipped_frames_v0.0.0/{rel_run}/frames"
            
            for i, meta in enumerate(actions):
                if 'src_idx' not in meta or 'tgt_idx' not in meta:
                    continue
                tgt_idx = meta['tgt_idx']
                tgt_path = os.path.join(frames_dir, f"{tgt_idx:06d}.jpg")
                if not os.path.exists(tgt_path):
                    continue
                t_img = cv2.imread(tgt_path)
                if t_img is None:
                    continue
                
                target_val = extract_target(t_img)
                all_z.append(z[i] if z.ndim > 1 else z)
                all_t.append(target_val)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    Z = torch.stack(all_z, dim=0)
    T = torch.stack(all_t, dim=0)
    torch.save(Z, cache_z)
    torch.save(T, cache_t)
    return Z, T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['adaworld', 'olafworld'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--force_extract', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z, targets = load_data(args.model, args.force_extract)
    if args.baseline:
        print("Using baseline (random z)...")
        z = torch.randn_like(z)

    n_test = max(1, int(z.shape[0] * 0.2))
    indices = torch.randperm(z.shape[0])
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_loader = DataLoader(TensorDataset(z[train_idx], targets[train_idx]), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(z[test_idx], targets[test_idx]), batch_size=args.batch_size, shuffle=False)

    model = build_mlp(z.shape[1], targets.shape[1], args.hidden_layers, args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch_z, batch_t in train_loader:
            optimizer.zero_grad()
            pred = model(batch_z.to(device))
            loss = criterion(pred, batch_t.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_z, batch_t in test_loader:
            pred = model(batch_z.to(device))
            test_loss += criterion(pred, batch_t.to(device)).item()
    print(f"Test MSE: {test_loss / len(test_loader):.6f}")

if __name__ == '__main__':
    main()
