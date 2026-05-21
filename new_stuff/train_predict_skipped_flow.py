#!/usr/bin/env python3
import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import random

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

def dense_flow(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return flow.astype(np.float32)

def extract_target(s_img, t_img):
    t = dense_flow(s_img, t_img)
    return torch.from_numpy(t).permute(2, 0, 1).float()

def load_data(model_name, force_extract=False):
    cache_z = f"latent_actions_skipped/{model_name}_z_mu_flow.pt"
    cache_t = f"latent_actions_skipped/{model_name}_flow_targets.pt"
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

            parts = f.split('/')
            rel_run = '/'.join(parts[2:-1])
            frames_dir = f"/scratch-shared/scur0531/skipped_frames_v0.0.0/{rel_run}/frames"
            
            for i, meta in enumerate(actions):
                if 'src_idx' not in meta or 'tgt_idx' not in meta:
                    continue
                src_idx = meta['src_idx']
                tgt_idx = meta['tgt_idx']
                src_path = os.path.join(frames_dir, f"{src_idx:06d}.jpg")
                tgt_path = os.path.join(frames_dir, f"{tgt_idx:06d}.jpg")
                if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                    continue
                s_img = cv2.imread(src_path)
                t_img = cv2.imread(tgt_path)
                if s_img is None or t_img is None:
                    continue
                
                target_val = extract_target(s_img, t_img)
                # optionally downscale to save memory like differences
                target_val = F.interpolate(target_val.unsqueeze(0), scale_factor=0.25, mode='bilinear').squeeze(0)
                
                all_z.append(z[i] if z.ndim > 1 else z)
                all_t.append(target_val)
        except Exception as e:
            pass

    print(f"Loaded {len(all_z)} sequences.")
    if len(all_z) == 0:
        return torch.empty(0), torch.empty(0)

    Z = torch.stack(all_z, dim=0)
    T = torch.stack(all_t, dim=0)
    torch.save(Z, cache_z)
    torch.save(T, cache_t)
    return Z, T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['adaworld', 'olafworld'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_channels', type=int, default=32)
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
    if z.shape[0] == 0:
        print("No valid data.")
        return
    if args.baseline:
        print("Using baseline (random z)...")
        z = torch.randn_like(z)

    n_test = max(1, int(z.shape[0] * 0.2))
    indices = torch.randperm(z.shape[0])
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_loader = DataLoader(TensorDataset(z[train_idx], targets[train_idx]), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(z[test_idx], targets[test_idx]), batch_size=args.batch_size, shuffle=False)

    out_channels = targets.shape[1]
    target_h = targets.shape[2]
    target_w = targets.shape[3]
    model = CNNDecoder(z.shape[1], out_channels, target_h, target_w, base_channels=args.base_channels).to(device)
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
