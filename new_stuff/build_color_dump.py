#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import cv2
import numpy as np


def sorted_frame_files(frames_dir: str) -> List[str]:
    files = [
        os.path.join(frames_dir, name)
        for name in os.listdir(frames_dir)
        if name.lower().endswith('.jpg')
    ]

    def sort_key(path: str):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    files.sort(key=sort_key)
    return files


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


def downscale_to_blocks(image_bgr: np.ndarray, size: int = 32) -> np.ndarray:
    blocks = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return blocks


def find_frames_dirs(src_root: str) -> List[str]:
    out = []
    for root, dirs, files in os.walk(src_root):
        if os.path.basename(root) == 'frames' and any(f.lower().endswith('.jpg') for f in files):
            out.append(root)
    out.sort()
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_dataset(src_root: str, dst_root: str, block_size: int) -> None:
    ensure_dir(dst_root)

    frames_dirs = find_frames_dirs(src_root)
    print(f"Found {len(frames_dirs)} frame folders")

    total_frames = 0
    per_game_counts: Dict[str, int] = {}

    for idx_dir, frames_dir in enumerate(frames_dirs, start=1):
        rel_frames_dir = os.path.relpath(frames_dir, src_root)
        game = rel_frames_dir.split(os.sep)[0]

        parent_rel = os.path.relpath(os.path.dirname(frames_dir), src_root)
        out_color_dir = os.path.join(dst_root, parent_rel, f'color_step1')
        ensure_dir(out_color_dir)

        frame_files = sorted_frame_files(frames_dir)
        local_frames = 0

        for frame_path in frame_files:
            image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            features = {
                'histogram': color_histogram(image),
                'blocks': downscale_to_blocks(image, size=block_size),
            }

            stem = os.path.splitext(os.path.basename(frame_path))[0]
            out_path = os.path.join(out_color_dir, f'{stem}.npz')
            np.savez_compressed(out_path, **features)

            total_frames += 1
            local_frames += 1
            per_game_counts[game] = per_game_counts.get(game, 0) + 1

        if idx_dir % 25 == 0:
            print(
                f"[{idx_dir}/{len(frames_dirs)}] {rel_frames_dir}: wrote {local_frames} color feature files (total {total_frames})"
            )

    summary = {
        'src_root': src_root,
        'dst_root': dst_root,
        'block_size': block_size,
        'total_frames': total_frames,
        'per_game_counts': per_game_counts,
    }
    with open(os.path.join(dst_root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Finished color extraction. Total frames: {total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute per-frame color features from the random actions dataset.'
    )
    parser.add_argument(
        '--src-root',
        default='/home/scur0531/random_actions_data/dataset/retro_act_v0.0.0_random',
        help='Source dataset root',
    )
    parser.add_argument(
        '--dst-root',
        default='/scratch-shared/FoMo-Atomic-Actions/color_dump/random_actions_data',
        help='Output root for color dump',
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=32,
        help='Downscaled block image size',
    )

    args = parser.parse_args()
    process_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        block_size=args.block_size,
    )


if __name__ == '__main__':
    main()