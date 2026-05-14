#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class GameSample:
    count_seen: int
    frame_a: str
    frame_b: str


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


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang * 180.0 / np.pi) / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def find_frames_dirs(src_root: str) -> List[str]:
    out = []
    for root, dirs, files in os.walk(src_root):
        if os.path.basename(root) == 'frames' and any(f.lower().endswith('.jpg') for f in files):
            out.append(root)
    out.sort()
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def update_reservoir(
    samples: Dict[str, GameSample], game: str, frame_a: str, frame_b: str
) -> None:
    if game not in samples:
        samples[game] = GameSample(count_seen=1, frame_a=frame_a, frame_b=frame_b)
        return

    sample = samples[game]
    sample.count_seen += 1
    if random.randint(1, sample.count_seen) == 1:
        sample.frame_a = frame_a
        sample.frame_b = frame_b


def game_name_from_rel_path(rel_path: str) -> str:
    return rel_path.split(os.sep)[0]


def process_dataset(src_root: str, dst_root: str, frame_step: int, pair_stride: int, seed: int) -> None:
    random.seed(seed)
    ensure_dir(dst_root)

    frames_dirs = find_frames_dirs(src_root)
    print(f"Found {len(frames_dirs)} frame folders")

    game_samples: Dict[str, GameSample] = {}
    game_pair_counts: Dict[str, int] = {}
    total_pairs = 0

    for idx_dir, frames_dir in enumerate(frames_dirs, start=1):
        rel_frames_dir = os.path.relpath(frames_dir, src_root)
        game = game_name_from_rel_path(rel_frames_dir)

        parent_rel = os.path.relpath(os.path.dirname(frames_dir), src_root)
        out_flow_dir = os.path.join(dst_root, parent_rel, f'optic_flow_step{frame_step}')
        ensure_dir(out_flow_dir)

        frame_files = sorted_frame_files(frames_dir)
        if len(frame_files) <= frame_step:
            continue

        local_pairs = 0
        for i in range(0, len(frame_files) - frame_step, pair_stride):
            frame_a = frame_files[i]
            frame_b = frame_files[i + frame_step]

            img_a = cv2.imread(frame_a, cv2.IMREAD_COLOR)
            img_b = cv2.imread(frame_b, cv2.IMREAD_COLOR)
            if img_a is None or img_b is None:
                continue

            flow = dense_flow(img_a, img_b)

            stem_a = os.path.splitext(os.path.basename(frame_a))[0]
            stem_b = os.path.splitext(os.path.basename(frame_b))[0]
            out_name = f"{stem_a}_to_{stem_b}.npy"
            out_path = os.path.join(out_flow_dir, out_name)
            np.save(out_path, flow)

            update_reservoir(game_samples, game, frame_a, frame_b)
            game_pair_counts[game] = game_pair_counts.get(game, 0) + 1
            total_pairs += 1
            local_pairs += 1

        if idx_dir % 25 == 0:
            print(
                f"[{idx_dir}/{len(frames_dirs)}] {rel_frames_dir}: wrote {local_pairs} flow files (total {total_pairs})"
            )

    print(f"Finished flow extraction. Total flow files: {total_pairs}")

    summary = {
        'src_root': src_root,
        'dst_root': dst_root,
        'frame_step': frame_step,
        'pair_stride': pair_stride,
        'seed': seed,
        'total_flow_files': total_pairs,
        'games_with_samples': len(game_samples),
        'game_pair_counts': game_pair_counts,
    }

    ensure_dir(dst_root)
    with open(os.path.join(dst_root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    for game, sample in game_samples.items():
        preview_dir = os.path.join(dst_root, game, 'random_preview')
        ensure_dir(preview_dir)

        img_a = cv2.imread(sample.frame_a, cv2.IMREAD_COLOR)
        img_b = cv2.imread(sample.frame_b, cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            continue

        flow = dense_flow(img_a, img_b)
        flow_vis = flow_to_color(flow)

        a_name = os.path.basename(sample.frame_a)
        b_name = os.path.basename(sample.frame_b)

        out_a = os.path.join(preview_dir, f"frame_a_{a_name}")
        out_b = os.path.join(preview_dir, f"frame_b_{b_name}")
        out_flow = os.path.join(preview_dir, 'flow_a_to_b.jpg')
        out_meta = os.path.join(preview_dir, 'sample_meta.json')

        cv2.imwrite(out_a, img_a)
        cv2.imwrite(out_b, img_b)
        cv2.imwrite(out_flow, flow_vis)

        meta = {
            'game': game,
            'source_frame_a': sample.frame_a,
            'source_frame_b': sample.frame_b,
            'frame_step': frame_step,
        }
        with open(out_meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    print('Saved random preview pairs for each game.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute optical flow dump from random actions dataset.'
    )
    parser.add_argument(
        '--src-root',
        default='/home/scur0531/random_actions_data/dataset/retro_act_v0.0.0_random',
        help='Source dataset root',
    )
    parser.add_argument(
        '--dst-root',
        default='/scratch-shared/FoMo-Atomic-Actions/optic_flow_dump/random_actions_data',
        help='Output root for optical flow dump',
    )
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Use frame i and frame i+frame_step',
    )
    parser.add_argument(
        '--pair-stride',
        type=int,
        default=1,
        help='Stride used while moving through frame indices',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample previews')

    args = parser.parse_args()
    process_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        frame_step=args.frame_step,
        pair_stride=args.pair_stride,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
