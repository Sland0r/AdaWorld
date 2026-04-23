"""
Latent Action Encoder — Extract latent action representations from frame pairs.

This script provides a framework to study the Latent Action Model (LAM) from
AdaWorld. Given two consecutive frames it outputs the latent action
representation that captures the "action" (transition) between them.

Usage examples
--------------
# Extract from two image files
python extract_latent_actions.py --frame1 frame_a.png --frame2 frame_b.png

# Extract from a video (consecutive frame pairs)
python extract_latent_actions.py --video path/to/video.mp4

# Extract from a video starting at a specific frame
python extract_latent_actions.py --video path/to/video.mp4 --start-frame 50

# Extract from every pair in a directory of frames (sorted alphabetically)
python extract_latent_actions.py --frame-dir path/to/frames/

# Save outputs to disk
python extract_latent_actions.py --video path/to/video.mp4 --save-dir outputs/latent_actions/

# Use a specific checkpoint
python extract_latent_actions.py --frame1 a.png --frame2 b.png --lam-ckpt /path/to/lam.ckpt
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WORLDMODEL_DIR = SCRIPT_DIR.parent / "worldmodel"
if str(WORLDMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(WORLDMODEL_DIR))
OLAFWORLD_DIR = SCRIPT_DIR.parent / "olafworld"

from external.lam.model import LAM

# ---------------------------------------------------------------------------
# Default paths & constants
# ---------------------------------------------------------------------------
DEFAULT_LAM_CKPT = str(WORLDMODEL_DIR / "checkpoints" / "lam.ckpt")
HF_LAM_URL = "https://huggingface.co/Little-Podi/AdaWorld/resolve/main/lam.ckpt"
RESOLUTION = 256

DEFAULT_OLAF_CKPT_ALIGN = str(OLAFWORLD_DIR / "checkpoints" / "lam" / "lam_vjepa_align.ckpt")
DEFAULT_OLAF_CKPT_VAE = str(OLAFWORLD_DIR / "checkpoints" / "lam" / "lam.ckpt")
HF_OLAF_REPO = "YuxinJ/Olaf-World"

# LAM architecture hyper-parameters (must match the pretrained checkpoint)
LAM_CONFIG = dict(
    image_channels=3,
    lam_model_dim=1024,
    lam_latent_dim=32,
    lam_patch_size=16,
    lam_enc_blocks=16,
    lam_dec_blocks=16,
    lam_num_heads=16,
)


# ========================== Model Loading ==================================


def download_checkpoint(url: str, save_path: str) -> str:
    """Download a checkpoint from a URL if it does not exist locally."""
    if os.path.exists(save_path):
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading LAM checkpoint from {url} ...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Little-Podi/AdaWorld",
            filename="lam.ckpt",
            local_dir=os.path.dirname(save_path),
        )
        return downloaded
    except ImportError:
        # Fallback: use urllib
        import urllib.request
        urllib.request.urlretrieve(url, save_path)
        return save_path


def load_lam(ckpt_path: str | None = None, device: str = "cuda") -> LAM:
    """Instantiate and load the LAM model from a checkpoint.

    If *ckpt_path* is ``None`` or does not exist, the checkpoint is
    downloaded from Hugging Face.
    """
    if ckpt_path is None:
        ckpt_path = DEFAULT_LAM_CKPT

    if not os.path.exists(ckpt_path):
        ckpt_path = download_checkpoint(HF_LAM_URL, ckpt_path)

    model = LAM(ckpt_path=ckpt_path, **LAM_CONFIG)
    model = model.to(device).eval()
    print(f"LAM loaded from {ckpt_path}  (device={device})")
    return model


def load_olaf_encoder(ckpt_path: str | None = None, variant: str = "align", device: str = "cuda"):
    """Load the OlafWorld FrozenLAMEncoder.

    Requires the OlafWorld repo cloned at ``<project_root>/olafworld/``.
    Checkpoint is downloaded from HuggingFace if not present locally.
    """
    if str(OLAFWORLD_DIR) not in sys.path:
        sys.path.insert(0, str(OLAFWORLD_DIR))

    try:
        from lam.inference import FrozenLAMEncoder
    except ImportError as e:
        print(
            f"ERROR: Could not import OlafWorld. Make sure the repo is cloned at {OLAFWORLD_DIR}\n"
            f"  git clone https://github.com/showlab/Olaf-World {OLAFWORLD_DIR}\n"
            f"  Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if ckpt_path is None:
        ckpt_path = DEFAULT_OLAF_CKPT_ALIGN if variant == "align" else DEFAULT_OLAF_CKPT_VAE

    if not os.path.exists(ckpt_path):
        hf_filename = "lam/lam_vjepa_align.ckpt" if variant == "align" else "lam/lam.ckpt"
        print(f"Downloading OlafWorld checkpoint ({variant}) from HuggingFace ...")
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=HF_OLAF_REPO,
                filename=hf_filename,
                local_dir=str(OLAFWORLD_DIR / "checkpoints"),
            )
        except Exception as e:
            print(
                f"ERROR: Could not download OlafWorld checkpoint: {e}\n"
                f"  Manually download from https://huggingface.co/{HF_OLAF_REPO}",
                file=sys.stderr,
            )
            sys.exit(1)

    encoder = FrozenLAMEncoder(ckpt_path=ckpt_path, variant=variant, device=device)
    print(f"OlafWorld encoder loaded from {ckpt_path}  (variant={variant}, device={device})")
    return encoder


# ========================== Image I/O ======================================


def load_image(path: str, resolution: int = RESOLUTION) -> torch.Tensor:
    """Load a single image and return a tensor of shape (H, W, C) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    return torch.from_numpy(np.array(img)).float() / 255.0


def load_video_frames(
    path: str,
    resolution: int = RESOLUTION,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_skip: int = 1,
) -> list[torch.Tensor]:
    """Load frames from a video file **or** a directory of ordered images.

    Parameters
    ----------
    path : str
        Either the path to a video file (e.g. ``.mp4``) or a directory
        containing ordered frame images (``.png``, ``.jpg``, …).
    resolution : int
        Spatial size to resize every frame to ``(resolution, resolution)``.
    start_frame : int
        Index of the first frame to include.
    max_frames : int | None
        Maximum number of frames to return.  ``None`` means no limit.
    frame_skip : int
        Keep every *frame_skip*-th frame (1 = keep all).

    Returns
    -------
    list[torch.Tensor]
        Each element is a ``(H, W, C)`` float tensor in ``[0, 1]``.
    """
    if os.path.isdir(path):
        # ---- directory of ordered frame images ----------------------------
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths: list[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(path, ext)))
        paths.sort()
        paths = paths[start_frame:]
        # Apply frame_skip and max_frames
        paths = paths[::frame_skip]
        if max_frames is not None:
            paths = paths[:max_frames]
        if len(paths) < 2:
            raise ValueError(
                f"Need at least 2 images in {path}, found {len(paths)}"
            )
        print(f"Loaded {len(paths)} frames from {path}")

        import json
        actions_list = []
        action_names = []
        actions_path = os.path.join(os.path.dirname(path), "actions.json")
        info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "info.json")
        
        if os.path.exists(actions_path) and os.path.exists(info_path):
            with open(actions_path, 'r') as f:
                all_actions = json.load(f).get("actions", [])
            with open(info_path, 'r') as f:
                info_data = json.load(f)
                captions = info_data.get("info", {}).get("action_captions", [])
                action_names = [c[0] for c in captions if c]
            
            original_indices = []
            for p in paths:
                try:
                    original_indices.append(int(os.path.splitext(os.path.basename(p))[0]))
                except ValueError:
                    original_indices.append(None)
                    
            action_map = {a.get("src_id"): a.get("action") for a in all_actions}
            
            for i in range(len(original_indices) - 1):
                src = original_indices[i]
                if src is not None:
                    actions_list.append(action_map.get(src))
                else:
                    actions_list.append(None)

        frames = [load_image(p, resolution) for p in paths]
        return frames, (actions_list, action_names)


def load_frame_directory(
    dir_path: str, resolution: int = RESOLUTION
) -> list[torch.Tensor]:
    """Load all images from a directory (sorted), returning (H, W, C) tensors."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    paths.sort()
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {dir_path}, found {len(paths)}")
    return [load_image(p, resolution) for p in paths]


# ========================== Core Extraction =================================


@torch.no_grad()
def extract_latent_action(
    model: LAM,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Extract the latent action between two frames.

    Parameters
    ----------
    model : LAM
        The loaded Latent Action Model.
    frame1 : Tensor
        First frame, shape ``(H, W, C)`` with values in ``[0, 1]``.
    frame2 : Tensor
        Second frame, shape ``(H, W, C)`` with values in ``[0, 1]``.
    device : str
        Device to run inference on.

    Returns
    -------
    dict with keys:
        ``z_mu``  — latent action mean, shape ``(latent_dim,)``
        ``z_var`` — latent action log-variance, shape ``(latent_dim,)``
        ``z_rep`` — latent action sample (= mu at eval), shape ``(latent_dim,)``
    """
    # LAM expects input as (B, T, H, W, C)
    video = torch.stack([frame1, frame2], dim=0)  # (2, H, W, C)
    video = video.unsqueeze(0).to(device)          # (1, 2, H, W, C)

    outputs = model.lam.encode(video)

    return {
        "z_mu": outputs["z_mu"].squeeze(0).cpu(),       # (latent_dim,)
        "z_var": outputs["z_var"].squeeze(0).cpu(),      # (latent_dim,)
        "z_rep": outputs["z_rep"].squeeze().cpu(),       # (latent_dim,)
    }


@torch.no_grad()
def extract_latent_actions_batch(
    model,
    frames: list[torch.Tensor],
    mu_only: bool,
    device: str = "cuda",
    model_type: str = "adaworld",
) -> list:
    """Extract latent actions for all consecutive pairs in *frames*.

    Returns a list of dicts (one per consecutive pair), each containing
    ``z_mu``, ``z_var``, ``z_rep`` of shape ``(latent_dim,)``.
    For OlafWorld, ``z_var`` is zeros (the encoder is deterministic).
    """
    results = []
    batch_size = 128

    pairs = [torch.stack([frames[i], frames[i+1]], dim=0) for i in range(len(frames) - 1)]

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        video_batch = torch.stack(batch, dim=0).to(device)  # (B, 2, H, W, C)

        if model_type == "olafworld":
            z = model(video_batch)          # (B, 1, D) or (B, D)
            if z.dim() == 3:
                z = z.squeeze(1)            # (B, D)
            for j in range(z.size(0)):
                z_j = z[j].cpu()
                if mu_only:
                    results.append(z_j)
                else:
                    results.append({
                        "z_mu": z_j,
                        "z_var": torch.zeros_like(z_j),
                        "z_rep": z_j,
                    })
        else:
            outputs = model.lam.encode(video_batch)
            if mu_only:
                for j in range(video_batch.size(0)):
                    results.append(outputs['z_mu'][j].cpu())
            else:
                for j in range(video_batch.size(0)):
                    results.append({
                        "z_mu": outputs["z_mu"][j].cpu(),
                        "z_var": outputs["z_var"][j].cpu(),
                        "z_rep": outputs["z_rep"][j].cpu(),
                    })

    return results


# ========================== Pretty-print ====================================


def print_latent_action(result: dict[str, torch.Tensor], pair_idx: int = 0) -> None:
    """Print a single latent action result in a readable way."""
    z_mu = result["z_mu"]
    z_var = result["z_var"]
    z_rep = result["z_rep"]
    print(f"\n{'='*60}")
    print(f"  Pair {pair_idx}")
    print(f"{'='*60}")
    print(f"  z_mu  shape: {list(z_mu.shape)}  |  norm: {z_mu.norm():.4f}")
    print(f"  z_var shape: {list(z_var.shape)}  |  norm: {z_var.norm():.4f}")
    print(f"  z_rep shape: {list(z_rep.shape)}  |  norm: {z_rep.norm():.4f}")
    print(f"  z_mu  stats: min={z_mu.min():.4f}  max={z_mu.max():.4f}  mean={z_mu.mean():.4f}  std={z_mu.std():.4f}")
    print(f"  z_var stats: min={z_var.min():.4f}  max={z_var.max():.4f}  mean={z_var.mean():.4f}  std={z_var.std():.4f}")
    print(f"  z_mu  values: {z_mu.numpy()}")


def save_results(
    results: list[dict[str, torch.Tensor]], save_dir: str, mu_only=False, actions=None, action_names=None
) -> None:
    """Save latent action results to disk."""
    os.makedirs(save_dir, exist_ok=True)

    if mu_only:
        all_mu = torch.stack(results)
        all_var = None
        all_rep = None
    else:
        all_mu = torch.stack([r["z_mu"] for r in results])
        all_var = torch.stack([r["z_var"] for r in results])
        all_rep = torch.stack([r["z_rep"] for r in results])

    save_dict = {"z_mu": all_mu, "z_var": all_var, "z_rep": all_rep}
    if actions is not None:
        save_dict["actions"] = actions
    if action_names is not None:
        save_dict["action_names"] = action_names

    torch.save(
        save_dict,
        os.path.join(save_dir, "latent_actions.pt"),
    )

    # Also save a human-readable CSV of z_mu
    np.savetxt(
        os.path.join(save_dir, "z_mu.csv"),
        all_mu.numpy(),
        delimiter=",",
        header=",".join([f"dim_{i}" for i in range(all_mu.shape[1])]),
    )
    print(f"\nSaved {len(results)} latent action(s) to {save_dir}/")
    print(f"  latent_actions.pt  — full tensors (z_mu, z_var, z_rep)")
    print(f"  z_mu.csv           — z_mu values as CSV")


# ========================== CLI =============================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract latent action representations from frame pairs using the LAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input sources (mutually exclusive)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--frame1",
        type=str,
        help="Path to the first frame image (requires --frame2).",
    )
    group.add_argument(
        "--video", type=str, help="Path to a video file, frame directory, game root directory, or dataset root directory."
    )
    group.add_argument(
        "--frame-dir", type=str, help="Directory of frame images (sorted alphabetically)."
    )

    p.add_argument("--frame2", type=str, help="Path to the second frame image.")
    p.add_argument(
        "--model",
        type=str,
        default="adaworld",
        choices=["adaworld", "olafworld"],
        help="Action encoder model to use (default: adaworld).",
    )
    p.add_argument("--lam-ckpt", type=str, default=None, help="Path to AdaWorld LAM checkpoint (.ckpt).")
    p.add_argument("--olaf-ckpt", type=str, default=None, help="Path to OlafWorld LAM checkpoint (.ckpt).")
    p.add_argument(
        "--olaf-variant",
        type=str,
        default="align",
        choices=["vae", "align"],
        help="OlafWorld checkpoint variant: 'align' (τ-aligned, recommended) or 'vae' (baseline).",
    )
    p.add_argument("--resolution", type=int, default=RESOLUTION, help="Input resolution (default: 256).")
    p.add_argument("--start-frame", type=int, default=0, help="Start frame index for video input.")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames to read from video.")
    p.add_argument("--frame-skip", type=int, default=1, help="Frame skip interval for video.")
    p.add_argument("--device", type=str, default="cuda", help="Device (default: cuda).")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to save results (optional).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-pair printing.")
    p.add_argument("--mu_only", action='store_true', help='Only save mean of the VAE')

    return p.parse_args()


def _find_run_frame_dirs(root_dir: str) -> list[str]:
    """Find run frame folders under a root using seed/episode/frames layout.

    This supports either a single-game root or a dataset root containing many games.
    """
    pattern = os.path.join(root_dir, "**", "frames")
    run_dirs = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    run_dirs = [p for p in run_dirs if len(Path(os.path.relpath(p, root_dir)).parts) >= 3]
    run_dirs.sort()
    return run_dirs


def _extract_single_source(
    model,
    source_path: str,
    args: argparse.Namespace,
    model_type: str = "adaworld",
) -> tuple[list[torch.Tensor], list | None, list | None]:
    """Load frames from one source and extract latent actions."""
    actions, action_names = None, None
    video_out = load_video_frames(
        source_path,
        resolution=args.resolution,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    if isinstance(video_out, tuple) and len(video_out) == 2:
        frames, (actions, action_names) = video_out
    else:
        frames = video_out
    results = extract_latent_actions_batch(model, frames, args.mu_only, device=args.device, model_type=model_type)
    return results, actions, action_names


def _run_batch_game_root(args: argparse.Namespace, model, model_type: str = "adaworld") -> list:
    """Process all runs under a game root and save to mirrored output directories."""
    root_dir = os.path.abspath(args.video)
    run_frame_dirs = _find_run_frame_dirs(root_dir)

    if not run_frame_dirs:
        print(f"No run folders found under {root_dir} (expected seed/episode/frames).", file=sys.stderr)
        sys.exit(1)

    save_root = args.save_dir if args.save_dir else "./latent_actions_dump"
    total = len(run_frame_dirs)
    success = 0
    failed = 0

    print(f"Batch mode: found {total} run(s) under {root_dir}")
    print(f"Saving outputs under: {save_root}/...")

    for idx, run_frames in enumerate(run_frame_dirs, start=1):
        rel = os.path.relpath(run_frames, root_dir)
        rel_run = os.path.dirname(rel)
        rel_parts = Path(rel).parts
        if len(rel_parts) == 3:
            run_id = f"{os.path.basename(os.path.normpath(root_dir))}/{rel_run}"
            run_save_dir = os.path.join(save_root, model_type, os.path.basename(os.path.normpath(root_dir)), rel_run)
        else:
            run_id = rel_run
            run_save_dir = os.path.join(save_root, model_type, rel_run)
        out_pt = os.path.join(run_save_dir, "latent_actions.pt")

        if os.path.exists(out_pt):
            print(f"[{idx}/{total}] Skipping existing: {run_id}")
            success += 1
            continue

        print(f"[{idx}/{total}] Processing: {run_id}")
        try:
            results, actions, action_names = _extract_single_source(model, run_frames, args, model_type=model_type)
            save_results(results, run_save_dir, args.mu_only, actions=actions, action_names=action_names)
            success += 1
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR on {run_id}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Batch complete. total={total} success={success} failed={failed}")
    if failed > 0:
        sys.exit(1)

    # Returning empty list prevents downstream single-run assumptions.
    return []


def main() -> None:
    args = parse_args()

    # --- Validate args ---
    if args.frame1 and not args.frame2:
        print("ERROR: --frame2 is required when using --frame1", file=sys.stderr)
        sys.exit(1)

    # --- Load model ---
    model_type = args.model
    if model_type == "olafworld":
        model = load_olaf_encoder(args.olaf_ckpt, args.olaf_variant, device=args.device)
    else:
        model = load_lam(args.lam_ckpt, device=args.device)

    # --- Batch mode for game root directories ---
    if args.video and os.path.isdir(args.video):
        run_frame_dirs = _find_run_frame_dirs(args.video)
        if run_frame_dirs:
            return _run_batch_game_root(args, model, model_type=model_type)

    # --- Load frames ---
    actions, action_names = None, None
    if args.frame1:
        frames = [
            load_image(args.frame1, args.resolution),
            load_image(args.frame2, args.resolution),
        ]
        print(f"Loaded 2 images: {args.frame1}, {args.frame2}")
    elif args.video:
        video_out = load_video_frames(
            args.video,
            resolution=args.resolution,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        if isinstance(video_out, tuple) and len(video_out) == 2:
            frames, (actions, action_names) = video_out
        else:
            frames = video_out
            actions, action_names = None, None
        print(f"Loaded {len(frames)} frames from {args.video}. Shape {frames[0].shape}")
        if actions:
            print(f"\n{'='*60}")
            print(f"  Video Actions ({len(actions)}), {len(action_names)} action names: {action_names}")
            
    else:
        frames = load_frame_directory(args.frame_dir, args.resolution)
        print(f"Loaded {len(frames)} frames from {args.frame_dir}. Shape {frames[0].shape}")

    if len(frames) < 2:
        print("ERROR: Need at least 2 frames.", file=sys.stderr)
        sys.exit(1)

    # --- Extract ---
    results = extract_latent_actions_batch(model, frames, args.mu_only, device=args.device, model_type=model_type)

    print(f"\n{'='*60}")
    print(f"  Summary: extracted {len(results)} latent action(s)")
    latent_dim = results[0].shape[-1] if args.mu_only else results[0]["z_mu"].shape[-1]
    print(f"  Latent dim: {latent_dim}")
    if len(results) > 1:
        all_mu = torch.stack([r for r in results]) if args.mu_only else torch.stack([r["z_mu"] for r in results])
        print(f"  Mean z_mu norm across pairs: {all_mu.norm(dim=-1).mean():.4f}")
        # Pairwise cosine similarity of consecutive actions
        cos_sims = F.cosine_similarity(all_mu[:-1], all_mu[1:], dim=-1)
        print(f"  Cosine sim (consecutive): mean={cos_sims.mean():.4f}  std={cos_sims.std():.4f}")
    print(f"{'='*60}")

    # --- Save ---
    if args.save_dir:
        save_results(results, args.save_dir, args.mu_only, actions=actions, action_names=action_names)

    return results


if __name__ == "__main__":
    main()
