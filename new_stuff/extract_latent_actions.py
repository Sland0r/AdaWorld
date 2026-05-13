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
import json as _json
import gc
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

try:
    from torchvision.io import read_video as torchvision_read_video
except Exception:
    torchvision_read_video = None

try:
    from google.protobuf.internal.decoder import _DecodeVarint
    from google.protobuf.internal.wire_format import (
        WIRETYPE_FIXED32,
        WIRETYPE_FIXED64,
        WIRETYPE_LENGTH_DELIMITED,
        WIRETYPE_VARINT,
    )
    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False

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
OLAFWORLD_RESOLUTION = (272, 480)  # (H, W) from olafworld/configs/lam/*.yaml

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

# ---------------------------------------------------------------------------
# P2P dataset label schema
# ---------------------------------------------------------------------------
# Keyboard keys with >9k frames in the dataset (sorted by frequency)
P2P_KEYBOARD_KEYS = ["w", "d", "a", "LeftArrow", "LeftShift", "RightArrow", "s", "UpArrow", "f", "Space"]
# Mouse button encoding: 0=left click, 1=right click, 2=no click
P2P_MOUSE_BUTTON_NONE = 2


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


def load_image(path: str, resolution: int | tuple[int, int] = RESOLUTION) -> torch.Tensor:
    """Load a single image and return a tensor of shape (H, W, C) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if isinstance(resolution, tuple):
        h, w = resolution
        img = img.resize((w, h), Image.BICUBIC)
    else:
        img = img.resize((resolution, resolution), Image.BICUBIC)
    return torch.from_numpy(np.array(img)).float() / 255.0


def _resize_frame_array(frame: np.ndarray, resolution: int | tuple[int, int]) -> torch.Tensor:
    image = Image.fromarray(frame).convert("RGB")
    if isinstance(resolution, tuple):
        h, w = resolution
        image = image.resize((w, h), Image.BICUBIC)
    else:
        image = image.resize((resolution, resolution), Image.BICUBIC)
    return torch.from_numpy(np.array(image)).float() / 255.0


def _read_video_file(path: str) -> list[torch.Tensor]:
    """Decode a video file into RGB frame tensors using ffmpeg."""
    import subprocess
    import tempfile

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            frames: list[torch.Tensor] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame))
            cap.release()
            if frames:
                return frames
    except Exception:
        pass
    
    try:
        # Use ffmpeg to extract frames as a sequence of PNG images piped to stdout
        cmd = [
            "ffmpeg",
            "-i", path,
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        frames = []
        # Read raw RGB frames from ffmpeg stdout
        # Each frame is W*H*3 bytes (24 bits per pixel)
        # First, we need to get video dimensions
        get_info_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            path
        ]
        try:
            info_output = subprocess.check_output(get_info_cmd, text=True).strip()
            w, h = map(int, info_output.split(","))
            frame_size = w * h * 3
            
            while True:
                data = proc.stdout.read(frame_size)
                if len(data) < frame_size:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                frames.append(torch.from_numpy(frame))
            
            proc.wait()
            if frames:
                return frames
        except Exception:
            proc.kill()
            raise
    except Exception as exc:
        raise RuntimeError(f"Unable to decode video file: {path}") from exc


def _extract_annotation_strings(annotation_path: str) -> list[str]:
    """Best-effort extraction of human-readable labels from a protobuf blob."""
    try:
        raw = Path(annotation_path).read_bytes().decode("latin1", errors="ignore")
    except OSError:
        return []

    seen = set()
    labels: list[str] = []
    for candidate in re.findall(r"[ -~]{6,}", raw):
        text = re.sub(r"^[^A-Za-z]+", "", candidate).strip()
        if not text or text in seen:
            continue
        if text.startswith(("$", "LLM-", "gemini-", "scur", "scur0531")):
            continue
        if any(token in text.lower() for token in ("checkpoint", "thinking", "preview", "flash", "gemma")):
            continue
        if text.count(" ") < 2 and not text.startswith(("Avoid", "Get", "Engage", "Approach", "Eat", "Activate")):
            continue
        seen.add(text)
        labels.append(text)
    return labels


def _load_video_annotations(video_path: str) -> tuple[list[tuple[str, ...]] | None, list[str] | None]:
    annotation_path = os.path.join(os.path.dirname(video_path), "annotation.proto")
    if not os.path.exists(annotation_path):
        return None, None

    labels = _extract_annotation_strings(annotation_path)
    if not labels:
        return None, None

    action_names = labels
    actions = [tuple(labels)]
    return actions, action_names


def _load_source_frames(
    source_path: str,
    resolution: int | tuple[int, int] = RESOLUTION,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_skip: int = 1,
) -> tuple[list[torch.Tensor], tuple[list[tuple[str, ...]] | None, list[str] | None] | None]:
    if os.path.isfile(source_path):
        if source_path.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            frames = _read_video_file(source_path)
            frames = frames[start_frame:]
            frames = frames[::frame_skip]
            if max_frames is not None:
                frames = frames[:max_frames]
            if len(frames) < 2:
                raise ValueError(f"Need at least 2 frames in {source_path}, found {len(frames)}")
            resized = [_resize_frame_array(frame.numpy(), resolution) for frame in frames]
            return resized, _load_video_annotations(source_path)
        raise FileNotFoundError(f"The path '{source_path}' is not a supported video file.")

    if os.path.isdir(source_path):
        video_candidates = []
        for ext in ("*.mp4", "*.webm", "*.mov", "*.mkv", "*.avi"):
            video_candidates.extend(glob.glob(os.path.join(source_path, ext)))
        preferred = [p for p in video_candidates if os.path.basename(p) == "video.mp4"]
        video_candidates = preferred or sorted(video_candidates)
        if video_candidates:
            return _load_source_frames(
                video_candidates[0],
                resolution=resolution,
                start_frame=start_frame,
                max_frames=max_frames,
                frame_skip=frame_skip,
            )

    raise FileNotFoundError(f"The path '{source_path}' does not exist or is not a supported video source.")


def load_video_frames(
    path: str,
    resolution: int | tuple[int, int] = RESOLUTION,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_skip: int = 1,
) -> list[torch.Tensor]:
    """Load frames from a video file, a directory of ordered images, or a video folder.

    Parameters
    ----------
    path : str
        Either the path to a video file (e.g. ``.mp4``) or a directory
        containing ordered frame images (``.png``, ``.jpg``, …).
    resolution : int or (H, W) tuple
        Spatial size to resize every frame to. Int → square; tuple → (H, W).
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

        # Robust search for actions/info files: walk up a few levels from the
        # frame directory, then fallback to a shallow walk under the frame dir.
        base_dir = os.path.dirname(path)
        actions_path = None
        info_path = None

        cur = base_dir
        for _ in range(6):
            cand_a = os.path.join(cur, "actions.json")
            cand_i = os.path.join(cur, "info.json")
            if actions_path is None and os.path.exists(cand_a):
                actions_path = cand_a
            if info_path is None and os.path.exists(cand_i):
                info_path = cand_i
            if actions_path and info_path:
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        # If not found, try a shallow walk under the base_dir (covers some dataset layouts).
        if actions_path is None:
            for root, dirs, files in os.walk(base_dir):
                if "actions.json" in files:
                    actions_path = os.path.join(root, "actions.json")
                    break
        if info_path is None:
            for root, dirs, files in os.walk(base_dir):
                if "info.json" in files:
                    info_path = os.path.join(root, "info.json")
                    break

        if actions_path and info_path:
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

    frames, annotations = _load_source_frames(
        path,
        resolution=resolution,
        start_frame=start_frame,
        max_frames=max_frames,
        frame_skip=frame_skip,
    )
    if annotations is not None:
        return frames, annotations
    return frames


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
    batch_size: int = 16,
) -> list:
    """Extract latent actions for all consecutive pairs in *frames*.

    Returns a list of dicts (one per consecutive pair), each containing
    ``z_mu``, ``z_var``, ``z_rep`` of shape ``(latent_dim,)``.
    For OlafWorld, ``z_var`` is zeros (the encoder is deterministic).
    """
    results = []
    n_pairs = len(frames) - 1

    for i in range(0, n_pairs, batch_size):
        end = min(i + batch_size, n_pairs)
        batch = [torch.stack([frames[j], frames[j + 1]], dim=0) for j in range(i, end)]
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

        del batch, video_batch
        if model_type == "olafworld":
            del z
        else:
            del outputs

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


# ========================== P2P Dataset =====================================


def _decode_raw_proto(data: bytes, max_depth: int = 8) -> dict:
    """Recursively decode raw protobuf binary data into a dict with field_N keys."""
    if not _HAS_PROTOBUF:
        raise RuntimeError("google.protobuf is required for P2P proto decoding")
    if max_depth <= 0:
        return {"_truncated": f"{len(data)} bytes (max depth)"}

    pos = 0
    result: dict = {}

    def _add(key, value):
        if key not in result:
            result[key] = value
        else:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[key] = [existing, value]

    while pos < len(data):
        try:
            tag, new_pos = _DecodeVarint(data, pos)
            fn = tag >> 3
            wt = tag & 0x7
            if fn == 0 or fn > 536870911:
                break
            pos = new_pos
            key = f"field_{fn}"

            if wt == WIRETYPE_VARINT:
                value, pos = _DecodeVarint(data, pos)
                _add(key, value)
            elif wt == WIRETYPE_FIXED64:
                value = struct.unpack("<d", data[pos:pos + 8])[0]
                pos += 8
                _add(key, round(value, 8))
            elif wt == WIRETYPE_LENGTH_DELIMITED:
                length, pos = _DecodeVarint(data, pos)
                if length > len(data) - pos:
                    break
                raw = data[pos:pos + length]
                pos += length
                try:
                    text = raw.decode("utf-8")
                    if all(c.isprintable() or c in "\n\r\t" for c in text):
                        _add(key, text)
                        continue
                except (UnicodeDecodeError, ValueError):
                    pass
                try:
                    sub = _decode_raw_proto(raw, max_depth - 1)
                    if sub:
                        _add(key, sub)
                    else:
                        _add(key, f"<{len(raw)} bytes binary>")
                except Exception:
                    _add(key, f"<{len(raw)} bytes binary>")
            elif wt == WIRETYPE_FIXED32:
                value = struct.unpack("<f", data[pos:pos + 4])[0]
                pos += 4
                _add(key, round(value, 6))
            else:
                break
        except Exception:
            break

    return result


def _proto_ensure_list(val):
    """Wrap a single value in a list; leave lists untouched."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _extract_keyboard_keys_from_action(action_raw: dict) -> list[str]:
    """Extract keyboard key names from a frame's field_1 (user_action) sub-message."""
    if not isinstance(action_raw, dict):
        return []
    kb_raw = action_raw.get("field_2")
    if kb_raw is None:
        return []
    if isinstance(kb_raw, str):
        keys = [k.strip() for k in kb_raw.replace("\n", "").split(",") if k.strip()]
        return keys
    if isinstance(kb_raw, dict):
        vals = kb_raw.get("field_1", [])
        return _proto_ensure_list(vals)
    return []


def _extract_mouse_from_action(action_raw: dict) -> tuple[float, float, float, int]:
    """Extract mouse data from a frame's field_1 (user_action) sub-message.

    Returns (mouse_absolute_px, mouse_absolute_py, scroll_delta_px, button).
    button: 0=left, 1=right, 2=none.
    """
    if not isinstance(action_raw, dict):
        return 0.0, 0.0, 0.0, P2P_MOUSE_BUTTON_NONE
    mouse_raw = action_raw.get("field_4")
    if not isinstance(mouse_raw, dict):
        return 0.0, 0.0, 0.0, P2P_MOUSE_BUTTON_NONE

    px = float(mouse_raw.get("field_1", 0))
    py = float(mouse_raw.get("field_2", 0))
    scroll = float(mouse_raw.get("field_3", 0))
    buttons = _proto_ensure_list(mouse_raw.get("field_4"))
    if not buttons:
        button = P2P_MOUSE_BUTTON_NONE
    elif 1 in buttons:
        button = 1  # right click
    elif 0 in buttons:
        button = 0  # left click
    else:
        button = P2P_MOUSE_BUTTON_NONE
    return px, py, scroll, button


def extract_p2p_labels(proto_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, int]:
    """Decode annotation.proto and extract per-frame keyboard + mouse labels.

    Returns
    -------
    keyboard_labels : (N, 10) int tensor — binary indicators for P2P_KEYBOARD_KEYS
    mouse_floats : (N, 3) float tensor — [px, py, scroll_delta]
    mouse_buttons : (N,) int tensor — 0=left, 1=right, 2=none
    metadata : dict — id, timestamp, user, fps, env
    n_frames : int — number of frame annotations in the proto
    """
    with open(proto_path, "rb") as f:
        data = f.read()
    decoded = _decode_raw_proto(data)

    # Extract metadata from field_2
    meta_raw = decoded.get("field_2", {})
    env_raw = meta_raw.get("field_6", {}) if isinstance(meta_raw, dict) else {}
    metadata = {
        "id": meta_raw.get("field_1", "") if isinstance(meta_raw, dict) else "",
        "timestamp": meta_raw.get("field_2", 0) if isinstance(meta_raw, dict) else 0,
        "user": meta_raw.get("field_3", "") if isinstance(meta_raw, dict) else "",
        "fps": meta_raw.get("field_5", 0) if isinstance(meta_raw, dict) else 0,
        "env": {
            "platform": env_raw.get("field_1", "") if isinstance(env_raw, dict) else "",
            "game": env_raw.get("field_2", "") if isinstance(env_raw, dict) else "",
        },
    }

    # Extract frame annotations from field_3
    raw_frames = _proto_ensure_list(decoded.get("field_3"))
    n_frames = len(raw_frames)

    key_to_idx = {k: i for i, k in enumerate(P2P_KEYBOARD_KEYS)}
    n_keys = len(P2P_KEYBOARD_KEYS)

    keyboard_labels = torch.zeros(n_frames, n_keys, dtype=torch.int)
    mouse_floats = torch.zeros(n_frames, 3, dtype=torch.float)
    mouse_buttons = torch.full((n_frames,), P2P_MOUSE_BUTTON_NONE, dtype=torch.int)

    for i, frame in enumerate(raw_frames):
        if not isinstance(frame, dict):
            continue
        action_raw = frame.get("field_1", {})

        # Keyboard
        keys = _extract_keyboard_keys_from_action(action_raw)
        for k in keys:
            idx = key_to_idx.get(k)
            if idx is not None:
                keyboard_labels[i, idx] = 1

        # Mouse
        px, py, scroll, button = _extract_mouse_from_action(action_raw)
        mouse_floats[i, 0] = px
        mouse_floats[i, 1] = py
        mouse_floats[i, 2] = scroll
        mouse_buttons[i] = button

    return keyboard_labels, mouse_floats, mouse_buttons, metadata, n_frames


def _run_batch_p2p(
    args: argparse.Namespace,
    model,
    model_type: str = "adaworld",
) -> list:
    """Process all folders in the open-p2p-subset dataset.

    For each folder:
      1. Check annotation.proto exists
      2. Decode proto → frame annotations count
      3. Read video.mp4 → video frame count
      4. Validate counts match
      5. Extract latent actions from consecutive frame pairs
      6. Extract keyboard + mouse labels from proto
      7. Save to <save-dir>/<model_type>/<uuid>/latent_actions.pt
    """
    p2p_dir = os.path.abspath(args.p2p_dir)
    save_root = args.save_dir if args.save_dir else "./latent_actions_dump"

    folders = sorted([
        d for d in os.listdir(p2p_dir)
        if os.path.isdir(os.path.join(p2p_dir, d))
    ])

    total = len(folders)
    success = 0
    skipped = 0
    failed = 0

    print(f"P2P batch mode: found {total} folder(s) in {p2p_dir}")
    print(f"Saving outputs under: {save_root}/{model_type}/...")
    print(f"Keyboard keys tracked ({len(P2P_KEYBOARD_KEYS)}): {P2P_KEYBOARD_KEYS}")
    print()

    for idx, folder in enumerate(folders, start=1):
        folder_path = os.path.join(p2p_dir, folder)
        proto_path = os.path.join(folder_path, "annotation.proto")
        video_path = os.path.join(folder_path, "video.mp4")
        run_save_dir = os.path.join(save_root, model_type, folder)
        out_pt = os.path.join(run_save_dir, "latent_actions.pt")

        # Skip if already processed
        if os.path.exists(out_pt):
            if not args.quiet:
                print(f"[{idx}/{total}] Skipping existing: {folder}")
            skipped += 1
            continue

        # Check proto exists
        if not os.path.isfile(proto_path):
            print(f"[{idx}/{total}] SKIP (no annotation.proto): {folder}", file=sys.stderr)
            skipped += 1
            continue

        # Check video exists
        if not os.path.isfile(video_path):
            print(f"[{idx}/{total}] SKIP (no video.mp4): {folder}", file=sys.stderr)
            skipped += 1
            continue

        try:
            # 1. Extract labels from proto
            keyboard_labels, mouse_floats, mouse_buttons, metadata, n_proto_frames = \
                extract_p2p_labels(proto_path)

            # 2. Read video frames
            frames = _read_video_file(video_path)
            n_video_frames = len(frames)

            # 3. Validate frame count
            if n_proto_frames != n_video_frames:
                print(
                    f"[{idx}/{total}] WARNING frame mismatch in {folder}: "
                    f"proto={n_proto_frames} video={n_video_frames}. "
                    f"Using min({n_proto_frames}, {n_video_frames}).",
                    file=sys.stderr,
                )
                n_use = min(n_proto_frames, n_video_frames)
                frames = frames[:n_use]
                keyboard_labels = keyboard_labels[:n_use]
                mouse_floats = mouse_floats[:n_use]
                mouse_buttons = mouse_buttons[:n_use]

            if len(frames) < 2:
                print(f"[{idx}/{total}] SKIP (< 2 frames): {folder}", file=sys.stderr)
                skipped += 1
                continue

            # 4. Resize frames in-place to avoid keeping two full frame lists in memory.
            resolution = args.resolution
            for i_frame, frame in enumerate(frames):
                frames[i_frame] = _resize_frame_array(frame.numpy(), resolution)
            gc.collect()

            # 5. Extract latent actions
            results = extract_latent_actions_batch(
                model, frames, args.mu_only,
                device=args.device, model_type=model_type, batch_size=args.batch_size,
            )

            # 6. Build save dict
            # Labels: use frame_i's label for pair (frame_i, frame_{i+1})
            # So labels go from index 0 to N-2 (N-1 pairs)
            kb_labels_pairs = keyboard_labels[:-1]  # (N-1, 10)
            mf_pairs = mouse_floats[:-1]            # (N-1, 3)
            mb_pairs = mouse_buttons[:-1]            # (N-1,)

            if args.mu_only:
                all_mu = torch.stack(results)
                save_dict = {"z_mu": all_mu, "z_var": None, "z_rep": None}
            else:
                all_mu = torch.stack([r["z_mu"] for r in results])
                all_var = torch.stack([r["z_var"] for r in results])
                all_rep = torch.stack([r["z_rep"] for r in results])
                save_dict = {"z_mu": all_mu, "z_var": all_var, "z_rep": all_rep}

            save_dict["keyboard_labels"] = kb_labels_pairs
            save_dict["keyboard_keys"] = P2P_KEYBOARD_KEYS
            save_dict["mouse_floats"] = mf_pairs
            save_dict["mouse_buttons"] = mb_pairs
            save_dict["metadata"] = metadata
            save_dict["game_name"] = metadata.get("env", {}).get("game", folder)

            # 7. Save
            os.makedirs(run_save_dir, exist_ok=True)
            torch.save(save_dict, out_pt)

            # Also save z_mu CSV
            z_mu_for_csv = save_dict["z_mu"]
            np.savetxt(
                os.path.join(run_save_dir, "z_mu.csv"),
                z_mu_for_csv.numpy(),
                delimiter=",",
                header=",".join([f"dim_{i}" for i in range(z_mu_for_csv.shape[1])]),
            )

            if not args.quiet:
                print(
                    f"[{idx}/{total}] OK {folder}: "
                    f"{len(results)} pairs, "
                    f"game={metadata['env'].get('game', '?')}"
                )
            del frames, results, save_dict
            gc.collect()
            success += 1

        except Exception as exc:
            exc_str = str(exc)
            if "inline_container.cc" in exc_str or "iostream error" in exc_str:
                print(f"[{idx}/{total}] oom error {folder}", file=sys.stderr)
            else:
                print(f"[{idx}/{total}] ERROR on {folder}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            failed += 1

    print(f"\nP2P batch complete. total={total} success={success} skipped={skipped} failed={failed}")
    if failed > 0:
        sys.exit(1)
    return []


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
    group.add_argument(
        "--p2p-dir", type=str,
        help="Path to the open-p2p-subset directory. Extracts latent actions + keyboard/mouse labels from annotation.proto."
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Pair batch size for latent extraction (default: 16; lower uses less memory).",
    )

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


def _find_run_video_files(root_dir: str) -> list[str]:
    """Find nested canonical video files under a dataset root."""
    pattern = os.path.join(root_dir, "**", "video.mp4")
    video_files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    video_files.sort()
    return video_files


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
    results = extract_latent_actions_batch(
        model,
        frames,
        args.mu_only,
        device=args.device,
        model_type=model_type,
        batch_size=args.batch_size,
    )
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


def _run_batch_video_root(args: argparse.Namespace, model, model_type: str = "adaworld") -> list:
    """Process all nested video.mp4 files under a dataset root."""
    root_dir = os.path.abspath(args.video)
    run_video_files = _find_run_video_files(root_dir)

    if not run_video_files:
        print(f"No video files found under {root_dir} (expected nested video.mp4 files).", file=sys.stderr)
        sys.exit(1)

    save_root = args.save_dir if args.save_dir else "./latent_actions_dump"
    total = len(run_video_files)
    success = 0
    failed = 0

    print(f"Batch mode: found {total} video(s) under {root_dir}")
    print(f"Saving outputs under: {save_root}/...")

    for idx, video_path in enumerate(run_video_files, start=1):
        rel = os.path.relpath(video_path, root_dir)
        rel_run = os.path.dirname(rel)
        if rel_run in ("", "."):
            rel_run = os.path.basename(os.path.normpath(root_dir))
        run_id = rel_run
        run_save_dir = os.path.join(save_root, model_type, rel_run)
        out_pt = os.path.join(run_save_dir, "latent_actions.pt")

        if os.path.exists(out_pt):
            print(f"[{idx}/{total}] Skipping existing: {run_id}")
            success += 1
            continue

        print(f"[{idx}/{total}] Processing: {run_id}")
        try:
            results, actions, action_names = _extract_single_source(model, video_path, args, model_type=model_type)
            save_results(results, run_save_dir, args.mu_only, actions=actions, action_names=action_names)
            success += 1
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR on {run_id}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Batch complete. total={total} success={success} failed={failed}")
    if failed > 0:
        sys.exit(1)

    return []


def main() -> None:
    args = parse_args()

    # --- Validate args ---
    if args.frame1 and not args.frame2:
        print("ERROR: --frame2 is required when using --frame1", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    # --- Load model ---
    model_type = args.model
    if model_type == "olafworld":
        model = load_olaf_encoder(args.olaf_ckpt, args.olaf_variant, device=args.device)
        if args.resolution == RESOLUTION:
            args.resolution = OLAFWORLD_RESOLUTION
    else:
        model = load_lam(args.lam_ckpt, device=args.device)

    # --- P2P dataset batch mode ---
    if hasattr(args, 'p2p_dir') and args.p2p_dir:
        return _run_batch_p2p(args, model, model_type=model_type)

    # --- Batch mode for game root directories ---
    if args.video and os.path.isdir(args.video):
        run_frame_dirs = _find_run_frame_dirs(args.video)
        if run_frame_dirs:
            return _run_batch_game_root(args, model, model_type=model_type)
        run_video_files = _find_run_video_files(args.video)
        if run_video_files:
            return _run_batch_video_root(args, model, model_type=model_type)

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
    results = extract_latent_actions_batch(
        model,
        frames,
        args.mu_only,
        device=args.device,
        model_type=model_type,
        batch_size=args.batch_size,
    )

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
