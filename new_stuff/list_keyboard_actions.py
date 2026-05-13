"""
Scan ALL folders in /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset
and collect every unique keyboard key and mouse control that appears in user_action fields.

Outputs:
  1. A sorted list of all unique individual keys (e.g. "w", "RightArrow", "spacebar")
  2. A sorted list of all unique key *combinations* per frame (e.g. "a + d + w")
  3. Per-key frequency counts
  4. Mouse button counts (left click, right click, no click)
  5. Histograms of mouse float values (mouse_absolute_px, mouse_absolute_py, scroll_delta_px)
"""

import json
import os
import struct
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

from google.protobuf.internal.decoder import _DecodeVarint
from google.protobuf.internal.wire_format import (
    WIRETYPE_FIXED32,
    WIRETYPE_FIXED64,
    WIRETYPE_LENGTH_DELIMITED,
    WIRETYPE_VARINT,
)

DATA_ROOT = "/scratch-shared/FoMo-Atomic-Actions/open-p2p-subset"


# ---------------------------------------------------------------------------
# Low-level raw protobuf decoder (same as print_proto.py)
# ---------------------------------------------------------------------------

def decode_raw_to_dict(data: bytes, max_depth: int = 8) -> dict:
    """Recursively decode raw protobuf binary data into a dict with field_N keys."""
    if max_depth <= 0:
        return {"_truncated": f"{len(data)} bytes (max depth)"}

    pos = 0
    result: dict = {}

    def _add(key: str, value):
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
            field_number = tag >> 3
            wire_type = tag & 0x7
            if field_number == 0 or field_number > 536870911:
                break
            pos = new_pos
            key = f"field_{field_number}"

            if wire_type == WIRETYPE_VARINT:
                value, pos = _DecodeVarint(data, pos)
                _add(key, value)
            elif wire_type == WIRETYPE_FIXED64:
                value = struct.unpack("<d", data[pos : pos + 8])[0]
                pos += 8
                _add(key, round(value, 8))
            elif wire_type == WIRETYPE_LENGTH_DELIMITED:
                length, pos = _DecodeVarint(data, pos)
                if length > len(data) - pos:
                    break
                raw = data[pos : pos + length]
                pos += length
                try:
                    text = raw.decode("utf-8")
                    if all(c.isprintable() or c in "\n\r\t" for c in text):
                        _add(key, text)
                        continue
                except (UnicodeDecodeError, ValueError):
                    pass
                try:
                    sub = decode_raw_to_dict(raw, max_depth - 1)
                    if sub:
                        _add(key, sub)
                    else:
                        _add(key, f"<{len(raw)} bytes binary>")
                except Exception:
                    _add(key, f"<{len(raw)} bytes binary>")
            elif wire_type == WIRETYPE_FIXED32:
                value = struct.unpack("<f", data[pos : pos + 4])[0]
                pos += 4
                _add(key, round(value, 6))
            else:
                break
        except Exception:
            break

    return result


# ---------------------------------------------------------------------------
# Keyboard key extraction
# ---------------------------------------------------------------------------

def _ensure_list(val):
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def extract_keyboard_keys(action_raw) -> list[str]:
    """Extract keyboard keys from a frame's field_1 (user_action) sub-message."""
    if not isinstance(action_raw, dict):
        return []

    kb_raw = action_raw.get("field_2")
    if kb_raw is None:
        return []

    if isinstance(kb_raw, str):
        # Sometimes stored as plain string like "\n\nRightArrow"
        keys = [k.strip() for k in kb_raw.replace("\n", "").split(",") if k.strip()]
        return keys

    if isinstance(kb_raw, dict):
        vals = kb_raw.get("field_1", [])
        return _ensure_list(vals)

    return []


def extract_mouse_data(action_raw) -> tuple[float | None, float | None, float | None, int | None]:
    """Extract mouse data from a frame's field_1 (user_action) sub-message.

    Returns (mouse_absolute_px, mouse_absolute_py, scroll_delta_px, button).
    button: 0=left, 1=right, None=not present.
    Returns all Nones if no mouse data is present.
    """
    if not isinstance(action_raw, dict):
        return None, None, None, None
    mouse_raw = action_raw.get("field_4")
    if not isinstance(mouse_raw, dict):
        return None, None, None, None

    px = float(mouse_raw.get("field_1", 0))
    py = float(mouse_raw.get("field_2", 0))
    scroll = float(mouse_raw.get("field_3", 0))
    buttons = _ensure_list(mouse_raw.get("field_4"))
    if not buttons:
        button = None  # no click
    elif 1 in buttons:
        button = 1  # right click
    elif 0 in buttons:
        button = 0  # left click
    else:
        button = None
    return px, py, scroll, button


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    folders = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])

    print(f"Scanning {len(folders)} folders in {DATA_ROOT}...\n")

    all_individual_keys: Counter = Counter()  # individual key → count
    all_combos: Counter = Counter()           # sorted combo string → count
    total_frames = 0
    frames_with_keys = 0
    errors = 0

    # Mouse data collectors
    mouse_button_counts: Counter = Counter()  # button state → count
    all_mouse_px: list[float] = []
    all_mouse_py: list[float] = []
    all_mouse_scroll: list[float] = []
    frames_with_mouse = 0

    for idx, folder in enumerate(folders):
        proto_path = os.path.join(DATA_ROOT, folder, "annotation.proto")
        if not os.path.isfile(proto_path):
            continue

        try:
            with open(proto_path, "rb") as f:
                data = f.read()
            decoded = decode_raw_to_dict(data)
        except Exception as e:
            print(f"  ERROR reading {folder}: {e}")
            errors += 1
            continue

        raw_frames = decoded.get("field_3")
        if raw_frames is None:
            continue
        if not isinstance(raw_frames, list):
            raw_frames = [raw_frames]

        for frame in raw_frames:
            total_frames += 1
            if not isinstance(frame, dict):
                continue

            action_raw = frame.get("field_1")

            # Keyboard
            keys = extract_keyboard_keys(action_raw)
            if keys:
                frames_with_keys += 1
                for k in keys:
                    all_individual_keys[k] += 1
                combo = " + ".join(sorted(keys))
                all_combos[combo] += 1

            # Mouse
            px, py, scroll, button = extract_mouse_data(action_raw)
            if px is not None:  # mouse data present
                frames_with_mouse += 1
                all_mouse_px.append(px)
                all_mouse_py.append(py)
                all_mouse_scroll.append(scroll)
                if button == 0:
                    mouse_button_counts['left_click'] += 1
                elif button == 1:
                    mouse_button_counts['right_click'] += 1
                else:
                    mouse_button_counts['no_click'] += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(folders)} folders "
                  f"({len(all_individual_keys)} unique keys so far)")

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Total folders scanned:   {len(folders)}")
    print(f"Total frames:            {total_frames:,}")
    print(f"Frames with keyboard:    {frames_with_keys:,} "
          f"({100 * frames_with_keys / max(total_frames, 1):.1f}%)")
    print(f"Frames with mouse:       {frames_with_mouse:,} "
          f"({100 * frames_with_mouse / max(total_frames, 1):.1f}%)")
    print(f"Errors:                  {errors}")

    print(f"\n{'=' * 70}")
    print(f"UNIQUE INDIVIDUAL KEYS ({len(all_individual_keys)} total)")
    print(f"{'=' * 70}")
    for key, count in all_individual_keys.most_common():
        print(f"  {key:30s}  {count:>8,} frames")

    print(f"\n{'=' * 70}")
    print(f"UNIQUE KEY COMBINATIONS ({len(all_combos)} total)")
    print(f"{'=' * 70}")
    for combo, count in all_combos.most_common():
        print(f"  {combo:50s}  {count:>8,} frames")

    # -----------------------------------------------------------------------
    # Mouse results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"MOUSE BUTTON COUNTS")
    print(f"{'=' * 70}")
    if mouse_button_counts:
        for btn, count in mouse_button_counts.most_common():
            print(f"  {btn:30s}  {count:>8,} frames")
    else:
        print("  (no mouse button data found)")

    print(f"\n{'=' * 70}")
    print(f"MOUSE FLOAT STATISTICS")
    print(f"{'=' * 70}")

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs_proto')
    os.makedirs(plot_dir, exist_ok=True)

    for name, values in [
        ("mouse_absolute_px", all_mouse_px),
        ("mouse_absolute_py", all_mouse_py),
        ("scroll_delta_px", all_mouse_scroll),
    ]:
        if not values:
            print(f"  {name}: (no data)")
            continue

        arr = np.array(values)
        print(f"\n  {name}:")
        print(f"    count:  {len(arr):,}")
        print(f"    min:    {arr.min():.4f}")
        print(f"    max:    {arr.max():.4f}")
        print(f"    mean:   {arr.mean():.4f}")
        print(f"    std:    {arr.std():.4f}")
        print(f"    median: {np.median(arr):.4f}")

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        # Filter out extreme outliers for better visualization
        q01, q99 = np.percentile(arr, [1, 99])
        filtered = arr[(arr >= q01) & (arr <= q99)]
        ax.hist(filtered, bins=100, edgecolor='black', alpha=0.7, color='#2196F3')
        ax.set_title(f'{name} distribution (1st-99th percentile)', fontsize=14)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Frame count', fontsize=12)
        ax.axvline(arr.mean(), color='red', linestyle='--', linewidth=1.5, label=f'mean={arr.mean():.2f}')
        ax.axvline(np.median(arr), color='orange', linestyle='--', linewidth=1.5, label=f'median={np.median(arr):.2f}')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(plot_dir, f'histogram_{name}.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"    histogram saved: {plot_path}")


if __name__ == "__main__":
    main()
