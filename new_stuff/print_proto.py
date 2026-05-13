"""
Print the decoded contents of the annotation.proto file from a random folder
inside /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset.

The .proto files are serialized protobuf binary data (not schema definitions).
We decode them into a human-readable JSON schema and pretty-print the result.

Output schema:
{
  "metadata": { ... },
  "frame_annotations": [ { "user_action": {...}, "system_action": {...},
                            "frame_text_annotation": {...} }, ... ],
  "video_summary": "...",
  "processing_log": [ ... ]
}
"""

import base64
import json
import os
import random
import struct

from google.protobuf.internal.decoder import _DecodeVarint
from google.protobuf.internal.wire_format import (
    WIRETYPE_FIXED32,
    WIRETYPE_FIXED64,
    WIRETYPE_LENGTH_DELIMITED,
    WIRETYPE_VARINT,
)

DATA_ROOT = "/scratch-shared/FoMo-Atomic-Actions/open-p2p-subset"

# Number of frame annotations to include in the printed output (None = all)
MAX_FRAMES_TO_PRINT = 10


# ---------------------------------------------------------------------------
# Low-level raw protobuf decoder
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
# Field-number → human-readable name mapping
# ---------------------------------------------------------------------------

def _ensure_list(val):
    """Wrap a single value in a list; leave lists untouched."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def map_metadata(raw: dict) -> dict:
    """Map raw field_2 (top-level metadata) to named keys."""
    env_raw = raw.get("field_6", {})
    env = {}
    if isinstance(env_raw, dict):
        env["platform"] = env_raw.get("field_1", "")
        env["game"] = env_raw.get("field_2", "")
    return {
        "id": raw.get("field_1", ""),
        "timestamp": raw.get("field_2", 0),
        "user": raw.get("field_3", ""),
        "session": raw.get("field_4", ""),
        "fps": raw.get("field_5", 0),
        "env": env,
    }


def map_keyboard(raw) -> list[str]:
    """Extract keyboard keys from the action's field_2."""
    if raw is None:
        return []
    if isinstance(raw, str):
        # Sometimes stored as a plain string like "\\n\\nRightArrow"
        keys = [k.strip() for k in raw.replace("\n", "").split(",") if k.strip()]
        return keys if keys else []
    if isinstance(raw, dict):
        vals = raw.get("field_1", [])
        return _ensure_list(vals)
    return []


def map_mouse(raw: dict) -> dict:
    """Extract mouse data from the action sub-message (if present)."""
    # Mouse data uses fixed32 fields for coordinates.
    return {
        "mouse_absolute_px": raw.get("field_1", 0),
        "mouse_absolute_py": raw.get("field_2", 0),
        "scroll_delta_px": raw.get("field_3", 0),
        "buttons_down": _ensure_list(raw.get("field_4")),
    }


def map_user_action(raw: dict) -> dict | None:
    """Map field_1 of a frame annotation to user_action."""
    if not raw or not isinstance(raw, dict):
        return None

    keyboard = map_keyboard(raw.get("field_2"))

    # Mouse data would be in field_4 if present
    mouse_raw = raw.get("field_4")
    mouse = map_mouse(mouse_raw) if isinstance(mouse_raw, dict) else None

    action = {"keyboard": keyboard}
    if mouse is not None:
        action["mouse"] = mouse
    return action


def map_text_annotation(raw: dict) -> dict | None:
    """Map field_7 of a frame annotation to frame_text_annotation."""
    if not raw or not isinstance(raw, dict):
        return None

    annotator_raw = raw.get("field_2", {})
    annotator = ""
    if isinstance(annotator_raw, dict):
        annotator = annotator_raw.get("field_2", annotator_raw.get("field_1", ""))

    # text_embedding_dict lives in field_4
    embed_raw = raw.get("field_4", {})
    text_embedding_dict = {}
    if isinstance(embed_raw, dict):
        tokenizer = embed_raw.get("field_1", "unknown")
        # The actual embedding is large binary; just note its presence
        embed_inner = embed_raw.get("field_2", {})
        if isinstance(embed_inner, dict):
            model_info = embed_inner.get("field_1", {})
            if isinstance(model_info, dict):
                tokenizer = model_info.get("field_1", tokenizer)
                embed_data = model_info.get("field_2", {})
                if isinstance(embed_data, dict):
                    size = embed_data.get("field_2", "")
                    if isinstance(size, str) and "bytes" in size:
                        text_embedding_dict[tokenizer] = size
                    else:
                        text_embedding_dict[tokenizer] = "<embedding vector>"
                else:
                    text_embedding_dict[tokenizer] = "<embedding vector>"

    return {
        "instruction": raw.get("field_1", ""),
        "frame_text_annotator": annotator,
        "duration": raw.get("field_3", 0),
        "text_embedding_dict": text_embedding_dict if text_embedding_dict else None,
    }


def map_frame_annotation(raw: dict) -> dict:
    """Map a single frame annotation (field_3 entry) to the output schema."""
    if not isinstance(raw, dict):
        return {}

    # field_1 = user_action, field_6 = system_action, field_7 = text annotation
    user_action = map_user_action(raw.get("field_1"))
    system_action_raw = raw.get("field_6")
    system_action = None
    if isinstance(system_action_raw, dict):
        system_action = map_user_action(system_action_raw)

    text_annotation = map_text_annotation(raw.get("field_7"))

    result = {}
    if user_action is not None:
        result["user_action"] = user_action
    if system_action is not None:
        result["system_action"] = system_action
    if text_annotation is not None:
        result["frame_text_annotation"] = text_annotation
    return result


def map_processing_log(raw) -> list[dict]:
    """Map field_8 entries to processing log."""
    entries = _ensure_list(raw)
    out = []
    for entry in entries:
        if isinstance(entry, dict):
            out.append({
                "action": entry.get("field_1", ""),
                "description": entry.get("field_2", ""),
                "timestamp": entry.get("field_3", 0),
            })
    return out


def build_output(decoded: dict) -> dict:
    """Transform the raw decoded dict into the human-readable output schema."""
    # Metadata
    metadata = map_metadata(decoded.get("field_2", {}))

    # Frame annotations
    raw_frames = _ensure_list(decoded.get("field_3"))
    frame_annotations = [map_frame_annotation(f) for f in raw_frames]

    # Video summary
    video_summary = None
    vs = decoded.get("field_7")
    if isinstance(vs, dict):
        video_summary = vs.get("field_1", "")
    elif isinstance(vs, str):
        video_summary = vs

    # Processing log
    processing_log = map_processing_log(decoded.get("field_8"))

    output = {
        "metadata": metadata,
        "frame_annotations": frame_annotations,
    }
    if video_summary:
        output["video_summary"] = video_summary
    if processing_log:
        output["processing_log"] = processing_log
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    folders = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    if not folders:
        print(f"No folders found in {DATA_ROOT}")
        return

    chosen = random.choice(folders)
    proto_path = os.path.join(DATA_ROOT, chosen, "annotation.proto")

    print(f"{'=' * 70}")
    print(f"Random folder: {chosen}")
    print(f"Proto file:    {proto_path}")
    print(f"{'=' * 70}")

    if not os.path.isfile(proto_path):
        print(f"ERROR: annotation.proto not found in {chosen}")
        print("Contents:", os.listdir(os.path.join(DATA_ROOT, chosen)))
        return

    file_size = os.path.getsize(proto_path)
    print(f"File size:     {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"{'=' * 70}\n")

    with open(proto_path, "rb") as f:
        data = f.read()

    decoded = decode_raw_to_dict(data)
    if not decoded:
        print("Could not decode any fields from the proto file.")
        return

    output = build_output(decoded)

    total_frames = len(output["frame_annotations"])
    output["_total_frame_annotations"] = total_frames

    print(output.keys())
    for i in range(1000):
        print(output['frame_annotations'][i])
    #print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

