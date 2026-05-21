import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from einops import rearrange

import extract_latent_actions as ela

@torch.no_grad()
def process_skipped_frames_run(model, run_dir, args, model_type):
    # `run_dir` is typically the "frames" dir, so we check its parent for actions.json
    parent_dir = os.path.dirname(run_dir) if os.path.basename(run_dir) == "frames" else run_dir
    actions_file = os.path.join(parent_dir, "actions.json")
    if not os.path.exists(actions_file):
        raise FileNotFoundError(f"Missing actions.json in {parent_dir}")
        
    with open(actions_file, "r") as f:
        data = json.load(f)
        
    actions_raw = data.get("actions", [])
    action_vocab = [v[0] for v in data.get("action_vocab", [])]
    action_descriptions = data.get("action_descriptions", [])
    noop = [0] * len(actions_raw[0]['action']) if len(actions_raw) > 0 else []
    
    pairs_to_extract = []
    actions_meta = []
    ignore_next = False
    
    current_idx = 0
    block_num = 0
    while current_idx + 20 <= len(actions_raw):
        block = actions_raw[current_idx:current_idx+20]
        action_val = block[16]['action']
        
        desc = "none"
        if action_val != noop:
            try:
                active_idx = action_val.index(1)
                if active_idx + 1 < len(action_descriptions):
                    desc = action_descriptions[active_idx + 1].lower()
            except ValueError:
                pass
                
        if ignore_next:
            ignore_next = False
            current_idx += 20
            block_num += 1
            continue
            
        if block_num == 0 and desc == 'left':
            current_idx += 20
            block_num += 1
            continue
            
        if desc != 'none':
            src_idx = current_idx + 15
            tgt_idx = None
            if desc in ['right', 'left']:
                tgt_idx = current_idx + 16 + 6
            elif desc == 'crouch':
                tgt_idx = current_idx + 16 + 3
            elif desc == 'jump':
                tgt_idx = current_idx + 16 + 10
                ignore_next = True
            else:
                tgt_idx = current_idx + 16 + 4
                
            pairs_to_extract.append((src_idx, tgt_idx))
            actions_meta.append({
                "block": block_num,
                "action": action_val,
                "desc": desc,
                "src_idx": src_idx,
                "tgt_idx": tgt_idx
            })
            
        current_idx += 20
        block_num += 1
        
    if not pairs_to_extract:
        return [], []
        
    max_frame_needed = max(t for s, t in pairs_to_extract) + 1
    
    video_out = ela.load_video_frames(
        run_dir,
        resolution=args.resolution,
        start_frame=0,
        max_frames=max_frame_needed,
        frame_skip=1
    )
    if isinstance(video_out, tuple) and len(video_out) == 2:
        frames, _ = video_out
    else:
        frames = video_out
        
    valid_pairs = []
    valid_meta = []
    for (s, t), meta in zip(pairs_to_extract, actions_meta):
        if s < len(frames) and t < len(frames):
            valid_pairs.append((s, t))
            valid_meta.append(meta)
            
    if not valid_pairs:
        return [], []
        
    results = []
    batch_size = args.batch_size
    n_pairs = len(valid_pairs)
    
    for i in range(0, n_pairs, batch_size):
        end = min(i + batch_size, n_pairs)
        batch = [torch.stack([frames[s], frames[t]], dim=0) for s, t in valid_pairs[i:end]]
        video_batch = torch.stack(batch, dim=0).to(args.device)
        
        if model_type == "olafworld":
            z = model(video_batch)
            if z.dim() == 3:
                z = z.squeeze(1)
            for j in range(z.size(0)):
                z_j = z[j].cpu()
                if args.mu_only:
                    results.append(z_j)
                else:
                    results.append({
                        "z_mu": z_j,
                        "z_var": torch.zeros_like(z_j),
                        "z_rep": z_j,
                    })
        elif model_type == "adaworld":
            outputs = model.lam.encode(video_batch)
            for j in range(video_batch.size(0)):
                z_mu_j = outputs["z_mu"][j].cpu()
                z_var_j = outputs["z_var"][j].cpu()
                z_rep_j = outputs["z_rep"][j].cpu()
                if args.mu_only:
                    results.append(z_mu_j)
                else:
                    results.append({
                        "z_mu": z_mu_j,
                        "z_var": z_var_j,
                        "z_rep": z_rep_j,
                    })

    return results, valid_meta

def main():
    args = ela.parse_args()
    
    if args.model == "adaworld":
        model = ela.load_lam(args.lam_ckpt, device=args.device)
    else:
        model = ela.load_olaf_encoder(args.lam_ckpt, variant="align", device=args.device)
        
    root_dir = os.path.abspath(args.video)
    run_frame_dirs = ela._find_run_frame_dirs(root_dir)
    
    if not run_frame_dirs:
        print(f"No run folders found under {root_dir}.")
        sys.exit(1)
        
    save_root = args.save_dir if args.save_dir else "./latent_actions_skipped"
    total = len(run_frame_dirs)
    success = 0
    
    print(f"Batch mode: found {total} run(s) under {root_dir}")
    print(f"Saving outputs under: {save_root}/...")
    
    for idx, run_frames in enumerate(run_frame_dirs, start=1):
        rel = os.path.relpath(run_frames, root_dir)
        rel_run = os.path.dirname(rel)
        rel_parts = Path(rel).parts
        if len(rel_parts) == 3:
            run_id = f"{os.path.basename(os.path.normpath(root_dir))}/{rel_run}"
            run_save_dir = os.path.join(save_root, args.model, os.path.basename(os.path.normpath(root_dir)), rel_run)
        else:
            run_id = rel_run
            run_save_dir = os.path.join(save_root, args.model, rel_run)
            
        out_pt = os.path.join(run_save_dir, "latent_actions.pt")
        if os.path.exists(out_pt):
            print(f"[{idx}/{total}] Skipping existing: {run_id}")
            continue
            
        print(f"[{idx}/{total}] Processing: {run_id}")
        try:
            results, meta = process_skipped_frames_run(model, run_frames, args, args.model)
            if results:
                ela.save_results(results, run_save_dir, args.mu_only, actions=meta)
                success += 1
            else:
                print(f"[{idx}/{total}] No valid pairs for {run_id}")
        except Exception as e:
            print(f"[{idx}/{total}] ERROR on {run_id}: {e}")

if __name__ == "__main__":
    main()
