import argparse
import glob
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Extract all latents from latent_actions_dump and compute statistics.")
    parser.add_argument("--dump-dir", type=int, choices=[1,2,3], default=1,
                        help="Which dump dir to use: 1 -> latent_actions_dump, 2 -> latent_actions_dump_2")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset subdir in the dump directory (e.g., adaworld)")
    parser.add_argument("--delete", action="store_true",
                        help="Skip (delete) samples with multiple simultaneous actions. "
                             "Default: count same sample once per individual action.")
    args = parser.parse_args()

    if args.dump_dir == 3:
        base = 'latent_actions_videoflextok'
    elif args.dump_dir == 2:
        base = 'latent_actions_dump_2'
    else:
        base = 'latent_actions_dump'

    dump_dir = os.path.join(base, args.dataset)
    files = glob.glob(os.path.join(dump_dir, "**", "latent_actions.pt"), recursive=True)
    print(f"Found {len(files)} latent action files in {args.dataset}, in {dump_dir}", flush=True)

    all_z_mu = []
    game_z_mu = {}
    action_z_mu = {}
    game_action_z_mu = {}
    original_action_z_mu = {}
    game_action_z_mu_single = {}

    for f in files:
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            continue
            
        parts = f.split('/')
        # Prefer game_name saved in the .pt file (set by P2P extraction from proto metadata).
        # Fall back to path-based detection for old retro dumps.
        if data.get('game_name'):
            game_name = data['game_name']
        elif len(parts) >= 5:
            game_name = parts[-4]  # retro: .../game_name/seed/episode/latent_actions.pt
        else:
            game_name = parts[-2] if len(parts) >= 2 else 'unknown'
        
        if 'z_mu' not in data:
            continue

        z_mu = data['z_mu'] # (N, D)

        # Build actions list
        actions = []
        raw_actions = data.get('actions', [])
        valid_actions = {'right', 'left', 'crouch', 'jump'}
        for a in raw_actions:
            if isinstance(a, dict) and 'desc' in a:
                desc = a['desc']
                if desc in valid_actions:
                    actions.append((desc,))
                else:
                    actions.append(('none',))
            else:
                actions.append(('none',))
        
        all_z_mu.append(z_mu)
        
        if game_name not in game_z_mu:
            game_z_mu[game_name] = []
        game_z_mu[game_name].append(z_mu)

        if len(actions) == len(z_mu):
            if game_name not in game_action_z_mu:
                game_action_z_mu[game_name] = {}

            for i in range(len(actions)):
                if actions[i] is None:
                    continue
                act = tuple(actions[i]) if not isinstance(actions[i], tuple) else actions[i]
                if not act or act == ('none',):
                    continue

                z_mu_i = z_mu[i].unsqueeze(0) # (1, D)

                if args.dump_dir == 2:
                    if not (args.delete and len(act) > 1):
                        if act not in original_action_z_mu:
                            original_action_z_mu[act] = []
                        original_action_z_mu[act].append(z_mu_i)

                    if len(act) == 1:
                        if game_name not in game_action_z_mu_single:
                            game_action_z_mu_single[game_name] = {}
                        if act not in game_action_z_mu_single[game_name]:
                            game_action_z_mu_single[game_name][act] = []
                        game_action_z_mu_single[game_name][act].append(z_mu_i)

                # Determine individual action keys to group by
                if len(act) > 1:
                    if args.delete:
                        # --delete mode: skip multi-action tuples entirely
                        continue
                    else:
                        # Default: expand tuple into individual single-key entries
                        individual_acts = [(_a,) for _a in act]
                else:
                    individual_acts = [act]

                for single_act in individual_acts:
                    if single_act not in action_z_mu:
                        action_z_mu[single_act] = []
                    action_z_mu[single_act].append(z_mu_i)

                    if single_act not in game_action_z_mu[game_name]:
                        game_action_z_mu[game_name][single_act] = []
                    game_action_z_mu[game_name][single_act].append(z_mu_i)

    if not all_z_mu:
        print("No z_mu data found.")
        return

    print("Compiling global data...", flush=True)
    all_z_mu_cat = torch.cat(all_z_mu, dim=0) # (Total, D)
    overall_var = torch.var(all_z_mu_cat, dim=0, unbiased=False)
    
    print("\n" + "="*80)
    print("  OVERALL VARIANCE (All Dataset)")
    print("="*80)
    print(f"Total samples: {all_z_mu_cat.shape[0]}")
    print("Variance per dimension:")
    print(np.array2string(overall_var.numpy()[:10], precision=4, suppress_small=True, separator=', '))



    print("\n" + "="*80)
    print("  VARIANCE PER ACTION")
    print("="*80)
    
    # Pre-cat for actions
    action_cats = {}
    for act, z_mus in action_z_mu.items():
        z_mu_cat = torch.cat(z_mus, dim=0)
        action_cats[act] = z_mu_cat
        
    for act in sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True):
        z_mu_cat = action_cats[act]
        var = torch.var(z_mu_cat, dim=0, unbiased=False) if z_mu_cat.shape[0] > 1 else torch.zeros(z_mu_cat.shape[1])
        print(f"\nAction Tuple: {act} (Samples: {z_mu_cat.shape[0]})")
        print(np.array2string(var.numpy()[:10], precision=4, suppress_small=True, separator=', '))


    print("\n" + "="*80)
    print("  ACTION DISTRIBUTION (All Dataset)")
    print("="*80)
    total_action_samples = sum(action_cats[a].shape[0] for a in action_cats)
    print(f"Total samples with actions: {total_action_samples}")
    for act in sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True):
        count = action_cats[act].shape[0]
        pct = 100.0 * count / total_action_samples if total_action_samples > 0 else 0.0
        print(f"  {act}: {count} ({pct:.2f}%)")

    # ------------------------------------------------------------------ PCA --
    print("\n" + "="*80)
    print("  PCA — ALL DATA")
    print("="*80)
    fit_and_print_pca("All data", all_z_mu_cat.numpy(), n_vectors=3)



    print("\n" + "="*80)
    print("  PCA — PER ACTION")
    print("="*80)
    for act in sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True):
        fit_and_print_pca(f"Action {act}", action_cats[act].numpy(), top_k=3)

    # ---- build labeled dataset shared by the four analyses below ----
    action_list = sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True)
    if not action_list:
        print("No per-step action data found; skipping action-labelled analyses.")
        return
    action_label = {act: f"A{i}" for i, act in enumerate(action_list)}
    X_labeled = np.concatenate([action_cats[act].numpy() for act in action_list], axis=0)
    y_labeled  = np.concatenate([
        np.full(action_cats[act].shape[0], i, dtype=np.int32)
        for i, act in enumerate(action_list)
    ])

    print("\nAction legend:")
    for act in action_list:
        print(f"  {action_label[act]}: {act}  (n={action_cats[act].shape[0]})")

    # ----------------------------------------------------------- 1. CENTROIDS --
    print("\n" + "="*80)
    print("  1. ACTION CENTROID DISTANCES")
    print("="*80)

    centroids = np.stack([action_cats[act].numpy().mean(axis=0) for act in action_list])

    print("\nPer-action centroids:")
    for act, c in zip(action_list, centroids):
        print(f"  {action_label[act]}: {np.array2string(c[:10], precision=4, suppress_small=True, separator=', ', max_line_width=120)}")

    print("\nPairwise L2 centroid distances:")
    n_acts = len(action_list)
    print("        " + "".join(f"  {action_label[a]:>6}" for a in action_list))
    for i, act_i in enumerate(action_list):
        row = f"  {action_label[act_i]:>6}  "
        for j in range(n_acts):
            row += f"  {np.linalg.norm(centroids[i] - centroids[j]):6.4f}"
        print(row)

    # -------------------------------------------------- 2. FISHER CRITERION --
    print("\n" + "="*80)
    print("  2a. FISHER CRITERION (Original Dimensions)")
    print("="*80)

    D_orig = X_labeled.shape[1]
    global_mean_orig = X_labeled.mean(axis=0)
    between_var_orig = np.zeros(D_orig)
    within_var_orig  = np.zeros(D_orig)
    n_total = X_labeled.shape[0]

    for i, act in enumerate(action_list):
        mask = y_labeled == i
        X_act = X_labeled[mask]
        n_k = X_act.shape[0]
        mu_k = X_act.mean(axis=0)
        between_var_orig += n_k * (mu_k - global_mean_orig) ** 2
        within_var_orig += ((X_act - mu_k) ** 2).sum(axis=0)

    between_var_orig /= n_total
    within_var_orig /= n_total
    fisher_orig = between_var_orig / (within_var_orig + 1e-10)

    print(f"\n  {'Dim':>4}  {'Fisher':>10}  {'Between':>10}  {'Within':>10}")
    for d in np.argsort(fisher_orig)[::-1][:10]:
        print(f"  {d:>4}  {fisher_orig[d]:>10.6f}  {between_var_orig[d]:>10.6f}  {within_var_orig[d]:>10.6f}")

    print("\n" + "="*80)
    print("  2b. FISHER CRITERION (Principal Components)")
    print("="*80)

    pca_fisher = PCA()
    X_proj = pca_fisher.fit_transform(X_labeled)

    D_proj = X_proj.shape[1]
    global_mean_proj = X_proj.mean(axis=0)
    between_var_proj = np.zeros(D_proj)
    within_var_proj  = np.zeros(D_proj)

    for i, act in enumerate(action_list):
        mask = y_labeled == i
        X_act_proj = X_proj[mask]
        n_k = X_act_proj.shape[0]
        mu_k = X_act_proj.mean(axis=0)
        between_var_proj += n_k * (mu_k - global_mean_proj) ** 2
        within_var_proj += ((X_act_proj - mu_k) ** 2).sum(axis=0)

    between_var_proj /= n_total
    within_var_proj /= n_total
    fisher_proj = between_var_proj / (within_var_proj + 1e-10)

    print(f"\n  {'PC':>4}  {'Fisher':>10}  {'Between':>10}  {'Within':>10}")
    for d in np.argsort(fisher_proj)[::-1][:10]:
        print(f"  {d+1:>4}  {fisher_proj[d]:>10.6f}  {between_var_proj[d]:>10.6f}  {within_var_proj[d]:>10.6f}")

    # ----------------------------------------------------------------- 3. LDA --
    print("\n" + "="*80)
    print("  3. Linear Discriminant Analysis")
    print("="*80)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_labeled, y_labeled)
    lda_ratio = lda.explained_variance_ratio_
    lda_cum   = np.cumsum(lda_ratio)

    print(f"\n  {'LD':>3}  {'VarRatio':>10}  {'CumRatio':>10}")
    for i, (r, c) in enumerate(zip(lda_ratio, lda_cum)):
        print(f"  {i+1:>3}  {r:>10.6f}  {c:>10.6f}")

    print("\nTop 3 discriminant direction vectors (unit-normalised):")
    for i in range(min(3, lda.scalings_.shape[1])):
        vec = lda.scalings_[:, i]
        vec = vec / np.linalg.norm(vec)
        print(f"  LD{i+1}: {np.array2string(vec[:10], precision=4, suppress_small=True, separator=', ', max_line_width=120)}")

    # -------------------------------------------------------- 4. LINEAR PROBE --
    print("\n" + "="*80)
    print("  4. LINEAR PROBE (Multi-class: predict 1 action at a time)")
    print("="*80)

    if len(X_labeled) > 0:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_labeled, y_labeled, test_size=0.2, random_state=42
        )
        
        probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
        probe.fit(X_tr, y_tr)
        
        y_pred = probe.predict(X_te)
        
        acc = accuracy_score(y_te, y_pred)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_te, y_pred, average='micro', zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_te, y_pred, average='macro', zero_division=0
        )
        
        print(f"\n  Overall Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Micro P/R/F1:     {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
        print(f"  Macro P/R/F1:     {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
        print(f"\n  Per-class metrics:")
        
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_te, y_pred, average=None, labels=np.arange(len(action_list)), zero_division=0
        )
        
        for i, act in enumerate(action_list):
            print(
                f"    Class {i}: {str(act[0]):15s} — "
                f"P/R/F1: {p_class[i]:.4f}/{r_class[i]:.4f}/{f1_class[i]:.4f}  "
                f"(support={support[i]})"
            )

    # --------------------------------------------------- 5. ENTANGLEMENT TEST --
    print("\n" + "="*80)
    print("  5. ENTANGLEMENT TEST")
    print("="*80)
    
    # We want to compare:
    # 1. Average distance between latents with SAME action, DIFFERENT game
    # 2. Average distance between latents with DIFFERENT action, SAME game
    
    num_samples_per_pair = 1000
    
    same_act_diff_game_dists = []
    diff_act_same_game_dists = []
    
    entanglement_source = game_action_z_mu
    if args.dump_dir == 2 and game_action_z_mu_single:
        entanglement_source = game_action_z_mu_single
        print("  Using only single-action latents (no expansion) for entanglement.")

    # 1. Same action, different game
    for act in action_list:
        games_with_act = [g for g in entanglement_source if act in entanglement_source[g]]
        if len(games_with_act) < 2:
            continue
        for i in range(len(games_with_act)):
            for j in range(i + 1, len(games_with_act)):
                g1, g2 = games_with_act[i], games_with_act[j]
                
                z1 = torch.cat(entanglement_source[g1][act], dim=0)
                z2 = torch.cat(entanglement_source[g2][act], dim=0)
                
                n_samples1 = min(z1.shape[0], num_samples_per_pair)
                n_samples2 = min(z2.shape[0], num_samples_per_pair)
                
                idx1 = torch.randperm(z1.shape[0])[:n_samples1]
                idx2 = torch.randperm(z2.shape[0])[:n_samples2]
                
                # Pairwise distances
                dists = torch.cdist(z1[idx1], z2[idx2])
                same_act_diff_game_dists.append(dists.mean().item())
                
    # 2. Different action, same game
    for g in entanglement_source:
        acts_in_game = [act for act in action_list if act in entanglement_source[g]]
        if len(acts_in_game) < 2:
            continue
        for i in range(len(acts_in_game)):
            for j in range(i + 1, len(acts_in_game)):
                a1, a2 = acts_in_game[i], acts_in_game[j]
                
                z1 = torch.cat(entanglement_source[g][a1], dim=0)
                z2 = torch.cat(entanglement_source[g][a2], dim=0)
                
                n_samples1 = min(z1.shape[0], num_samples_per_pair)
                n_samples2 = min(z2.shape[0], num_samples_per_pair)
                
                idx1 = torch.randperm(z1.shape[0])[:n_samples1]
                idx2 = torch.randperm(z2.shape[0])[:n_samples2]
                
                dists = torch.cdist(z1[idx1], z2[idx2])
                diff_act_same_game_dists.append(dists.mean().item())

    mean_same_act = np.mean(same_act_diff_game_dists) if same_act_diff_game_dists else float('nan')
    mean_diff_act = np.mean(diff_act_same_game_dists) if diff_act_same_game_dists else float('nan')
    
    print(f"  Avg distance (Same Action, Diff Game)    : {mean_same_act:.4f}  (n_pairs={len(same_act_diff_game_dists)})")
    print(f"  Avg distance (Diff Action, Same Game)    : {mean_diff_act:.4f}  (n_pairs={len(diff_act_same_game_dists)})")
    if not np.isnan(mean_same_act) and not np.isnan(mean_diff_act):
        print(f"  Ratio (SameAct-DiffGame / DiffAct-SameGame) : {mean_same_act / mean_diff_act:.4f}")
        print("  (A lower ratio implies actions are well-aligned across games despite domain gap)")


def fit_and_print_pca(label, X_np, top_k=None, n_vectors=0):
    n, d = X_np.shape
    if n < 2:
        print(f"  (skipped — only {n} sample)")
        return
    n_components = min(n, d)
    pca = PCA(n_components=n_components)
    pca.fit(X_np)
    var = pca.explained_variance_
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    show = min(top_k, n_components) if top_k is not None else n_components
    print(f"\n{label} (samples={n}, dims={d}, components={n_components})")
    header = f"  {'PC':>4}  {'Variance':>12}  {'VarRatio':>10}  {'CumRatio':>10}"
    print(header)
    for i in range(show):
        print(f"  {i+1:>4}  {var[i]:>12.6f}  {ratio[i]:>10.6f}  {cumulative[i]:>10.6f}")
        if i < n_vectors:
            vec = pca.components_[i]
            print(f"        vector: {np.array2string(vec[:10], precision=4, suppress_small=True, separator=', ', max_line_width=120)}")
    weighted_avg = (ratio[:, None] * pca.components_).sum(axis=0)
    print(f"  weighted avg vector: {np.array2string(weighted_avg[:10], precision=4, suppress_small=True, separator=', ', max_line_width=120)}")


if __name__ == '__main__':
    main()
