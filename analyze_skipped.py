import argparse
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def plot_tsne(latents, actions_list, title, save_path, perplexity=30, learning_rate="auto"):
    print(f"Fitting t-SNE for {title}")
    
    if len(latents) > 50000:
        idx = np.random.choice(len(latents), 50000, replace=False)
        latents_sub = latents[idx]
        actions_sub = [actions_list[i] for i in idx]
    else:
        latents_sub = latents
        actions_sub = actions_list

    perp = min(perplexity, max(1, len(latents_sub) - 1))
    
    reducer = TSNE(n_components=2, perplexity=perp, learning_rate=learning_rate, random_state=42, init='pca')
    embedding = reducer.fit_transform(latents_sub)
    
    unique_actions = list(set(actions_sub))
    action_to_idx = {act: i for i, act in enumerate(unique_actions)}
    labels = [action_to_idx[act] for act in actions_sub]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=5, alpha=0.7)
    
    if len(unique_actions) <= 20:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_actions, title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.colorbar(scatter, label="Action ID")
        
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {save_path}\n")

def plot_umap_fn(latents, actions_list, title, save_path, n_neighbors=15, min_dist=0.1):
    print(f"Fitting UMAP for {title}")
    
    if len(latents) > 50000:
        idx = np.random.choice(len(latents), 50000, replace=False)
        latents_sub = latents[idx]
        actions_sub = [actions_list[i] for i in idx]
    else:
        latents_sub = latents
        actions_sub = actions_list

    nn = min(n_neighbors, max(2, len(latents_sub) - 1))

    reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(latents_sub)
    
    unique_actions = list(set(actions_sub))
    action_to_idx = {act: i for i, act in enumerate(unique_actions)}
    labels = [action_to_idx[act] for act in actions_sub]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=5, alpha=0.7)
    
    if len(unique_actions) <= 20:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_actions, title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.colorbar(scatter, label="Action ID")
        
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {save_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze skipped latency vectors")
    parser.add_argument("--dataset", type=str, required=True, choices=["adaworld", "olafworld"], help="Dataset to analyze")
    args = parser.parse_args()
    args.dump_dir = 2

    dump_dir = os.path.join("latent_actions_skipped", args.dataset)
    if not os.path.isdir(dump_dir):
        print(f"Directory {dump_dir} not found.")
        return

    files = glob.glob(os.path.join(dump_dir, "**", "latent_actions.pt"), recursive=True)
    print(f"Found {len(files)} latent action files in {dump_dir}.", flush=True)

    all_z_mu = []
    action_z_mu = {}
    game_action_z_mu = {}
    original_action_z_mu = {}
    game_action_z_mu_single = {}
    
    tsne_latents = []
    tsne_actions = []

    for f in files:
        parts = os.path.normpath(f).split(os.sep)
        game_name = parts[-4] if len(parts) >= 4 else args.dataset
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            continue
            
        if 'z_mu' not in data:
            continue

        z_mu = data['z_mu'] # (N, D)
        actions = data.get('actions', [])
        
        if len(actions) == len(z_mu):
            for i in range(len(actions)):
                act_dict = actions[i]
                if not isinstance(act_dict, dict) or 'desc' not in act_dict:
                    continue
                act = act_dict['desc']
                if act not in ['right', 'left', 'jump', 'none', 'crouch']:
                    continue
                    
                z_mu_i = z_mu[i].unsqueeze(0)
                
                single_act = (act,)
                
                all_z_mu.append(z_mu_i)
                tsne_latents.append(z_mu[i].numpy())
                tsne_actions.append(act)
                
                if single_act not in action_z_mu:
                    action_z_mu[single_act] = []
                action_z_mu[single_act].append(z_mu_i)

                if game_name not in game_action_z_mu:
                    game_action_z_mu[game_name] = {}
                if single_act not in game_action_z_mu[game_name]:
                    game_action_z_mu[game_name][single_act] = []
                game_action_z_mu[game_name][single_act].append(z_mu_i)

                if single_act not in original_action_z_mu:
                    original_action_z_mu[single_act] = []
                original_action_z_mu[single_act].append(z_mu_i)

                if game_name not in game_action_z_mu_single:
                    game_action_z_mu_single[game_name] = {}
                if single_act not in game_action_z_mu_single[game_name]:
                    game_action_z_mu_single[game_name][single_act] = []
                game_action_z_mu_single[game_name][single_act].append(z_mu_i)

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
    print(np.array2string(overall_var.numpy(), precision=4, suppress_small=True, separator=', '))



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
        print(np.array2string(var.numpy(), precision=4, suppress_small=True, separator=', '))


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
        print(f"  {action_label[act]}: {np.array2string(c, precision=4, suppress_small=True, separator=', ', max_line_width=120)}")

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
    for d in np.argsort(fisher_orig)[::-1]:
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
    for d in np.argsort(fisher_proj)[::-1]:
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
        print(f"  LD{i+1}: {np.array2string(vec, precision=4, suppress_small=True, separator=', ', max_line_width=120)}")

    # -------------------------------------------------------- 4. LINEAR PROBE --
    print("\n" + "="*80)
    print(f"  4. LINEAR PROBE (Multi-class: predict exact action tuple)")
    print("="*80)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42
    )
    
    probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    
    try:
        probe.fit(X_tr, y_tr)
        
        y_pred = probe.predict(X_te)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        acc = accuracy_score(y_te, y_pred)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_te, y_pred, average='micro', zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_te, y_pred, average='macro', zero_division=0
        )

        print(f"\n  Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Micro P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
        print(f"  Macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
        print(f"\n  Per-class metrics:")
        
        for i, act in enumerate(action_list):
            mask_te = (y_te == i).astype(int)
            pos_count = mask_te.sum()
            if pos_count > 0:
                mask_pred = (y_pred == i).astype(int)
                class_acc = accuracy_score(mask_te, mask_pred)
                class_p, class_r, class_f1, _ = precision_recall_fscore_support(
                    mask_te, mask_pred, average='binary', pos_label=1, zero_division=0
                )
                print(
                    f"    Class {i}: {action_label[act]} {str(act):15s} — "
                    f"Acc: {class_acc:.4f}, "
                    f"P/R/F1: {class_p:.4f}/{class_r:.4f}/{class_f1:.4f}  "
                    f"(n={pos_count})"
                )
    except Exception as e:
        print(f"  Could not fit linear probe: {e}")

    # --------------------------------------------------- 5. ENTANGLEMENT TEST --
    print("\n" + "="*80)
    print("  5. ENTANGLEMENT TEST")
    print("="*80)
    
    # We want to compare:
    # 1. Average distance between latents with SAME action, DIFFERENT game
    # 2. Average distance between latents with DIFFERENT action, SAME game
    
    same_act_diff_game_sum = 0.0
    same_act_diff_game_count = 0
    diff_act_same_game_sum = 0.0
    diff_act_same_game_count = 0
    
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
                
                # Pairwise distances
                dists = torch.cdist(z1, z2)
                same_act_diff_game_sum += dists.sum().item()
                same_act_diff_game_count += dists.numel()
                
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
                
                dists = torch.cdist(z1, z2)
                diff_act_same_game_sum += dists.sum().item()
                diff_act_same_game_count += dists.numel()

    mean_same_act = same_act_diff_game_sum / same_act_diff_game_count if same_act_diff_game_count > 0 else float('nan')
    mean_diff_act = diff_act_same_game_sum / diff_act_same_game_count if diff_act_same_game_count > 0 else float('nan')
    
    print(f"  Avg distance (Same Action, Diff Game)    : {mean_same_act:.4f}  (n_pairs={same_act_diff_game_count})")
    print(f"  Avg distance (Diff Action, Same Game)    : {mean_diff_act:.4f}  (n_pairs={diff_act_same_game_count})")
    if not np.isnan(mean_same_act) and not np.isnan(mean_diff_act):
        print(f"  Ratio (SameAct-DiffGame / DiffAct-SameGame) : {mean_same_act / mean_diff_act:.4f}")
        print("  (A lower ratio implies actions are well-aligned across games despite domain gap)")
    print("\n" + "="*80)
    print("  t-SNE and UMAP")
    print("="*80)
    if len(tsne_latents) > 0:
        tsne_latents_np = np.array(tsne_latents)
        tsne_dir = "tsne_plots"
        os.makedirs(tsne_dir, exist_ok=True)
        plot_tsne(tsne_latents_np, tsne_actions, f"t-SNE - Skipped {args.dataset}", os.path.join(tsne_dir, f"{args.dataset}_skipped_tsne.png"))
        umap_dir = "umap_plots"
        os.makedirs(umap_dir, exist_ok=True)
        plot_umap_fn(tsne_latents_np, tsne_actions, f"UMAP - Skipped {args.dataset}", os.path.join(umap_dir, f"{args.dataset}_skipped_umap.png"))



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
            print(f"        vector: {np.array2string(vec, precision=4, suppress_small=True, separator=', ', max_line_width=120)}")
    weighted_avg = (ratio[:, None] * pca.components_).sum(axis=0)
    print(f"  weighted avg vector: {np.array2string(weighted_avg, precision=4, suppress_small=True, separator=', ', max_line_width=120)}")



if __name__ == '__main__':
    main()
