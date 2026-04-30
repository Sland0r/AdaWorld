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
    parser.add_argument("--dump-dir", type=str, default="adaworld", help="Directory containing latent actions")
    args = parser.parse_args()

    dump_dir = 'latent_actions_dump/' + args.dump_dir 
    files = glob.glob(os.path.join(dump_dir, "*/*/*/latent_actions.pt"))
    print(f"Found {len(files)} latent action files.", flush=True)

    all_z_mu = []
    game_z_mu = {}
    action_z_mu = {}
    game_action_z_mu = {}

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            continue
            
        parts = f.split('/')
        game_name = parts[-4] # 'latent_actions_dump/adaworld/game_name/seed/episode/latent_actions.pt'
        
        if 'z_mu' not in data:
            continue

        z_mu = data['z_mu'] # (N, D)
        actions = data.get('actions', [])
        
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
                act = tuple(actions[i])
                if not act:
                    continue
                    
                z_mu_i = z_mu[i].unsqueeze(0) # (1, D)
                
                if act not in action_z_mu:
                    action_z_mu[act] = []
                action_z_mu[act].append(z_mu_i)
                
                if act not in game_action_z_mu[game_name]:
                    game_action_z_mu[game_name][act] = []
                game_action_z_mu[game_name][act].append(z_mu_i)

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
    print("  VARIANCE PER VIDEOGAME")
    print("="*80)
    for game in sorted(game_z_mu.keys()):
        z_mu_cat = torch.cat(game_z_mu[game], dim=0)
        var = torch.var(z_mu_cat, dim=0, unbiased=False) if z_mu_cat.shape[0] > 1 else torch.zeros(z_mu_cat.shape[1])
        print(f"\nGame: {game} (Samples: {z_mu_cat.shape[0]})")
        print(np.array2string(var.numpy(), precision=4, suppress_small=True, separator=', '))

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
    print("  PCA — PER VIDEOGAME")
    print("="*80)
    for game in sorted(game_z_mu.keys()):
        z_np = torch.cat(game_z_mu[game], dim=0).numpy()
        fit_and_print_pca(f"Game: {game}", z_np, top_k=3)

    print("\n" + "="*80)
    print("  PCA — PER ACTION")
    print("="*80)
    for act in sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True):
        fit_and_print_pca(f"Action {act}", action_cats[act].numpy(), top_k=3)

    # ---- build labeled dataset shared by the four analyses below ----
    action_list = sorted(action_cats.keys(), key=lambda a: action_cats[a].shape[0], reverse=True)
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
    print("  2. FISHER CRITERION (between / within class variance per dimension)")
    print("="*80)

    D = X_labeled.shape[1]
    global_mean  = X_labeled.mean(axis=0)
    between_var  = np.zeros(D)
    within_var   = np.zeros(D)
    n_total      = X_labeled.shape[0]

    for act in action_list:
        X_act = action_cats[act].numpy()
        n_k   = X_act.shape[0]
        mu_k  = X_act.mean(axis=0)
        between_var += n_k * (mu_k - global_mean) ** 2
        within_var  += ((X_act - mu_k) ** 2).sum(axis=0)

    between_var /= n_total
    within_var  /= n_total
    fisher       = between_var / (within_var + 1e-10)

    print(f"\n  {'Dim':>4}  {'Fisher':>10}  {'Between':>10}  {'Within':>10}")
    for d in np.argsort(fisher)[::-1]:
        print(f"  {d:>4}  {fisher[d]:>10.6f}  {between_var[d]:>10.6f}  {within_var[d]:>10.6f}")

    # ----------------------------------------------------------------- 3. LDA --
    print("\n" + "="*80)
    print("  3. LDA (Linear Discriminant Analysis)")
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
    print("  4. LINEAR PROBE (logistic regression  z_mu → action)")
    print("="*80)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    probe.fit(X_tr, y_tr)

    acc    = probe.score(X_te, y_te)
    chance = np.bincount(y_labeled).max() / len(y_labeled)
    y_pred = probe.predict(X_te)

    print(f"\n  Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Chance level  : {chance:.4f}  ({chance*100:.1f}%)")
    print(f"  Lift          : {acc/chance:.2f}x")
    print("\n  Per-action accuracy:")
    for i, act in enumerate(action_list):
        mask    = y_te == i
        cls_acc = (y_pred[mask] == i).mean() if mask.sum() > 0 else float('nan')
        print(f"    {action_label[act]} {act}: {cls_acc:.4f}  (n={mask.sum()})")


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
