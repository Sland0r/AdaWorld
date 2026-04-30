import argparse
import glob
import os
import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def linear_cka(X, Y):
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    dot_prod = np.linalg.norm(X_c.T @ Y_c, ord='fro') ** 2
    norm_X = np.linalg.norm(X_c.T @ X_c, ord='fro')
    norm_Y = np.linalg.norm(Y_c.T @ Y_c, ord='fro')
    
    if norm_X == 0 or norm_Y == 0:
        return 0.0
    return dot_prod / (norm_X * norm_Y)

def rbf_cka(X, Y, gamma=None, max_samples=2000):
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        
    K = rbf_kernel(X, gamma=gamma)
    L = rbf_kernel(Y, gamma=gamma)
    
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    num = np.sum(Kc * Lc)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    
    if den == 0:
        return 0.0
    return num / den

def procrustes_error(X, Y):
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    # Ensure dimensions match for orthogonal procrustes
    if X_c.shape[1] < Y_c.shape[1]:
        X_c = np.pad(X_c, ((0,0), (0, Y_c.shape[1] - X_c.shape[1])))
    elif Y_c.shape[1] < X_c.shape[1]:
        Y_c = np.pad(Y_c, ((0,0), (0, X_c.shape[1] - Y_c.shape[1])))
        
    R, scale = orthogonal_procrustes(X_c, Y_c)
    Y_pred = X_c @ R
    
    mse = np.mean(np.sum((Y_c - Y_pred)**2, axis=1))
    return mse

def effective_dimensionality(X, variance_thresh=0.90):
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= variance_thresh) + 1
    return d

def evaluate_action_clustering(X, Y, actions, max_samples=5000):
    # Ensure dimensions match the actions array
    if len(actions) == 0:
        return -1, -1, -1, -1
        
    # We only care about non-None actions if any
    valid_idx = [i for i, a in enumerate(actions) if a is not None]
    if len(valid_idx) == 0:
        return -1, -1, -1, -1
        
    X = X[valid_idx]
    Y = Y[valid_idx]
    act = np.array([str(actions[i]) for i in valid_idx])
    
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        act = act[idx]
        
    # Silhouette score
    try:
        if len(np.unique(act)) > 1:
            sil_X = silhouette_score(X, act)
            sil_Y = silhouette_score(Y, act)
        else:
            sil_X, sil_Y = 0.0, 0.0
    except:
        sil_X, sil_Y = 0.0, 0.0
        
    # Cross transfer (Linear probe / KNN)
    try:
        if len(np.unique(act)) > 1:
            X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(X, act, Y, test_size=0.2)
            knn_X = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
            knn_Y = KNeighborsClassifier(n_neighbors=3).fit(Y_train, y_train)
            acc_X = knn_X.score(X_test, y_test)
            acc_Y = knn_Y.score(Y_test, y_test)
        else:
            acc_X, acc_Y = 1.0, 1.0
    except:
        acc_X, acc_Y = 0.0, 0.0

    return sil_X, sil_Y, acc_X, acc_Y

def main():
    parser = argparse.ArgumentParser(description="Compute CKA and extended metrics between two latent action sets")
    parser.add_argument("--dir1", type=str, default="adaworld")
    parser.add_argument("--dir2", type=str, default="olafworld")
    args = parser.parse_args()

    base_dir1 = os.path.join('latent_actions_dump', args.dir1)
    base_dir2 = os.path.join('latent_actions_dump', args.dir2)
    
    files1 = glob.glob(os.path.join(base_dir1, "*/*/*/latent_actions.pt"))
    print(f"Found {len(files1)} files in {args.dir1}", flush=True)

    cka_per_game = {}

    for f1 in files1:
        rel_path = os.path.relpath(f1, base_dir1)
        f2 = os.path.join(base_dir2, rel_path)

        if not os.path.exists(f2):
            continue

        try:
            data1 = torch.load(f1, map_location='cpu')
            data2 = torch.load(f2, map_location='cpu')
        except Exception as e:
            continue
            
        if 'z_mu' not in data1 or 'z_mu' not in data2:
            continue

        z1 = data1['z_mu'].numpy() 
        z2 = data2['z_mu'].numpy()
        
        a1 = data1.get('actions', [])
        a2 = data2.get('actions', [])

        min_len = min(z1.shape[0], z2.shape[0])
        z1 = z1[:min_len]
        z2 = z2[:min_len]
        actions = a1[:min_len] if len(a1) >= min_len else a2[:min_len] if len(a2) >= min_len else []

        parts = rel_path.split(os.sep)
        game_name = parts[0]
        
        if game_name not in cka_per_game:
            cka_per_game[game_name] = {'X': [], 'Y': [], 'A': []}
            
        cka_per_game[game_name]['X'].append(z1)
        cka_per_game[game_name]['Y'].append(z2)
        cka_per_game[game_name]['A'].extend(actions)

    print("\n" + "="*125)
    print(f"{'Game':<30} | {'Lin CKA':<7} | {'RBF CKA':<7} | {'ProcMSE':<7} | {'Dim 1':<5} | {'Dim 2':<5} | {'Silh 1':<6} | {'Silh 2':<6} | {'KNN 1':<5} | {'KNN 2':<5}")
    print("="*125)
    
    game_results = []
    for game in cka_per_game.keys():
        X_g = np.concatenate(cka_per_game[game]['X'], axis=0)
        Y_g = np.concatenate(cka_per_game[game]['Y'], axis=0)
        A_g = cka_per_game[game]['A']
        
        # Calculate point 3 metrics
        dim_x = effective_dimensionality(X_g)
        dim_y = effective_dimensionality(Y_g)
        
        # Core alignment metrics
        val_lin = linear_cka(X_g, Y_g)
        val_rbf = rbf_cka(X_g, Y_g)
        proc_err = procrustes_error(X_g, Y_g)
        
        # Clustering metrics
        sil_x, sil_y, acc_x, acc_y = evaluate_action_clustering(X_g, Y_g, A_g)
        
        game_results.append((game, X_g.shape[0], val_lin, val_rbf, proc_err, dim_x, dim_y, sil_x, sil_y, acc_x, acc_y))
        
    game_results.sort(key=lambda x: x[2], reverse=True)
    
    for row in game_results:
        game, samples, lin, rbf, proc, dx, dy, sx, sy, ax, ay = row
        print(f"{game:<30} | {lin:.5f} | {rbf:.5f} | {proc:.5f} | {dx:<5d} | {dy:<5d} | {sx:>6.3f} | {sy:>6.3f} | {ax:>5.3f} | {ay:>5.3f}")

if __name__ == '__main__':
    main()
