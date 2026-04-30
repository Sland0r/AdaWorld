import argparse
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, hsv_to_rgb

def get_action_data(dump_dir):
    files = glob.glob(os.path.join(dump_dir, "*/*/*/latent_actions.pt"))
    print(f"Found {len(files)} latent action files in {dump_dir}.", flush=True)

    latents = []
    actions_list = []

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            continue
            
        if 'z_mu' not in data:
            continue

        z_mu = data['z_mu'] # (N, D)
        actions = data.get('actions', [])
        
        if len(actions) == len(z_mu):
            for i in range(len(actions)):
                if actions[i] is None:
                    continue
                act = tuple(actions[i])
                if not act:
                    continue
                    
                latents.append(z_mu[i].numpy())
                actions_list.append(act)
                
        # Subsample to avoid huge arrays
        if len(latents) > 100000:
            break
            
    if not latents:
        return None, None
        
    latents = np.array(latents)
    
    # Subsample if needed
    if len(latents) > 50000:
        idx = np.random.choice(len(latents), 50000, replace=False)
        latents = latents[idx]
        actions_list = [actions_list[i] for i in idx]

    return latents, actions_list

def plot_tsne(latents, actions_list, title, save_path, perplexity=30, learning_rate="auto"):
    print(f"Fitting t-SNE for {title} (perplexity={perplexity}, learning_rate={learning_rate})", flush=True)
    reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, init='pca')
    embedding = reducer.fit_transform(latents)
    
    unique_actions = list(set(actions_list))
    action_to_idx = {act: i for i, act in enumerate(unique_actions)}
    labels = [action_to_idx[act] for act in actions_list]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=1, alpha=0.5)
    
    # We won't add a legend if there are too many actions, but let's try to add a colorbar or legend if < 20
    if len(unique_actions) <= 20:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_actions, title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.colorbar(scatter, label="Action ID")
        
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {save_path}", flush=True)

def main():
    out_dir = "tsne_plots"
    os.makedirs(out_dir, exist_ok=True)
    
    perplexity_list = [5, 30, 50]
    learning_rate_list = ['auto', 200, 500]
    
    for name in ['adaworld', 'olafworld']:
        dump_dir = os.path.join('latent_actions_dump', name)
        if not os.path.isdir(dump_dir):
            print(f"{dump_dir} not found.")
            continue
            
        latents, actions_list = get_action_data(dump_dir)
        if latents is not None:
            for perp in perplexity_list:
                for lr in learning_rate_list:
                    # format learning rate for filename
                    lr_str = str(lr)
                    filename = f"{name}_tsne_perp{perp}_lr{lr_str}.png"
                    save_path = os.path.join(out_dir, filename)
                    title = f"t-SNE - {name} (perp={perp}, lr={lr_str})"
                    plot_tsne(latents, actions_list, title, save_path, perplexity=perp, learning_rate=lr)

if __name__ == '__main__':
    main()