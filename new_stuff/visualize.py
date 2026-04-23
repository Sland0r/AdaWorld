import sys
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import the main function from the extraction script
from extract_latent_actions import main as extract_main

def main():
    print("Starting extraction of latent actions...")
    
    # We call extract_main(), which will parse sys.argv.
    # Therefore, you can run visualize.py with the exact same arguments 
    # as extract_latent_actions.py.
    results = extract_main()

    if not results:
        print("No results returned. Exiting.")
        return

    # Extract z_mu from the results
    all_mu = torch.stack([r for r in results]).numpy()

    print(f"Applying PCA on {all_mu.shape[0]} samples of dimension {all_mu.shape[1]}...")
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_mu)

    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    # Plot a line connecting the points to show temporal progression
    # plt.plot(pca_result[:, 0], pca_result[:, 1], linestyle='-', alpha=0.4, color='gray')
    
    # Plot the points themselves, colored by their temporal index
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=range(len(pca_result)), cmap='viridis', s=50)
    
    plt.colorbar(scatter, label='Time (Frame Pair Index)')
    plt.title('2D PCA of Latent Actions (z_mu)')
    
    # Show explained variance on axes
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True)
    
    out_file = 'pca_visualization.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_file}")

if __name__ == "__main__":
    main()
