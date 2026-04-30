# CKA Analysis Results & Insights

Based on the CKA and extended metrics evaluation between AdaWorld (Model 1) and OlafWorld (Model 2) latent action spaces, we observe several significant structural trends. The results span 50 matching games. Below is an ordered breakdown and commentary on the findings.

## 1. High-Level Alignment (Linear & RBF CKA)

We see a massive variance in alignment across different games, ranging from near-perfect geometric alignment (>0.97) down to complete misalignment (<0.01). 
* **RBF vs. Linear:** In almost all cases, RBF CKA is slightly higher than Linear CKA, but strictly follows the exact same trend. This implies that the transformation between the two spaces is largely linear/rigid—there are no complex non-linear manifolds in one model that the other model completely misses.

### Top-Tier Games (High Transferability)
Games like `retro_chuckrock-sms`, `retro_cliffhanger-nes`, `retro_buraifighter-nes`, and `retro_armadillo-nes` exhibit **Lin CKA > 0.93**. 
* **Insight:** For these environments, AdaWorld and OlafWorld organize their latent representations almost identically (up to rotation). Downstream policies or zero-shot transfer should be highly effective here.

### Bottom-Tier Games (Low Transferability)
Games like `retro_bubblebobble-nes`, `retro_castlevaniaiiidraculascurse-nes`, and `retro_balloonfight-nes` show **Lin CKA < 0.05**.
* **Insight:** The structure of the latent spaces completely diverges for these games.

## 2. Dimensionality Collapse (Dim 1 vs. Dim 2)

An interesting pattern emerges when looking at effective dimensionality (components explaining 90% of variance):
* **AdaWorld (Dim 1)** heavily compresses the latent space. Across the board, it relies on just **1 or 2 dimensions** to explain the vast majority of its variance. 
* **OlafWorld (Dim 2)** distributes its variance across slightly more dimensions, frequently using **3 to 6 dimensions**. 
* **Insight:** AdaWorld might be experiencing a stronger regularization or a bottleneck causing a tighter subspace collapse. For instance, in `retro_bubblebobble-nes`, Dim 1 is 1 while Dim 2 is 6. This severe mismatch in intrinsic dimensionality directly explains the near-zero CKA scores at the bottom of the list.

## 3. Procrustes MSE (Pointwise Mapping)

Even among highly aligned games (High CKA), the Procrustes Mean Squared Error (ProcMSE) shows variance:
* `retro_chacknpop-nes` has a high CKA (0.915) *and* an exceptionally low ProcMSE (0.066). This means the latent trajectories map almost perfectly point-per-point after rotation.
* Conversely, `retro_bonkerswaxup-sms` has high CKA (0.919) but higher ProcMSE (0.954). The geometric boundary of the spaces align, but the exact dynamic trajectory traces are scaled differently or are noisier.

## 4. Action Clustering (Silhouette & KNN limits)

We evaluated how well the continuous latents cluster purely by discrete action keys:
* **Silhouette Scores are uniformly negative (-0.01 to -0.23):** For both models, discrete actions do *not* form well-separated isotropic clusters. Heavy overlap exists. This is common in continuous latent spaces where temporal dynamics or state-context push action representations around, rather than maintaining static clusters.
* **KNN Accuracies (15% to 50%):** A simple 3-Nearest-Neighbor classifier struggles to predict the exact discrete action from the latents alone, with success rates typically hovering between 15% and 30% (though `retro_buckyohare-nes` reaches ~49%). 
* **Insight:** The latent actions are highly contextual. They encode "what happened in the transition" rather than acting as a simple 1-to-1 lookup table for the discrete gamepad button that was pressed.

## Summary & Next Steps

1. **Dimensionality Mismatch:** The biggest driving factor for low CKA seems to be AdaWorld compressing into 1D/2D lines/planes while OlafWorld utilizes 4D-6D volumes. Future debugging should look into AdaWorld's bottleneck/regularization to see why it drops dimensions.
2. **Transfer Learning:** Tests mapping AdaWorld to OlafWorld should start with `retro_chuckrock-sms` and `retro_chacknpop-nes` for sanity checks, given their brilliant geometric overlap.