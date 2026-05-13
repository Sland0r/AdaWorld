# Latent Action Space Analysis: AdaWorld vs OlafWorld

## 1. Overview

This report analyses the latent action representations ($z_\mu$) extracted by two world-model action encoders — **AdaWorld** and **OlafWorld** — across two experimental settings:

| Setting | Source | Actions | AdaWorld Samples | OlafWorld Samples |
|---|---|---|---|---|
| **Retro** (`dump_1`) | 50 retro NES/SMS games | 6 abstract (numeric indices 0–5) | 279,460 | 279,460 |
| **P2P** (`dump_2`) | Open-P2P game subset | 11 keyboard keys (w, a, s, d, arrows, shift, f, space, none) | 2,413,066 | ~1,300,000 |

Both encoders produce 32-dimensional latent vectors. The central question is: **does the latent action space meaningfully encode which action was taken, or is it dominated by other factors (game identity, visual state)?**

---

## 2. Variance Structure

### 2.1 Overall Variance per Dimension

> [!IMPORTANT]
> AdaWorld concentrates all variance into 4 dimensions; OlafWorld spreads variance across ~20 dimensions.

**AdaWorld (P2P, 2.4M samples):**
Only 4 dimensions carry meaningful variance: dim 17 (0.864), dim 11 (0.195), dim 15 (0.141), dim 7 (0.094). The other 28 dimensions have variance ≈ 0.0001 — effectively dead.

**OlafWorld (P2P, 1.3M samples):**
Variance is distributed much more broadly. Top dimensions: dim 22 (0.744), dim 25 (0.415), dim 13 (0.210), dim 17 (0.170), dim 21 (0.102), dim 23 (0.089), plus ~15 more dimensions with variance > 0.01.

**Retro setting (280K samples each):**
The same pattern holds — AdaWorld uses 3 effective dimensions (dim 17=0.632, dim 7=0.197, dim 11=0.173), while OlafWorld uses ~15+ (dim 22=0.512, dim 25=0.373, dim 21=0.251, etc.).

### 2.2 Commentary

AdaWorld's latent space is **extremely collapsed**: a 32-dimensional code effectively operates in a 3–4 dimensional subspace. OlafWorld uses the capacity more fully, which in principle allows for richer, more disentangled representations — but as we will see, neither model encodes action identity well.

---

## 3. PCA Analysis

### 3.1 Global PCA

| | PC1 | PC2 | PC3 | PC4 | Top-3 CumVar | Top-4 CumVar |
|---|---|---|---|---|---|---|
| **AdaWorld P2P** | 66.8% | 15.4% | 11.0% | 6.8% | 93.1% | **99.96%** |
| **OlafWorld P2P** | 40.0% | 20.8% | 11.5% | 8.7% | 72.3% | 81.0% |
| **AdaWorld Retro** | 62.8% | 19.5% | 16.6% | 1.0% | 98.9% | **99.96%** |
| **OlafWorld Retro** | 35.3% | 20.2% | 16.1% | 13.4% | 71.6% | 85.0% |

> [!NOTE]
> **AdaWorld** reaches 99.96% explained variance with just 4 PCs in both settings. **OlafWorld** requires 10–15 PCs for similar coverage, confirming a fundamentally higher-dimensional representation.

### 3.2 PC1 Direction

- **AdaWorld PC1** loads almost exclusively on dim 17 (weight 0.998) — the single dominant dimension is "the embedding".
- **OlafWorld PC1** loads primarily on dim 22 (0.928) and dim 23 (0.220), spreading across at least two raw dimensions.

### 3.3 Per-Action PCA

Within each action class, the PCA structure is remarkably similar to the global PCA for both models. This means actions do *not* occupy distinct, lower-dimensional submanifolds — they all live in essentially the same subspace with slightly shifted means.

---

## 4. Action Distribution

Both models were evaluated on the same P2P game set. The action distributions are similar:

| Action | AdaWorld % | OlafWorld % |
|---|---|---|
| `w` (forward) | 40.2% | 38.5% |
| `d` (right) | 12.8% | 11.6% |
| `none` | 10.9% | 12.7% |
| `a` (left) | 10.4% | 9.8% |
| `UpArrow` | 4.1% | 8.2% |
| `LeftArrow` | 5.9% | 5.9% |
| `RightArrow` | 5.8% | 5.9% |
| `s` (down) | 5.0% | 5.0% |
| `LeftShift` | 4.4% | 1.7% |
| `f` | 0.5% | 0.6% |
| `Space` | 0.1% | 0.1% |

The heavy imbalance toward `w` reflects the nature of the P2P games (side-scrollers where forward movement dominates).

---

## 5. Fisher Criterion

The Fisher criterion measures how well each dimension (or PC) separates action classes: $F = \sigma^2_{\text{between}} / \sigma^2_{\text{within}}$.

### 5.1 Original Dimensions

| Metric | AdaWorld (best dim) | OlafWorld (best dim) |
|---|---|---|
| **Top Fisher score** | 0.230 (dim 11) | 0.290 (dim 25) |
| **2nd Fisher score** | 0.162 (dim 23) | 0.245 (dim 16) |
| **3rd Fisher score** | 0.159 (dim 17) | 0.220 (dim 8) |

> [!WARNING]
> All Fisher scores are far below 1.0, meaning within-class variance greatly exceeds between-class variance on every dimension. Actions are not well-separated in the latent space of *either* model.

For reference, a well-separated space would show Fisher scores >> 1 on at least some dimensions. The highest we observe is 0.29 (OlafWorld dim 25), indicating only ≈23% as much between-class spread as within-class spread.

### 5.2 Principal Components

| | PC with best Fisher | Fisher score | Between var | Within var |
|---|---|---|---|---|
| **AdaWorld** | PC2 | 0.242 | 0.049 | 0.204 |
| **OlafWorld** | PC2 | 0.303 | 0.130 | 0.430 |

OlafWorld shows modestly better Fisher separation on PC2. For AdaWorld, PC1 (the dominant axis, carrying 66.8% of variance) has Fisher = 0.159 — meaning the axis of greatest overall variation is only weakly associated with action identity.

### 5.3 Retro Setting

In the retro setting (6 balanced action classes), Fisher scores collapse dramatically:

| | Best Fisher | Dimension |
|---|---|---|
| **AdaWorld Retro** | 0.005 (dim 11) | Near-zero on all dims |
| **OlafWorld Retro** | 0.004 (dim 25) | Near-zero on all dims |

The latent space is essentially action-agnostic when actions are abstract numeric indices without clear visual correlates.

---

## 6. Linear Discriminant Analysis (LDA)

LDA finds the optimal linear projections for class separation. With 11 actions, there are 10 discriminant axes (LDs).

### 6.1 P2P Setting

| LD | AdaWorld VarRatio | AdaWorld CumRatio | OlafWorld VarRatio | OlafWorld CumRatio |
|---|---|---|---|---|
| 1 | 0.321 | 0.321 | 0.384 | 0.384 |
| 2 | 0.271 | 0.592 | 0.198 | 0.582 |
| 3 | 0.198 | 0.790 | 0.182 | 0.764 |
| 4 | 0.090 | 0.880 | 0.090 | 0.854 |
| 5 | 0.045 | 0.925 | 0.080 | 0.934 |

Both models show that 3 LDs capture ~79% (AdaWorld) to ~76% (OlafWorld) of the between-class variance. This suggests that while 11 actions exist, the latent space only supports ~3–4 meaningful discrimination axes.

### 6.2 LD Direction Vectors

The LDA directions for both models are broadly distributed across all 32 raw dimensions (no single dimension dominates), indicating that the weak action signal is spread and mixed with game/visual-state information.

### 6.3 Retro Setting

| LD | AdaWorld VarRatio | OlafWorld VarRatio |
|---|---|---|
| 1 | 0.481 | 0.552 |
| 2 | 0.236 | 0.260 |
| 3 | 0.172 | 0.115 |
| 4 | 0.083 | 0.058 |
| 5 | 0.028 | 0.015 |

With only 6 action classes, LD1 concentrates more variance. OlafWorld's LD1 at 55.2% suggests it picks up slightly more discriminative structure than AdaWorld (48.1%).

---

## 7. Linear Probe (Action Prediction)

### 7.1 P2P Setting — Multi-label Logistic Regression

> [!IMPORTANT]
> Key result: Neither model's latent space supports accurate action prediction. OlafWorld is moderately better.

| Metric | AdaWorld | OlafWorld |
|---|---|---|
| **Exact match accuracy** | 31.4% | 43.1% |
| **Hamming loss** | 10.5% | 8.8% |
| **Micro F1** | 0.542 | 0.606 |
| **Macro F1** | 0.270 | 0.391 |

**Per-key breakdown (P2P, F1 scores):**

| Key | AdaWorld F1 | OlafWorld F1 | Comment |
|---|---|---|---|
| `w` | 0.790 | 0.767 | Both models detect the dominant action reasonably |
| `d` | 0.279 | 0.440 | OlafWorld 60% better |
| `none` | 0.241 | 0.678 | OlafWorld nearly 3× better |
| `a` | 0.220 | 0.384 | OlafWorld 75% better |
| `LeftArrow` | 0.453 | 0.599 | OlafWorld 32% better |
| `RightArrow` | 0.534 | 0.590 | Similar |
| `s` | 0.176 | 0.312 | OlafWorld 77% better |
| `LeftShift` | 0.002 | 0.018 | Both essentially zero |
| `UpArrow` | 0.001 | 0.121 | OlafWorld captures some signal |
| `f` | 0.000 | 0.000 | Both fail — too rare |

### 7.2 Retro Setting — Multiclass Logistic Regression

| | AdaWorld | OlafWorld |
|---|---|---|
| **Test accuracy** | 19.2% | 21.1% |
| **Chance level** | 16.7% | 16.7% |
| **Lift** | 1.15× | 1.26× |

Both barely beat chance. The latent space encodes almost no action information in the retro setting.

---

## 8. Neural Network Probe (MLP, Retro Setting)

Deeper probes with 1–3 hidden layers were trained on the retro-dump latent actions (6 classes, 244K–432K train samples).

| Model | Layers | Action Acc | Game Acc |
|---|---|---|---|
| **OlafWorld** | 1 hidden | 29.5% | 99.2% |
| **OlafWorld** | 2 hidden | 37.1% | 99.3% |
| **AdaWorld** | 1 hidden | 27.5% | 99.1% |
| **AdaWorld** | 2 hidden | 32.2% | 99.2% |
| **OlafWorld** | 3 hidden | 25.8% | 99.3% |
| **AdaWorld** | 3 hidden | 23.2% | 99.3% |

> [!CAUTION]
> **Game identity is almost perfectly decodable (99%+) while action identity barely exceeds chance (16.7%).**
> This is the central finding: the latent action space is dominated by game/visual-context encoding, not action semantics.

Best action accuracy is 37.1% (OlafWorld, 2 hidden layers) — only 2.2× above chance for a 6-class problem. Deeper networks (3 layers) begin overfitting and perform worse.

---

## 9. Entanglement Test

This test directly measures whether action information is disentangled from game identity by comparing:
- **Same Action, Different Game** distance: should be *small* if actions are represented game-independently
- **Different Action, Same Game** distance: should be *large* if actions are well-separated

| Setting | Model | Same-Act-Diff-Game | Diff-Act-Same-Game | Ratio |
|---|---|---|---|---|
| **P2P** | AdaWorld | 1.409 | 1.397 | **1.009** |
| **P2P** | OlafWorld | 2.022 | 2.062 | **0.981** |
| **Retro** | AdaWorld | 0.985 | 0.675 | **1.460** |
| **Retro** | OlafWorld | 1.620 | 1.147 | **1.412** |

> [!NOTE]
> A ratio < 1 indicates good action alignment across games. A ratio >> 1 means game identity dominates.

**P2P setting:** Both models achieve a ratio near 1.0, which seems promising — but in context, this reflects that both distances are nearly equal and large, meaning **the latent space doesn't strongly separate either factor**. OlafWorld has a ratio slightly below 1 (0.981), hinting at marginal cross-game action alignment.

**Retro setting:** Ratios of ~1.4–1.5 confirm that switching games moves the latent representation far more than switching actions. The latent space is fundamentally organised by visual game context.

---

## 10. Centroid Distances

### 10.1 P2P Setting

**AdaWorld** centroid L2 distances range from 0.07 to 1.62. The largest separations occur for `LeftArrow` and `RightArrow` (≈ 1.0–1.6 from other actions), while `w`, `UpArrow`, `LeftShift`, and `f` cluster tightly (mutual distances 0.07–0.22). The `none` action is well-separated (0.44–0.90 from others), likely because it corresponds to static frames.

**OlafWorld** centroids are more spread overall (distances 0.12–1.90). `LeftArrow` and `RightArrow` are the most separated pair (1.86), and they also diverge from other actions (1.0–1.8). This suggests OlafWorld's encoder captures directional movement more distinctly.

### 10.2 Retro Setting

Both models show minimal centroid separation (max L2 distance: 0.10 AdaWorld, 0.14 OlafWorld). The 6 abstract retro actions produce nearly indistinguishable centroids.

---

## 11. CKA Similarity (AdaWorld vs OlafWorld, Retro)

Centred Kernel Alignment (CKA) measures the similarity of the latent representations produced by the two models for the *same input frames*.

**Overall Linear CKA: 0.697**

This moderate similarity (neither identical nor independent) suggests the two encoders extract overlapping but non-identical features from the same visual inputs.

**Per-game CKA distribution:**

| CKA Range | Count | Examples |
|---|---|---|
| 0.90–1.00 | 6 games | chuckrock (0.97), cliffhanger (0.97), buraifighter (0.95) |
| 0.70–0.90 | 8 games | bramstokersdracula (0.87), cityconnection (0.87), alfredchicken (0.85) |
| 0.40–0.70 | 12 games | baddudes (0.67), batmanreturns (0.63), bananaprince (0.54) |
| 0.10–0.40 | 12 games | captainamerica (0.31), athena (0.31), castlevania (0.22) |
| 0.00–0.10 | 12 games | balloonfight (0.03), bubblebobble (0.01), castlevaniaiii (0.01) |

The wide range (0.007 to 0.973) shows that model agreement is highly game-dependent. Games with simpler, more horizontally-scrolling mechanics tend to produce higher CKA (the encoders agree on what matters), while games with unique mechanics (e.g., Bubble Bobble's vertical movement, Balloon Fight's floating) produce near-zero CKA.

**Extended per-game metrics** (from log `cka_22358396`) additionally report KNN accuracy, Silhouette scores (all negative, confirming no natural clustering by action), and effective dimensionality (AdaWorld consistently 1–3, OlafWorld 1–6).

---

## 12. Summary of Key Findings

| Metric | AdaWorld | OlafWorld | Winner |
|---|---|---|---|
| Effective dimensionality | 3–4 / 32 | 10–20 / 32 | OlafWorld |
| Top-3 PCA cumulative variance | 93.1% | 72.3% | — (AdaWorld more compressed) |
| Best Fisher criterion | 0.230 | 0.290 | OlafWorld (+26%) |
| Linear probe exact match (P2P) | 31.4% | 43.1% | OlafWorld (+37%) |
| Linear probe micro F1 (P2P) | 0.542 | 0.606 | OlafWorld (+12%) |
| Linear probe macro F1 (P2P) | 0.270 | 0.391 | OlafWorld (+45%) |
| MLP action acc (retro, best) | 32.2% | 37.1% | OlafWorld (+15%) |
| MLP game acc (retro) | 99.3% | 99.3% | Tied |
| Entanglement ratio (P2P) | 1.009 | 0.981 | OlafWorld (marginally) |
| Entanglement ratio (retro) | 1.460 | 1.412 | OlafWorld (marginally) |

---

## 13. Conclusions

### The latent action space is not about actions

> [!CAUTION]
> **Neither AdaWorld nor OlafWorld learns a latent action space that meaningfully encodes which action was taken.** The representation is overwhelmingly dominated by game identity and visual state information.

Evidence:
1. **Game identity is perfectly decodable** (99%+ MLP accuracy), while **action identity barely exceeds chance** (19–37% vs 16.7% chance for 6 classes; 31–43% exact match for 11 classes).
2. **Fisher scores are uniformly below 0.3** on all dimensions and PCs — within-class variance (variation across frames with the *same* action label) dwarfs between-class variance.
3. **Entanglement ratios ≈ 1.0–1.5**: the latent space is more sensitive to game switches than action switches.
4. **Centroids cluster by game, not by action**: in the retro setting, all 6 action centroids are within L2 distance 0.10 of each other, while game-switch distances are orders of magnitude larger.

### AdaWorld has a severely collapsed representation

AdaWorld's 32-dimensional code is effectively 3–4 dimensional. Nearly all variance concentrates on dimensions 17 (z_mu), 11, 15, and 7 — with 28 dimensions at near-zero variance. This extreme compression may reflect aggressive regularisation (strong KL penalty) in the VAE objective, or simply that the encoder learns a very low-rank mapping from frames to latents.

### OlafWorld is moderately better but still far from usable

OlafWorld uses 10–20 active dimensions and achieves consistently better (though still weak) action-discriminative metrics: +26% higher Fisher scores, +37% higher linear probe exact match, +45% higher macro F1. The improvement is consistent across every evaluation method, suggesting OlafWorld's encoder captures more action-relevant features — but the absolute level of action encoding remains poor.

### What the latent space *does* encode

The representation is best described as a **compressed visual state summary**:
- It captures game identity almost perfectly
- It encodes broad visual dynamics (which game, which scene)
- It partially captures directional movement (`LeftArrow` and `RightArrow` are the most separated actions in both models)
- It fails to encode abstract action semantics

### CKA reveals game-dependent agreement

The two encoders agree most on games with canonical side-scrolling mechanics (CKA > 0.9) and disagree most on games with unusual movement patterns (CKA < 0.1). This suggests that when the visual dynamics are "typical," both architectures converge on similar representations — but their inductive biases lead to different encodings for atypical visual patterns.

### Implications

The near-total absence of action information in the latent space raises questions about:
1. Whether the action encoder architecture (VAE bottleneck) is appropriate for disentangling actions from visual state
2. Whether the training objective incentivises action encoding at all, or whether the decoder can reconstruct frames purely from visual-state information
3. Whether explicit action-prediction losses or contrastive objectives would be needed to inject action semantics into the bottleneck
