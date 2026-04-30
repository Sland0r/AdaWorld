import numpy as np
from scipy.stats import pearsonr, spearmanr
import re

with open('logs/cka_22358396.out', 'r') as f:
    lines = [l.strip() for l in f.readlines() if 'retro_' in l]

cka = []
dim1 = []
dim2 = []

for l in lines:
    parts = [p.strip() for p in l.split('|')]
    cka.append(float(parts[1]))
    dim1.append(float(parts[4]))
    dim2.append(float(parts[5]))

cka = np.array(cka)
dim1 = np.array(dim1)
dim2 = np.array(dim2)
diff = dim2 - dim1
ratio = dim2 / np.maximum(dim1, 1)

print(f"Pearson Correlation (CKA vs AdaWorld Dim 1): {pearsonr(cka, dim1)[0]:.3f}")
print(f"Pearson Correlation (CKA vs OlafWorld Dim 2): {pearsonr(cka, dim2)[0]:.3f}")
print(f"Pearson Correlation (CKA vs Diff [Dim2 - Dim1]): {pearsonr(cka, diff)[0]:.3f}")
print(f"Pearson Correlation (CKA vs Ratio [Dim2 / Dim1]): {pearsonr(cka, ratio)[0]:.3f}")
