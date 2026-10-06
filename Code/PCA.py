import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = REPO_ROOT / "Data"

def get_data(data_path):
    df = pd.read_csv(data_path)
    data_cols = sorted(
        [c for c in df.columns if c.startswith('d') and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    D = len(data_cols)
    print(f"D = {D}, columns: {data_cols[0]} to {data_cols[-1]}")
    return df, data_cols

# df, gene_names = get_data(DATA_DIR / "citeseq_22.csv")
# data = df[gene_names].values.astype('float32')

# n_components = min(100, data.shape[0], data.shape[1])
# pca = PCA(n_components=n_components)
# pca.fit(data)

# eigenvalues = pca.explained_variance_
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:n_components])

# fig, axes = plt.subplots(2, 1, figsize=(7, 11))

# axes[0].semilogy(np.arange(1, n_components + 1), eigenvalues, marker='o', markersize=3)
# axes[0].set_xlabel('Principal Component')
# axes[0].set_ylabel('Eigenvalue')

# axes[1].plot(np.arange(1, n_components + 1), cumulative_variance, marker='o', markersize=3)
# axes[1].set_xlabel('Number of Components')
# axes[1].set_ylabel('Cumulative Explained Variance')
# axes[1].axhline(0.95, color='red', linestyle='--', linewidth=1)

# plt.tight_layout()
# fig.savefig("Figures/PCA_d=22050.pdf", bbox_inches="tight")
# plt.show()

df, gene_names = get_data(DATA_DIR / "citeseq.csv")
data = df[gene_names].values.astype('float32')

n_components = min(100, data.shape[0], data.shape[1])
pca = PCA(n_components=n_components)
pca.fit(data)

eigenvalues = pca.explained_variance_
cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:n_components])

fig, axes = plt.subplots(2, 1, figsize=(7, 11))

axes[0].semilogy(np.arange(1, n_components + 1), eigenvalues, marker='o', markersize=3)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Eigenvalue')

axes[1].plot(np.arange(1, n_components + 1), cumulative_variance, marker='o', markersize=3)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].axhline(0.95, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
fig.savefig("Figures/PCA_d=100.pdf", bbox_inches="tight")
plt.show()

