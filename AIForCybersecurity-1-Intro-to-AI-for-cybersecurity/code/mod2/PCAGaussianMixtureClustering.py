import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

X, y_true = make_blobs(n_samples=300, n_features=8, centers=8, random_state=42)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X_reduced)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", edgecolors="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM clusters on PCA reduced data")
out = Path(__file__).resolve().parent / "PCAGaussianMixtureClustering.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
