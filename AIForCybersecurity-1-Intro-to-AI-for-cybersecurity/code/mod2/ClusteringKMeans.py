import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:,1], c=labels, cmap="viridis", edgecolors="k", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="red",marker="X",s=200, linewidths=2, label="centroids",)

plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("K-means clustering")
plt.legend()
out = Path(__file__).resolve().parent / "ClusteringKMeans.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
