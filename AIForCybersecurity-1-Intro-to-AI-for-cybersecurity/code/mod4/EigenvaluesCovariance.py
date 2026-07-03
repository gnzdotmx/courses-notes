import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Toy 2D dataset: rows = samples, cols = features (both features rise together)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)

# Covariance of features (columns); rowvar=False → variables are columns
C = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(C)

print("Covariance matrix:\n", C)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns = principal axes):\n", eigenvectors)

# Variance explained: each λ / sum(λ)
total_var = eigenvalues.sum()
explained = eigenvalues / total_var
for i, (lam, pct) in enumerate(zip(eigenvalues, explained)):
    print(f"PC{i + 1}: λ={lam:.4f}, variance explained={pct * 100:.1f}%")

mean = X.mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: data + eigenvector directions scaled by sqrt(λ)
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c="steelblue", edgecolors="k", zorder=3)
colors = ["crimson", "seagreen"]
for i in range(len(eigenvalues)):
    vec = eigenvectors[:, i]
    scale = np.sqrt(eigenvalues[i]) * 1.5
    ax.annotate(
        "",
        xy=mean + vec * scale,
        xytext=mean,
        arrowprops=dict(arrowstyle="->", color=colors[i], lw=2),
    )
    ax.text(
        mean[0] + vec[0] * scale * 1.1,
        mean[1] + vec[1] * scale * 1.1,
        f"PC{i + 1}",
        color=colors[i],
    )
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.set_title("Data + covariance eigenvectors")
ax.grid(alpha=0.3)

# Right: variance explained by each principal component
ax = axes[1]
ax.bar([f"PC{i + 1}" for i in range(len(eigenvalues))], explained * 100, color=colors)
ax.set_ylabel("% variance explained")
ax.set_title("Eigenvalue importance")
ax.set_ylim(0, 100)

out = Path(__file__).resolve().parent / "EigenvaluesCovariance.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
