import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

X, y = make_classification(n_features=2, n_redundant=0, n_clusters_per_class=1)
clf = LogisticRegression().fit(X, y)

DecisionBoundaryDisplay.from_estimator(clf, X, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
out = Path(__file__).resolve().parent / "LogisticRegression.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
