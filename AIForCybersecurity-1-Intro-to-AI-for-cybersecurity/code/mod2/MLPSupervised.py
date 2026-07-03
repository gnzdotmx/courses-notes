import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 500 samples, 10 numeric features, binary labels (supervised classification)
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale using train statistics only (avoid leaking test distribution)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP: input(10) -> hidden(64) -> hidden(32) -> output(1 class score)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy: {acc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(clf.loss_curve_)
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("loss")
axes[0].set_title("MLP training loss")
axes[0].grid(alpha=0.3)

axes[1].bar(["train", "test"], [clf.score(X_train, y_train), acc], color=["steelblue", "seagreen"])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("accuracy")
axes[1].set_title("MLP accuracy (scaled 10-feature data)")

out = Path(__file__).resolve().parent / "MLPSupervised.png"
plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
