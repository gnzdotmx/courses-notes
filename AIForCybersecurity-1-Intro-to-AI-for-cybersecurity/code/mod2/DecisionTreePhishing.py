import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

phishing = fetch_ucirepo(id=327)
X = phishing.data.features
y = phishing.data.targets.squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print(accuracy_score(y_test, tree.predict(X_test)))

# Feature importances: which attributes the tree splits on most often
feature_names = np.array(phishing.data.features.columns)
importances = tree.feature_importances_
top_idx = np.argsort(importances)[-10:][::-1]

plt.barh(feature_names[top_idx][::-1], importances[top_idx][::-1])
plt.xlabel("feature importance")
plt.title("Decision tree: top phishing features")
out = Path(__file__).resolve().parent / "DecisionTreePhishing.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
