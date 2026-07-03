import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

phishing = fetch_ucirepo(id=327)
X = phishing.data.features
y = phishing.data.targets.squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

# Top features by |coefficient|: which URL/SSL signals push toward phishing (-1)
feature_names = np.array(phishing.data.features.columns)
coef = clf.coef_.ravel()
top_idx = np.argsort(np.abs(coef))[-10:][::-1]

plt.barh(feature_names[top_idx][::-1], coef[top_idx][::-1])
plt.xlabel("coefficient (log-odds)")
plt.title("Logistic regression: top phishing features")
out = Path(__file__).resolve().parent / "LogisticRegressionPhishing.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
