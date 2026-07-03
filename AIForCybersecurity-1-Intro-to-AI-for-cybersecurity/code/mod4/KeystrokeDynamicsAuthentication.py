import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data_path = Path(__file__).resolve().parent / "data" / "keystroke_biometric.csv"
print(data_path)

df = pd.read_csv(data_path)

feature_cols = [c for c in df.columns if c not in ("user_id", "session_id")]
X = df[feature_cols].values
y = df["user_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results[name] = acc
    print(f"{name}: {acc:.3f}")

plt.bar(results.keys(), results.values())
plt.ylim(0, 1)
plt.ylabel("accuracy")
plt.title("Keystroke dynamics: MLP vs SVM vs KNN")
out = Path(__file__).resolve().parent / "KeystrokeDynamicsAuthentication.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
