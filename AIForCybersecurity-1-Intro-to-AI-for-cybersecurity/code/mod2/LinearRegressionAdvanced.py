import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import kagglehub

path = Path(kagglehub.dataset_download("uciml/sms-spam-collection-dataset"))
print(path)

# UCI SMS dataset is not UTF-8; latin-1 avoids UnicodeDecodeError
df = pd.read_csv(path / "spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})

keywords = ["free", "win"]


def keyword_freq(text, word):
    return str(text).lower().count(word)


df["kw1"] = df["message"].apply(lambda t: keyword_freq(t, keywords[0]))
df["kw2"] = df["message"].apply(lambda t: keyword_freq(t, keywords[1]))

X = df[["kw1", "kw2"]].values
y = df["label"].map({"ham": 0, "spam": 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = (model.predict(X_test) >= 0.5).astype(int)

print(accuracy_score(y_test, y_pred))

# Threshold boundary at 0.5: linear regression outputs a score, not a class
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], colors="k", linestyles="--")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k")
plt.xlabel(keywords[0])
plt.ylabel(keywords[1])
plt.title("Linear regression spam filter (threshold = 0.5)")
out = Path(__file__).resolve().parent / "LinearRegressionAdvanced.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
