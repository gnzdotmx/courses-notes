import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
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
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Perceptron(eta0=0.1, max_iter=1000)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

DecisionBoundaryDisplay.from_estimator(clf, X_test, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k")
plt.xlabel(keywords[0])
plt.ylabel(keywords[1])
out = Path(__file__).resolve().parent / "PerceptronSpamFilter.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
