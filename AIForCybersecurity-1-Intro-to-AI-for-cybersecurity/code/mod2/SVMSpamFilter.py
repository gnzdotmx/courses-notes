import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import kagglehub

path = Path(kagglehub.dataset_download("uciml/sms-spam-collection-dataset"))
print(path)

# UCI SMS dataset is not UTF-8; latin-1 avoids UnicodeDecodeError
df = pd.read_csv(path / "spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})

suspect_words = ["free", "win", "prize"]   # spam signals
neutral_words = ["meeting", "thanks", "ok"]  # ham signals


def count_any(text, words):
    t = str(text).lower()
    return sum(t.count(w) for w in words)


df["suspect_freq"] = df["message"].apply(lambda t: count_any(t, suspect_words))
df["neutral_freq"] = df["message"].apply(lambda t: count_any(t, neutral_words))

X = df[["suspect_freq", "neutral_freq"]].values
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel="rbf", C=1.0)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

DecisionBoundaryDisplay.from_estimator(clf, X_test, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k")
plt.xlabel("suspect freq")
plt.ylabel("neutral freq")
out = Path(__file__).resolve().parent / "SVMSpamFilter.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
