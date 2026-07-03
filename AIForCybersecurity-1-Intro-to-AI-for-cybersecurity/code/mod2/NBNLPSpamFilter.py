import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import kagglehub

path = Path(kagglehub.dataset_download("uciml/sms-spam-collection-dataset"))
print(path)

# UCI SMS dataset is not UTF-8; latin-1 avoids UnicodeDecodeError
df = pd.read_csv(path / "spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})

X_text = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB()),
])

pipe.fit(X_train, y_train)

print(accuracy_score(y_test, pipe.predict(X_test)))

# Top spam-indicative words learned from training data (class 1 = spam)
nb = pipe.named_steps["nb"]
vocab = pipe.named_steps["tfidf"].get_feature_names_out()
scores = nb.feature_log_prob_[1]
top_idx = np.argsort(scores)[-15:][::-1]

plt.barh(vocab[top_idx][::-1], scores[top_idx][::-1])
plt.xlabel("log P(word | spam)")
plt.title("Top spam words (Naive Bayes)")
out = Path(__file__).resolve().parent / "NBNLPSpamFilter.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
