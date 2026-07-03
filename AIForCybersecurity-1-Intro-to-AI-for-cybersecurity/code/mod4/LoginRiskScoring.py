import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_path = Path(__file__).resolve().parent / "data" / "login_events.csv"
print(data_path)

df = pd.read_csv(data_path)

feature_cols = [
    "failed_attempts_1min",
    "passwords_tried_count",
    "geo_distance_km",
    "os_is_usual",
    "accounts_same_ip",
    "activity_burst_seconds",
]

X = df[feature_cols]
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))

importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1]

plt.barh(np.array(feature_cols)[top_idx][::-1], importances[top_idx][::-1])
plt.xlabel("feature importance")
plt.title("Login risk scoring: top attack signals")
out = Path(__file__).resolve().parent / "LoginRiskScoring.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
