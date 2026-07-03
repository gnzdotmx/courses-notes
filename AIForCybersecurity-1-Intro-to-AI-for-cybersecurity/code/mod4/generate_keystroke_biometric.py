"""Generate synthetic keystroke_biometric.csv for KeystrokeDynamicsAuthentication.py."""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n_users = 51
samples_per_user = 400
n_features = 31

feature_cols = (
    [f"hold_{i:02d}" for i in range(1, 11)]
    + [f"flight_{i:02d}" for i in range(1, 11)]
    + [f"digraph_{i:02d}" for i in range(1, 10)]
    + ["total_duration_ms"]
)

# Rhythm clusters with overlap: mimics distinct but confusable typing profiles
n_clusters = 14
centroids = np.random.uniform(90, 200, (n_clusters, n_features))
cluster_ids = np.random.randint(0, n_clusters, n_users)
user_profiles = centroids[cluster_ids] + np.random.normal(0, 10, (n_users, n_features))

rows = []
for user_id in range(n_users):
    profile = user_profiles[user_id]
    for session_id in range(samples_per_user):
        noise = np.random.normal(0, 28, n_features)
        sample = np.clip(profile + noise, 20, 500)
        rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                **dict(zip(feature_cols, sample)),
            }
        )

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = Path(__file__).resolve().parent / "data" / "keystroke_biometric.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows, {n_users} users, {n_features} features to {out}")
