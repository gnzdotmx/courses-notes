"""Generate synthetic login_events.csv for LoginRiskScoring.py."""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n_legit = 4000
n_attack = 1000

legit = pd.DataFrame(
    {
        "failed_attempts_1min": np.clip(np.random.poisson(0.5, n_legit) + np.random.randint(0, 2, n_legit), 0, 6),
        "passwords_tried_count": np.clip(np.random.poisson(1.2, n_legit), 0, 5),
        "geo_distance_km": np.abs(np.random.exponential(80, n_legit)),
        "os_is_usual": np.random.binomial(1, 0.88, n_legit),
        "accounts_same_ip": np.clip(np.random.poisson(1.2, n_legit), 0, 4),
        "activity_burst_seconds": np.random.uniform(45, 3600, n_legit),
        "is_attack": 0,
    }
)

attack = pd.DataFrame(
    {
        "failed_attempts_1min": np.clip(np.random.poisson(6, n_attack) + np.random.randint(0, 3, n_attack), 0, 15),
        "passwords_tried_count": np.clip(np.random.poisson(12, n_attack), 1, 30),
        "geo_distance_km": np.abs(np.random.exponential(500, n_attack)),
        "os_is_usual": np.random.binomial(1, 0.25, n_attack),
        "accounts_same_ip": np.clip(np.random.poisson(6, n_attack), 1, 15),
        "activity_burst_seconds": np.random.uniform(3, 180, n_attack),
        "is_attack": 1,
    }
)

df = pd.concat([legit, attack], ignore_index=True).sample(frac=1, random_state=42)

# Borderline cases: overlapping feature ranges (realistic false-positive risk)
borderline_legit = df[df["is_attack"] == 0].sample(200, random_state=7).index
df.loc[borderline_legit, "failed_attempts_1min"] = np.random.randint(2, 7, 200)
df.loc[borderline_legit, "passwords_tried_count"] = np.random.randint(3, 8, 200)
df.loc[borderline_legit, "geo_distance_km"] = np.abs(np.random.exponential(350, 200))

borderline_attack = df[df["is_attack"] == 1].sample(80, random_state=11).index
df.loc[borderline_attack, "failed_attempts_1min"] = np.random.randint(0, 2, 80)
df.loc[borderline_attack, "geo_distance_km"] = np.abs(np.random.exponential(60, 80))
df.loc[borderline_attack, "os_is_usual"] = 1

out = Path(__file__).resolve().parent / "data" / "login_events.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
