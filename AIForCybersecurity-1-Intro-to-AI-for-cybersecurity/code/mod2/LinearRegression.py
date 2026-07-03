import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Create synthetic data
X = np.random.randn(100, 1)

y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.4

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, alpha=0.6)
plt.plot(X, model.predict(X), color="red")
out = Path(__file__).resolve().parent / "LinearRegression.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
