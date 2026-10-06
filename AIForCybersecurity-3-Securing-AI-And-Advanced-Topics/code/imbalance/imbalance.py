"""Pictures of class imbalance: raw counts, undersample, copy, SMOTE.

Green = legit (class 0), red = fraud (class 1). A 2D cartoon, not the
31-feature card table. Resampling here is the whole drawing. In the lab,
resample the training split only and score on the original test mix.

    cd code/imbalance
    python imbalance.py

Writes figures/counts.png, figures/points.png, figures/boundaries.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

OUT = Path(__file__).resolve().parent / "figures"
RNG = np.random.RandomState(4)

GREEN = "#1b7f3a"
RED = "#c0392b"
DROPPED = "#c5d5c8"
SYNTH = "#e67e22"


def toy_set(n_legit=280, n_fraud=14):
    """Majority cloud, small fraud cloud sitting on its edge.

    The fraud cloud overlaps the legit one, so a model can score well by
    calling almost everything legit.
    """
    legit = RNG.normal(loc=[0.0, 0.0], scale=[0.85, 0.65], size=(n_legit, 2))
    fraud = RNG.normal(loc=[1.15, 0.45], scale=[0.28, 0.28], size=(n_fraud, 2))
    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


def undersample(X, y):
    """Keep every fraud row. Draw a legit subset of that same size."""
    legit = np.flatnonzero(y == 0)
    fraud = np.flatnonzero(y == 1)
    kept_legit = RNG.choice(legit, size=len(fraud), replace=False)
    dropped = np.setdiff1d(legit, kept_legit)
    keep = np.concatenate([kept_legit, fraud])
    return X[keep], y[keep], dropped


def random_oversample(X, y):
    """Copy fraud rows with replacement until the two counts match."""
    legit = np.flatnonzero(y == 0)
    fraud = np.flatnonzero(y == 1)
    extra = RNG.choice(fraud, size=len(legit) - len(fraud), replace=True)
    idx = np.concatenate([legit, fraud, extra])
    return X[idx], y[idx], extra


def smote(X, y, k=3):
    """Place new fraud rows on the segment between a fraud row and a neighbor.

    x_new = x_i + lambda * (x_z - x_i), lambda in [0, 1].
    Legit rows are copied through unchanged.
    """
    legit = X[y == 0]
    fraud = X[y == 1]
    n_new = len(legit) - len(fraud)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(fraud))).fit(fraud)
    synth = []
    parents = []  # (index of x_i, index of neighbor) so the figure can draw the segment
    for _ in range(n_new):
        i = int(RNG.randint(0, len(fraud)))
        # Neighbor 0 is the point itself. Take one of the others.
        _, ind = nn.kneighbors(fraud[i : i + 1])
        z = int(RNG.choice(ind[0][1:]))
        lam = float(RNG.uniform(0.15, 0.85))
        synth.append(fraud[i] + lam * (fraud[z] - fraud[i]))
        parents.append((i, z))
    X_new = np.vstack([legit, fraud, np.vstack(synth)])
    y_new = np.array([0] * len(legit) + [1] * (len(fraud) + n_new))
    return X_new, y_new, fraud, np.vstack(synth), parents


def counts(y):
    return int((y == 0).sum()), int((y == 1).sum())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.2, 2.2)


def scatter_classes(ax, X, y, s=16):
    for label, color in ((0, GREEN), (1, RED)):
        m = y == label
        ax.scatter(X[m, 0], X[m, 1], s=s, c=color, edgecolors="white", linewidths=0.3, zorder=3)


def show_counts(y_raw, y_under, y_over, y_smote):
    """Bar height is the row count the trainer would see."""
    labels = ["Raw", "Undersample", "Copy fraud", "SMOTE"]
    series = [counts(y) for y in (y_raw, y_under, y_over, y_smote)]
    legit = [a for a, _ in series]
    fraud = [b for _, b in series]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.18, legit, width=0.36, color=GREEN, label="legit")
    ax.bar(x + 0.18, fraud, width=0.36, color=RED, label="fraud")
    ax.set_xticks(x, labels)
    ax.set_ylabel("rows")
    # Sits in the empty gap between the short undersample bars and the tall copy bars.
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.36, 0.55))
    ax.set_title("Rebalancing changes the counts the trainer sees. The test set stays raw.")
    for i, (a, b) in enumerate(series):
        ax.text(i - 0.18, a + 4, str(a), ha="center", va="bottom", fontsize=8, color=GREEN)
        ax.text(i + 0.18, b + 4, str(b), ha="center", va="bottom", fontsize=8, color=RED)
    fig.tight_layout()
    save(fig, "counts.png")


def show_points(X, y, dropped, extra_idx, fraud, synth, parents):
    """What each method does to the rows, before any model is fit."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    scatter_classes(axes[0], X, y, s=14)
    n0, n1 = counts(y)
    axes[0].set_title(f"Raw\n{n0} legit, {n1} fraud", fontsize=10)

    # Dropped legit rows stay on the plot in gray so the cut is visible.
    axes[1].scatter(X[dropped, 0], X[dropped, 1], s=12, c=DROPPED, zorder=2)
    Xu, yu, _ = undersample_kept_view(X, y, dropped)
    scatter_classes(axes[1], Xu, yu, s=22)
    axes[1].set_title(f"Undersample\nkeep {n1} legit, drop the rest", fontsize=10)

    # Copies sit on the same coordinates. Marker area shows how many times a fraud row was drawn.
    scatter_classes(axes[2], X[y == 0], np.zeros((y == 0).sum(), dtype=int), s=14)
    fraud_idx = np.flatnonzero(y == 1)
    copies = np.bincount(np.concatenate([fraud_idx, extra_idx]), minlength=len(y))
    axes[2].scatter(
        X[fraud_idx, 0], X[fraud_idx, 1],
        s=18 + 10 * copies[fraud_idx],
        c=RED, edgecolors="black", linewidths=0.5, zorder=3,
    )
    axes[2].set_title("Random oversample\nrings = the same fraud row, copied", fontsize=10)

    scatter_classes(axes[3], X, y, s=14)
    # Full synthetic cloud, then a few segments on top so the interpolation is visible.
    axes[3].scatter(synth[:, 0], synth[:, 1], s=10, c=SYNTH, marker="D", alpha=0.35, zorder=3)
    show = RNG.choice(len(parents), size=min(8, len(parents)), replace=False)
    for j in show:
        i, z = parents[j]
        axes[3].plot(
            [fraud[i, 0], fraud[z, 0]],
            [fraud[i, 1], fraud[z, 1]],
            color="#d35400", linewidth=0.8, zorder=4,
        )
        axes[3].scatter(synth[j, 0], synth[j, 1], s=28, c=SYNTH, marker="D", edgecolors="black", linewidths=0.4, zorder=5)
    axes[3].set_title("SMOTE\ndiamond = a point on the segment", fontsize=10)

    for ax in axes:
        style(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=7, label="legit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=7, label="fraud"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DROPPED, markersize=7, label="dropped legit"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=SYNTH, markersize=6, label="synthetic fraud"),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=7)
    fig.suptitle("Imbalance fixes change the training rows. Gray points are legit rows undersampling throws away.", fontsize=11)
    fig.tight_layout()
    save(fig, "points.png")


def undersample_kept_view(X, y, dropped):
    """Rows that remain after those legit indexes are removed."""
    keep = np.ones(len(y), dtype=bool)
    keep[dropped] = False
    return X[keep], y[keep], dropped


def grid(step=0.04):
    x0 = np.arange(-2.6, 2.6, step)
    x1 = np.arange(-2.2, 2.2, step)
    xx, yy = np.meshgrid(x0, x1)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def paint(ax, xx, yy, score):
    z = score.reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.9, vmin=0, vmax=1)
    ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.0)


def fraud_recall(model, X, y):
    pred = model.predict(X)
    pos = y == 1
    return float((pred[pos] == 1).mean()) if pos.any() else 0.0


def show_boundaries(X, y, sets):
    """Same logistic model. Only the training rows change.

    Recall is measured on the original raw points, which is the test-style question:
    of the real fraud rows, how many does the model still catch?
    """
    xx, yy, pts = grid()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (title, Xtr, ytr) in zip(axes, sets):
        model = LogisticRegression(max_iter=400).fit(Xtr, ytr)
        paint(ax, xx, yy, model.predict_proba(pts)[:, 1])
        scatter_classes(ax, X, y, s=12)
        recall = fraud_recall(model, X, y)
        ax.set_title(f"{title}\nfraud recall on the raw rows: {recall:.2f}", fontsize=10)
        style(ax)
    fig.suptitle("Black line = 0.5. A raw fit can leave fraud on the green side. Balancing pulls the line back.", fontsize=11)
    fig.tight_layout()
    save(fig, "boundaries.png")


def main():
    X, y = toy_set()
    Xu, yu, dropped = undersample(X, y)
    Xo, yo, extra = random_oversample(X, y)
    Xs, ys, fraud, synth, parents = smote(X, y)

    show_counts(y, yu, yo, ys)
    show_points(X, y, dropped, extra, fraud, synth, parents)
    show_boundaries(X, y, [
        ("Raw", X, y),
        ("Undersample", Xu, yu),
        ("Copy fraud", Xo, yo),
        ("SMOTE", Xs, ys),
    ])


if __name__ == "__main__":
    main()
