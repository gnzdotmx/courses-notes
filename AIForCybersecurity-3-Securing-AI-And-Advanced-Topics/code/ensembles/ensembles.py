"""Pictures of bagging, boosting, and stacking on one 2D toy set.

Green = class 0, red = class 1. The set is a cartoon (two moons), not the
31-feature card table from the fraud lab. Same three ensemble ideas.

    cd code/ensembles
    python ensembles.py

Writes figures/bagging.png, figures/boosting.png, figures/stacking.png.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parent / "figures"
RNG = np.random.RandomState(7)

GREEN = "#1b7f3a"
RED = "#c0392b"


def toy_set(n=160, noise=0.28):
    """Two interlocking half-circles. One straight cut cannot separate them."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=7)
    return X, y


def grid(X, step=0.02, pad=0.45):
    x0 = np.arange(X[:, 0].min() - pad, X[:, 0].max() + pad, step)
    x1 = np.arange(X[:, 1].min() - pad, X[:, 1].max() + pad, step)
    xx, yy = np.meshgrid(x0, x1)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def paint(ax, xx, yy, score, X, y, size=22, emphasize=None):
    """score is P(class 1). The black line is the 0.5 vote.

    emphasize=True marks the rows to look at (a duplicate draw, or a miss
    the next round will boost). Those dots get a black ring.
    """
    z = score.reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.9, vmin=0, vmax=1)
    ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.1)
    sizes = np.full(len(y), size) if np.ndim(size) == 0 else np.asarray(size, dtype=float)
    edges = np.full(len(y), "white", dtype=object)
    if emphasize is not None:
        edges[np.asarray(emphasize)] = "black"
    for label, color in ((0, GREEN), (1, RED)):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=sizes[mask],
            c=color,
            edgecolors=edges[mask],
            linewidths=0.6,
            zorder=3,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def class_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="white", markersize=8, label="class 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=8, label="class 1"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def show_bagging(X, y, n_trees=25):
    """Bagging = bootstrap aggregating.

    A bag is one dataset drawn with replacement (a row can appear twice, or
    not at all). Train a tree on each bag in parallel; none of them sees the
    others' mistakes. Aggregate = the mean of every tree's vote.
    The first three panels are three of those bags. The last panel averages
    all of them, which is what a random forest does.
    """
    n = len(y)
    xx, yy, pts = grid(X)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    votes = []

    for i in range(n_trees):
        # One bag: n draws with replacement from the n rows.
        drawn = RNG.randint(0, n, size=n)
        counts = np.bincount(drawn, minlength=n)
        # max_features=1: each split looks at one axis, so bags disagree.
        tree = DecisionTreeClassifier(max_depth=4, max_features=1, random_state=i)
        tree.fit(X[drawn], y[drawn])
        votes.append(tree.predict_proba(pts)[:, 1])
        if i >= 3:
            continue
        # Absent from this bag → no dot. Drawn twice → a larger dot with a ring.
        paint(
            axes[i], xx, yy, votes[-1], X, y,
            size=np.where(counts == 0, 0, 14 + 28 * counts),
            emphasize=counts >= 2,
        )
        unique = int(np.count_nonzero(counts))
        axes[i].set_title(f"Bag {i + 1}\n{unique} of {n} unique rows", fontsize=10)

    # Aggregating: each tree casts a probability; the forest takes the mean.
    paint(axes[-1], xx, yy, np.mean(votes, axis=0), X, y)
    axes[-1].set_title(f"Aggregate\nmean of all {n_trees} votes", fontsize=10)
    class_legend(axes[0])
    fig.suptitle("Bagging: draw a bag of rows, train one tree per bag, average the votes", fontsize=12)
    fig.tight_layout()
    save(fig, "bagging.png")


def show_boosting(X, y):
    """Boosting: each next stump boosts the rows the ensemble still gets wrong.

    A stump is a one-split tree. It cannot follow the moons alone. AdaBoost
    raises the sample weight of the rows the current vote misses, fits
    another stump on those weights, and adds it in. learning_rate < 1 keeps
    one stump from overturning the vote. Black rings are the rows still wrong.
    """
    xx, yy, pts = grid(X)
    # SAMME.R (sklearn 1.5) adds a real-valued vote, not only a hard ±1.
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=6,
        learning_rate=0.5,
        algorithm="SAMME.R",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        ada.fit(X, y)
        # Stage i is the vote of the first i+1 stumps.
        on_grid = list(ada.staged_decision_function(pts))
        on_train = list(ada.staged_predict(X))

    show_at = (0, 1, 3, 5)  # 1, 2, 4, 6 stumps: misses fall across these
    fig, axes = plt.subplots(1, len(show_at), figsize=(14, 3.6))
    for ax, stage in zip(axes, show_at):
        misses = on_train[stage] != y
        score = 1.0 / (1.0 + np.exp(-on_grid[stage].ravel()))
        paint(ax, xx, yy, score, X, y, size=np.where(misses, 64, 16), emphasize=misses)
        ax.set_title(
            f"{stage + 1} stump{'s' if stage else ''}\n{int(misses.sum())} rows still wrong",
            fontsize=10,
        )
    class_legend(axes[0])
    fig.suptitle("Boosting: each new stump is aimed at the rows the vote still misses", fontsize=12)
    fig.tight_layout()
    save(fig, "boosting.png")


def show_stacking(X, y):
    """Stacking: base models on the bottom, one model stacked on their outputs.

    The three base models see the raw (x0, x1) points. The top model does
    not. It sees only the three probabilities they emit, and learns how to
    trust them. sklearn's StackingClassifier fits that top model on
    out-of-fold predictions, so it cannot simply copy a memorized base fit.
    """
    bases = [
        ("linear", LogisticRegression(max_iter=400), "Bottom: straight cut"),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=0), "Bottom: shallow tree"),
        ("knn", KNeighborsClassifier(n_neighbors=7), "Bottom: 7 nearest neighbors"),
    ]
    xx, yy, pts = grid(X)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for ax, (name, model, title) in zip(axes[:-1], bases):
        model.fit(X, y)
        score = model.predict_proba(pts)[:, 1]
        paint(ax, xx, yy, score, X, y)
        ax.set_title(title, fontsize=10)

    stack = StackingClassifier(
        estimators=[(name, model) for name, model, _ in bases],
        final_estimator=LogisticRegression(max_iter=400),
        cv=5,
        passthrough=False,  # top model sees base scores only, not the raw x
    )
    stack.fit(X, y)
    score = stack.predict_proba(pts)[:, 1]
    paint(axes[-1], xx, yy, score, X, y)
    axes[-1].set_title("Top of the stack\nlogistic blend of the 3 scores", fontsize=10)
    class_legend(axes[0])
    fig.suptitle("Stacking: a meta-learner sits on the predictions, not on the raw features", fontsize=12)
    fig.tight_layout()
    save(fig, "stacking.png")


def main():
    X, y = toy_set()
    show_bagging(X, y)
    show_boosting(X, y)
    show_stacking(X, y)


if __name__ == "__main__":
    main()
