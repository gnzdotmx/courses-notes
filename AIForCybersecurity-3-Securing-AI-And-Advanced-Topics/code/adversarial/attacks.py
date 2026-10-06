"""Pictures of FGSM, PGD, a patch mask, transfer, a Siamese threshold, and three defenses.

Two blobs in a plane, or a drawn scene. Not a real detector, malware model, or face system.

    cd code/adversarial
    python attacks.py

Writes figures/fgsm_pgd.png, figures/patch.png, figures/transfer.png,
figures/siamese.png, figures/defenses.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent / "figures"
RNG = np.random.RandomState(2)
GREEN = "#1b7f3a"
RED = "#c0392b"
BLUE = "#2471a3"


def toy():
    """Class 0 on the left, class 1 on the right. The boundary sits near x = 0."""
    x0 = RNG.normal(loc=[-1.6, 0.1], scale=0.38, size=(50, 2))
    x1 = RNG.normal(loc=[1.6, -0.1], scale=0.38, size=(50, 2))
    X = np.vstack([x0, x1])
    y = np.array([0] * len(x0) + [1] * len(x1))
    model = LogisticRegression().fit(X, y)
    return X, y, model


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def grad_x(model, x, y):
    """d(cross-entropy)/dx = (p - y) * w. Sign of this raises the loss on the true label."""
    w = model.coef_.ravel()
    b = float(model.intercept_[0])
    p = sigmoid(x @ w + b)
    return (p - y) * w


def fgsm(model, x, y, eps):
    return x + eps * np.sign(grad_x(model, x, y))


def pgd(model, x, y, eps, alpha, steps):
    """FGSM, repeated, then clipped back into the L-inf square around the clean x."""
    path = [x.copy()]
    cur = x.copy()
    for _ in range(steps):
        cur = cur + alpha * np.sign(grad_x(model, cur, y))
        cur = np.clip(cur, x - eps, x + eps)
        path.append(cur.copy())
    return np.vstack(path)


def grid():
    g = np.linspace(-3.2, 3.2, 220)
    xx, yy = np.meshgrid(g, g)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def paint(ax, model, X, y):
    xx, yy, pts = grid()
    score = model.predict_proba(pts)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, score, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.9, vmin=0, vmax=1)
    ax.contour(xx, yy, score, levels=[0.5], colors="black", linewidths=1.0)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c=GREEN, edgecolors="white", linewidths=0.3, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c=RED, edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def show_fgsm_pgd():
    """One clean point just inside class 0. FGSM jumps once. PGD takes smaller steps inside the square."""
    X, y, model = toy()
    # A class-0 point close to the boundary, so a modest budget can cross it.
    x = np.array([-0.35, 0.15])
    label = 0
    eps = 0.9
    adv = fgsm(model, x, label, eps)
    path = pgd(model, x, label, eps=eps, alpha=0.3, steps=4)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax in axes:
        paint(ax, model, X, y)
        square = Rectangle((x[0] - eps, x[1] - eps), 2 * eps, 2 * eps, fill=False, linestyle="--", linewidth=1.0, color=BLUE)
        ax.add_patch(square)

    axes[0].scatter(*x, s=60, c="black", zorder=4)
    axes[0].set_title("Clean\npoint is still on the green side", fontsize=10)

    axes[1].annotate("", xy=adv, xytext=x, arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    axes[1].scatter(*x, s=40, c="black", zorder=4)
    axes[1].scatter(*adv, s=60, c=BLUE, zorder=4)
    axes[1].set_title("FGSM\none step of size ε, along sign(∇)", fontsize=10)

    axes[2].plot(path[:, 0], path[:, 1], color="black", linewidth=1.2, zorder=4)
    axes[2].scatter(path[:, 0], path[:, 1], s=28, c=BLUE, zorder=5)
    axes[2].set_title("PGD\nsmaller steps, clipped back into the square", fontsize=10)

    fig.suptitle("Dashed square = L∞ budget. Black line = 0.5. The point starts green and is pushed into red.", fontsize=11)
    fig.tight_layout()
    save(fig, "fgsm_pgd.png")


def show_patch():
    """A scene stays put. Only a sticker-sized mask is free to change."""
    scene = np.ones((64, 64, 3)) * 0.85
    yy, xx = np.mgrid[0:64, 0:64]
    blob = (xx - 34) ** 2 + (yy - 36) ** 2 < 16 ** 2
    scene[blob] = (0.15, 0.45, 0.75)
    patched = scene.copy()
    patched[8:24, 8:24] = (0.85, 0.25, 0.15)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].imshow(scene)
    axes[0].set_title("Clean scene\nevery pixel is fixed", fontsize=10)
    axes[1].imshow(patched)
    rect = Rectangle((7.5, 7.5), 16, 16, fill=False, linewidth=1.2, color="black")
    axes[1].add_patch(rect)
    axes[1].set_title("Patch\nonly the boxed pixels are optimized", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("A patch attack does not nudge the whole image. The mask M is 1 inside the sticker and 0 elsewhere.", fontsize=11)
    fig.tight_layout()
    save(fig, "patch.png")


def show_transfer():
    """Queries label a few points. A local model fit on those labels is attacked. The step also crosses the real boundary."""
    X, y, true = toy()
    # A thin query set: the attacker never sees weights, only these labels.
    q = RNG.choice(len(y), size=24, replace=False)
    surrogate = LogisticRegression().fit(X[q], true.predict(X[q]))
    x = np.array([-0.45, -0.2])
    adv = fgsm(surrogate, x, 0, eps=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    paint(axes[0], true, X, y)
    axes[0].scatter(X[q, 0], X[q, 1], s=36, facecolors="none", edgecolors="black", linewidths=0.8, zorder=4)
    axes[0].set_title("Black box\nrings are the queries that return a label", fontsize=10)

    # Draw the true fill, then the surrogate's 0.5 line on top.
    paint(axes[1], true, X, y)
    xx, yy, pts = grid()
    s_score = surrogate.predict_proba(pts)[:, 1].reshape(xx.shape)
    axes[1].contour(xx, yy, s_score, levels=[0.5], colors=BLUE, linewidths=1.4)
    axes[1].annotate("", xy=adv, xytext=x, arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    axes[1].scatter(*x, s=40, c="black", zorder=5)
    axes[1].scatter(*adv, s=60, c=BLUE, zorder=5)
    axes[1].set_title("Transfer\nblue line = surrogate. The step was computed on it.", fontsize=10)
    fig.suptitle("The attacker fits a local model on query labels, runs FGSM there, and sends that point to the real model.", fontsize=11)
    fig.tight_layout()
    save(fig, "transfer.png")


def show_siamese():
    """Match when the embedding distance is inside the circle. A nudge of one face vector can exit it."""
    e1 = np.array([0.0, 0.0])
    e2 = np.array([0.55, 0.25])
    e2_adv = np.array([1.35, 0.85])
    tau = 1.05

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.set_xlim(-0.4, 2.3)
    ax.set_ylim(-0.4, 2.1)
    ax.add_patch(Circle(e1, tau, fill=False, linestyle="--", color=BLUE, linewidth=1.2))
    ax.scatter(*e1, s=70, c=GREEN, zorder=3)
    ax.scatter(*e2, s=70, c=GREEN, zorder=3)
    ax.scatter(*e2_adv, s=70, c=RED, zorder=3)
    ax.annotate("", xy=e2_adv, xytext=e2, arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.text(e1[0] - 0.05, e1[1] - 0.22, "e1", fontsize=9)
    ax.text(e2[0] + 0.06, e2[1] + 0.06, "e2  match", fontsize=9, color=GREEN)
    ax.text(e2_adv[0] + 0.06, e2_adv[1], "e2 adv  no match", fontsize=9, color=RED)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Dashed circle = threshold τ around e1.\nInside: same person. The step pushes e2 outside.")
    fig.tight_layout()
    save(fig, "siamese.png")


def show_defenses():
    """Same budget, three different responses: flag the odd point, follow a fake arrow, move the boundary."""
    X, y, model = toy()
    x = np.array([-0.35, 0.15])
    eps = 0.9
    adv = fgsm(model, x, 0, eps)
    # Scrambled gradient: a direction that spends the budget without crossing.
    scrambled = x + np.array([0.0, eps])

    # Adversarial training, cartoon: push every class-0 point, label those
    # attacked copies with the true class, and repeat them so the boundary
    # has to move. One copy is not enough against the original cloud.
    seeds = X[y == 0]
    advs = np.vstack([fgsm(model, s, 0, eps) for s in seeds])
    X_aug = np.vstack([X, advs, advs, advs, advs, advs])
    y_aug = np.concatenate([y, np.zeros(len(advs) * 5, dtype=int)])
    robust = LogisticRegression().fit(X_aug, y_aug)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    paint(axes[0], model, X, y)
    axes[0].scatter([2.4], [2.4], s=70, c="#f4d03f", edgecolors="black", zorder=4)
    axes[0].set_title("Statistical\na point off the clouds is flagged\nbefore the class is trusted", fontsize=10)

    paint(axes[1], model, X, y)
    axes[1].annotate("", xy=adv, xytext=x, arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=1.2))
    axes[1].annotate("", xy=scrambled, xytext=x, arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    axes[1].scatter(*x, s=40, c="black", zorder=4)
    axes[1].scatter(*scrambled, s=50, c=BLUE, zorder=4)
    axes[1].set_title("Gradient masking\nblack arrow is the scrambled ∇.\nGray arrow is the real one.", fontsize=10)

    paint(axes[2], robust, X, y)
    axes[2].scatter(*adv, s=60, c=BLUE, zorder=4)
    axes[2].scatter(*x, s=40, c="black", zorder=4)
    axes[2].set_title("Adversarial training\nsame attacked point, refit with its true label.\nIt lands back on green.", fontsize=10)

    fig.suptitle("Three defenses. Detect the odd input, hide the true step, or train on the attacked points.", fontsize=11)
    fig.tight_layout()
    save(fig, "defenses.png")


def main():
    show_fgsm_pgd()
    show_patch()
    show_transfer()
    show_siamese()
    show_defenses()


if __name__ == "__main__":
    main()
