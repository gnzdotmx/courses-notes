"""Pictures of accuracy, precision, recall, F1, and ROC-AUC.

Toy counts, not the card table. 95 legit rows and 5 fraud rows.

    cd code/metrics
    python metrics.py

Writes figures/bars.png, figures/confusion.png, figures/roc.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

OUT = Path(__file__).resolve().parent / "figures"
GREEN = "#1b7f3a"
RED = "#c0392b"
ORANGE = "#e67e22"
BLUE = "#2471a3"

# Locked example so the notes and the pictures share one arithmetic.
# Always legit: never alerts. Detector: catches 4 of 5 fraud, also flags 6 legit.
CASES = {
    "Always legit": {"tp": 0, "fp": 0, "fn": 5, "tn": 95},
    "Detector": {"tp": 4, "fp": 6, "fn": 1, "tn": 89},
}


def rates(c):
    """Precision, recall, F1, accuracy from one confusion matrix."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n
    # No alerts → the precision denominator is 0. Treat it as "did not fire."
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def show_bars():
    """Accuracy can rise while F1 stays at 0. F1 is the bar that notices."""
    names = list(CASES)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, name in enumerate(names):
        vals = [rates(CASES[name])[m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width, label=name, color=[GREEN, RED][i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("95 legit, 5 fraud. Always-legit wins on accuracy and catches nothing.")
    fig.tight_layout()
    save(fig, "bars.png")


def draw_matrix(ax, name, c):
    """Rows are the truth. Columns are the prediction."""
    grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    # Color by role: correct cells green, errors red. Alpha follows the count.
    color = np.zeros(grid.shape + (4,))
    for (i, j), count in np.ndenumerate(grid):
        base = (0x1b / 255, 0x7f / 255, 0x3a / 255) if (i, j) in ((0, 0), (1, 1)) else (0xC0 / 255, 0x39 / 255, 0x2B / 255)
        alpha = 0.15 + 0.75 * (count / grid.max())
        color[i, j] = (*base, alpha)
    ax.imshow(color, vmin=0, vmax=1)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), count in np.ndenumerate(grid):
        ax.text(j, i - 0.12, labels[i][j], ha="center", va="center", fontsize=9, color="#333")
        ax.text(j, i + 0.18, f"{int(count)}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1], ["pred legit", "pred fraud"])
    ax.set_yticks([0, 1], ["true legit", "true fraud"])
    r = rates(c)
    fired = "no alerts" if (c["tp"] + c["fp"]) == 0 else f"P {r['Precision']:.2f}"
    ax.set_title(f"{name}\nacc {r['Accuracy']:.2f}   {fired}   R {r['Recall']:.2f}   F1 {r['F1']:.2f}", fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, name in zip(axes, CASES):
        draw_matrix(ax, name, CASES[name])
    fig.suptitle("Same 100 rows. The detector is less accurate and is the one that catches fraud.", fontsize=11)
    fig.tight_layout()
    save(fig, "confusion.png")


def show_roc():
    """One score per row. Each threshold is a point. AUC is the area under the path."""
    rng = np.random.RandomState(1)
    n_legit, n_fraud = 200, 40
    y = np.array([0] * n_legit + [1] * n_fraud)
    # Overlap on purpose: a useful ranker, not a perfect one.
    scores = np.concatenate([
        rng.normal(0.0, 1.0, n_legit),
        rng.normal(1.5, 1.0, n_fraud),
    ])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", label="chance  AUC 0.50")
    fpr, tpr, _ = roc_curve(y, scores)
    area = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=RED, linewidth=2, label=f"ranker  AUC {area:.2f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color=RED)

    # Three operating points on that same curve. High threshold = few alerts.
    for tau, text in ((2.2, "strict"), (1.0, "middle"), (0.0, "loose")):
        pred = scores >= tau
        tp = int(((y == 1) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        tn = int(((y == 0) & ~pred).sum())
        px, py = fp / (fp + tn), tp / (tp + fn)
        ax.scatter([px], [py], s=40, zorder=3, color=BLUE)
        ax.annotate(text, (px, py), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("FPR  =  FP / (FP + TN)   legit rows flagged")
    ax.set_ylabel("TPR  =  TP / (TP + FN)   fraud rows caught")
    ax.set_title("ROC: sweep the threshold. AUC is the shaded area.")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "roc.png")


def main():
    show_bars()
    show_confusion()
    show_roc()


if __name__ == "__main__":
    main()
