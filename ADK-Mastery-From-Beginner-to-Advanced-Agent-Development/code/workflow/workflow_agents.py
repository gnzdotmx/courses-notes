"""Draw the sequential pipeline and the parallel fan-out.

Toy diagram of the two workflow shapes in this module.
Regenerate the notes figure with:

    python workflow_agents.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "sequential_parallel.png"

NAVY = "#1f4e79"
TEAL = "#0f6e56"
INK = "#1c1917"
MUTED = "#57534e"
PAPER = "#fafaf9"
BOX = "#ffffff"


def box(ax, x, y, w, h, text, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=BOX,
        edgecolor=edge,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color=INK,
        linespacing=1.25,
    )


def arrow(ax, start, end, color, label=None, label_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )


def draw_sequential(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.set_axis_off()
    ax.set_title("SequentialAgent  ·  list order is the run order", loc="left", fontsize=11, color=NAVY, pad=6)

    specs = [
        (0.15, "User"),
        (2.15, "Research"),
        (4.4, "Itinerary"),
        (6.65, "Optimizer"),
        (8.5, "Reply"),
    ]
    w, h, y = 1.4, 0.78, 1.05
    for x, text in specs:
        box(ax, x, y, w, h, text, NAVY)

    gaps = [
        (1.55, 2.15, None),
        (3.55, 4.4, "destination_research"),
        (5.8, 6.65, "travel_itinerary"),
        (8.05, 8.5, None),
    ]
    mid_y = y + h / 2
    for x0, x1, label in gaps:
        arrow(ax, (x0, mid_y), (x1, mid_y), NAVY)
        if label:
            ax.text((x0 + x1) / 2, y - 0.08, label, ha="center", va="top", fontsize=7, color=NAVY)


def draw_parallel(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.set_axis_off()
    ax.set_title("ParallelAgent  ·  every worker starts from the same message", loc="left", fontsize=11, color=TEAL, pad=6)

    names = ["Blog", "SEO", "Visual", "Social", "Email"]
    worker_w, worker_h = 1.7, 0.58
    gap = 0.18
    top = 3.85
    ys = [top - i * (worker_h + gap) for i in range(len(names))]
    worker_x = 3.7
    user_h, reply_h = 0.7, 0.7
    mid = (ys[0] + worker_h + ys[-1]) / 2

    box(ax, 0.35, mid - user_h / 2, 1.45, user_h, "User", TEAL)
    box(ax, 8.15, mid - reply_h / 2, 1.45, reply_h, "Reply", TEAL)
    for y, name in zip(ys, names):
        box(ax, worker_x, y, worker_w, worker_h, name, TEAL)
        arrow(ax, (1.8, mid), (worker_x, y + worker_h / 2), TEAL)
        arrow(ax, (worker_x + worker_w, y + worker_h / 2), (8.15, mid), TEAL)


def main():
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 6.6), dpi=140, gridspec_kw={"height_ratios": [1, 1.7]}
    )
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
    draw_sequential(axes[0])
    draw_parallel(axes[1])
    fig.text(
        0.01,
        0.01,
        "Toy control flow. Workers are the module examples, not a live run.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
