"""Draw the three callback pairs on one request.

Toy diagram of agent, model, and tool hooks.
Regenerate the notes figure with:

    python callbacks.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "callback_hooks.png"

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
        linespacing=1.2,
    )


def arrow(ax, start, end, color):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def lane(ax, y, label, steps):
    ax.text(0.05, y + 0.38, label, ha="left", va="center", fontsize=8, color=MUTED)
    w, h, gap = 1.85, 0.78, 0.38
    x = 1.55
    centers = []
    for text, edge in steps:
        box(ax, x, y, w, h, text, edge)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(len(centers) - 1):
        arrow(ax, (centers[i] + w / 2, y + h / 2), (centers[i + 1] - w / 2, y + h / 2), NAVY)
    return y


def main():
    fig, ax = plt.subplots(figsize=(10.6, 4.6), dpi=140)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.35)
    ax.set_axis_off()
    ax.set_title(
        "One request  ·  teal boxes are your functions",
        loc="left",
        fontsize=11,
        color=NAVY,
        pad=8,
    )

    lane(
        ax,
        3.15,
        "agent",
        [
            ("user", NAVY),
            ("before\nagent", TEAL),
            ("agent", NAVY),
            ("after\nagent", TEAL),
            ("reply", NAVY),
        ],
    )
    lane(
        ax,
        1.7,
        "model",
        [
            ("before\nmodel", TEAL),
            ("model", NAVY),
            ("after\nmodel", TEAL),
        ],
    )
    lane(
        ax,
        0.25,
        "tool",
        [
            ("before\ntool", TEAL),
            ("tool", NAVY),
            ("after\ntool", TEAL),
        ],
    )

    fig.text(
        0.01,
        0.01,
        "Toy order. The model lane runs inside the agent step. The tool lane runs only when the model calls a tool.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
