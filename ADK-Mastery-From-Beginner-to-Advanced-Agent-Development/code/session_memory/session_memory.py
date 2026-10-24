"""Draw one session, and where that thread lives after the process exits.

Toy diagram of the session object and the two stores in this module.
Regenerate the notes figure with:

    python session_memory.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "session_state_memory.png"

NAVY = "#1f4e79"
TEAL = "#0f6e56"
INK = "#1c1917"
MUTED = "#57534e"
PAPER = "#fafaf9"
BOX = "#ffffff"
FRAME = "#f3f1ee"


def box(ax, x, y, w, h, text, edge, size=8):
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
        fontsize=size,
        color=INK,
        linespacing=1.25,
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


def draw_session(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.15)
    ax.set_axis_off()
    ax.set_title(
        "One session  ·  events in order, state for this thread",
        loc="left",
        fontsize=11,
        color=NAVY,
        pad=6,
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.12, 0.42),
            11.76,
            2.15,
            boxstyle="round,pad=0.012,rounding_size=0.1",
            facecolor=FRAME,
            edgecolor="#e7e5e4",
            linewidth=1,
        )
    )
    box(ax, 0.35, 0.78, 2.55, 1.45, "id\napp_name\nuser_id\nlast_update_time", NAVY, 8)
    xs = [3.2, 4.55, 5.9, 7.25]
    for x, name in zip(xs, ["A", "B", "C", "D"]):
        box(ax, x, 1.05, 1.1, 0.9, name, NAVY, 10)
    for i in range(3):
        arrow(ax, (xs[i] + 1.1, 1.5), (xs[i + 1], 1.5), NAVY)
    box(ax, 8.7, 0.78, 2.9, 1.45, "state\nprofile, recipes", TEAL, 8)
    ax.text(5.75, 0.58, "events", ha="center", va="center", fontsize=7, color=MUTED)


def draw_stores(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()
    ax.set_title(
        "Same user, next process  ·  RAM drops the thread, SQL reloads it",
        loc="left",
        fontsize=11,
        color=TEAL,
        pad=6,
    )

    ax.text(2.9, 4.05, "InMemorySessionService", ha="center", va="center", fontsize=9, color=NAVY)
    box(ax, 1.35, 2.55, 3.1, 1.15, "session A\nstate: James", NAVY, 8)
    box(ax, 1.35, 0.7, 3.1, 1.15, "session B\nstate starts empty", NAVY, 8)
    arrow(ax, (2.9, 2.55), (2.9, 1.85), NAVY)
    ax.text(4.7, 2.15, "process\nexits", ha="left", va="center", fontsize=7, color=MUTED)

    ax.text(9.05, 4.05, "DatabaseSessionService", ha="center", va="center", fontsize=9, color=TEAL)
    box(ax, 6.15, 2.55, 2.35, 1.15, "session A\nrecipes", TEAL, 8)
    box(ax, 8.75, 2.55, 2.7, 1.15, "recipe_data.db\nsessions, events", TEAL, 8)
    box(ax, 7.45, 0.7, 2.7, 1.15, "same session A\nchef, recipes", TEAL, 8)
    arrow(ax, (8.5, 3.12), (8.75, 3.12), TEAL)
    arrow(ax, (10.1, 2.55), (9.2, 1.85), TEAL)


def main():
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 7.0), dpi=140, gridspec_kw={"height_ratios": [1, 1.45]}
    )
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
    draw_session(axes[0])
    draw_stores(axes[1])
    fig.text(
        0.01,
        0.01,
        "Toy of one thread and two stores. Names are the module examples.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
