"""Draw the README's training-architecture diagram (docs/images/training_pipeline.png) in the style of the data
pipeline diagram: white-outlined boxes on black, joined only by straight horizontal and vertical lines.

    python scripts/draw_training_diagram.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

W, H, BW, BH = 1590, 380, 230, 64
# Box centres (pixels, y down) and labels.
BOXES = {
    "data": (150, 210, "Expert dataset\n(robot state, perceived objects,\nexpert action)"),
    "imit": (510, 110, "Imitation\nchunked MLP (cups) / BC (throw)"),
    "dp": (510, 310, "Diffusion Policy\n(1D U-Net over action chunks)"),
    "ppo": (830, 110, "RL fine-tuning: PPO\n+ fading BC term"),
    "place": (1160, 150, "Place policy\n(cup A, cup B, top cup)"),
    "throw": (1160, 270, "Throw policy"),
    "full": (1460, 210, "Whole task:\nplace, then throw"),
}
WHITE = "white"


def edge(name, side):
    x, y, _ = BOXES[name]
    return {"l": (x - BW / 2, y), "r": (x + BW / 2, y), "t": (x, y - BH / 2), "b": (x, y + BH / 2)}[side]


def path(ax, pts, arrow=True, dashed=False):
    """Straight segments through `pts`, with an arrowhead at the end."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=WHITE, lw=1, ls=(0, (4, 3)) if dashed else "-", solid_capstyle="butt")
    if arrow:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x1 - 0.01 * (x1 - x0), y1 - 0.01 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1, mutation_scale=10))


def main(out="docs/images/training_pipeline.png"):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(H, 0), ax.axis("off")
    text = dict(color=WHITE, ha="center", va="center", fontsize=8.5, family="DejaVu Sans")
    for x, y, label in BOXES.values():
        ax.add_patch(Rectangle((x - BW / 2, y - BH / 2), BW, BH, fc="black", ec=WHITE, lw=1))
        ax.text(x, y, label, **text)

    # Dataset -> imitation and diffusion (one split).
    split = 330
    path(ax, [edge("data", "r"), (split, 210)], arrow=False)
    path(ax, [(split, 210), (split, 110), edge("imit", "l")])
    path(ax, [(split, 210), (split, 310), edge("dp", "l")])
    # DAgger: the expert labels the learner's runs, back into the dataset.
    path(ax, [edge("imit", "t"), (510, 30), (150, 30), edge("data", "t")], dashed=True)
    ax.text(330, 20, "DAgger: the expert labels the learner's runs", **text)
    # Imitation -> PPO.
    path(ax, [edge("imit", "r"), edge("ppo", "l")])
    # Every trained policy (imitation, PPO, diffusion) can serve as the place or the throw policy.
    bus = 995
    path(ax, [edge("ppo", "r"), (bus, 110)], arrow=False)
    path(ax, [(510, 110 + BH / 2), (510, 210), (bus, 210)], arrow=False)
    path(ax, [edge("dp", "r"), (bus, 310)], arrow=False)
    path(ax, [(bus, 110), (bus, 310)], arrow=False)
    path(ax, [(bus, 150), edge("place", "l")])
    path(ax, [(bus, 270), edge("throw", "l")])
    ax.text(bus - 120, 225, "any trained policy", **text)
    # Place and throw -> whole task.
    join = 1310
    path(ax, [edge("place", "r"), (join, 150), (join, 270), edge("throw", "r")], arrow=False)
    path(ax, [(join, 210), edge("full", "l")])
    fig.savefig(out, dpi=100, facecolor="black")


if __name__ == "__main__":
    main()
