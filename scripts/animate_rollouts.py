"""Side-view animation + time plots of recorded rollouts (from scripts/record_rollout.py).

Each panel shows the wall, the floor, and the robot from the side: blue body, black wheels, and for each propeller
(front = orange, rear = green) a disc and an ARROW = the force that propeller puts on the robot (length = thrust).

Usage:
    uv run python scripts/animate_rollouts.py videos/data/A videos/data/B videos/data/C \
        --labels "A: sees everything" "B: IMU + IR" "C: IMU + IR + encoders" --out videos/side_view
Writes <out>.mp4 (animation) and <out>_plots.png (height, push direction and thrust over time).
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle  # noqa: E402

# robot geometry (must match scripts/make_vertigo_urdf.py)
BODY_L, BODY_T = 0.36, 0.06
WHEEL_R, WHEEL_X = 0.06, 0.15
PROP_X, PROP_Z, PROP_R = 0.11, 0.07, 0.09
ARROW_LEN = 0.35  # [m] arrow length at full thrust
COLORS = {"front": "#e8743b", "rear": "#2fae5a", "body": "#2f5fd0"}
STARTS = (("floor", "start on floor"), ("wall", "start on wall (climb, then hold)"))


def unit(a):
    return np.array([math.cos(a), math.sin(a)])


def push_angle_deg(tilt):
    """Push direction relative to the robot: 0 = into the surface (belly side), +90 = forward, 180 = away."""
    return np.degrees(np.arctan2(np.sin(-tilt), np.cos(-tilt)))


class Panel:
    def __init__(self, ax, data, title):
        self.ax, self.d = ax, data
        wx = float(data["wall_x"])
        ax.set_xlim(wx - 1.5, wx + 0.35)
        ax.set_ylim(-0.08, 3.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_ylabel("height [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.add_patch(Rectangle((wx, 0), 0.2, 3.0, color="#cfcfc8", zorder=0))
        ax.add_patch(Rectangle((wx - 2, -0.1), 2.6, 0.1, color="#9aa0a6", zorder=0))
        ax.set_title(title, fontsize=9, loc="left")
        (self.trail,) = ax.plot([], [], color=COLORS["body"], alpha=0.25, lw=1)
        self.body = Polygon(np.zeros((4, 2)), closed=True, color=COLORS["body"], zorder=3)
        ax.add_patch(self.body)
        self.wheels = [Circle((0, 0), WHEEL_R, color="black", zorder=4) for _ in range(2)]
        for w in self.wheels:
            ax.add_patch(w)
        (self.disc_f,) = ax.plot([], [], color=COLORS["front"], lw=3, zorder=5)
        (self.disc_r,) = ax.plot([], [], color=COLORS["rear"], lw=3, zorder=5)
        self.arrows = []
        self.text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    def draw(self, i):
        d = self.d
        n = len(d["x"])
        k = min(i, n - 1)
        c = np.array([d["x"][k], d["z"][k]])
        fx, up = unit(d["pitch"][k]), unit(d["up"][k])
        hl, ht = BODY_L / 2, BODY_T / 2
        self.body.set_xy([c + sx * hl * fx + sz * ht * up for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        for w, sx in zip(self.wheels, (1, -1)):
            w.center = tuple(c + sx * WHEEL_X * fx)
        self.trail.set_data(d["x"][: k + 1], d["z"][: k + 1])
        for a in self.arrows:
            a.remove()
        self.arrows = []
        for j, (sx, disc, col) in enumerate(((1, self.disc_f, COLORS["front"]), (-1, self.disc_r, COLORS["rear"]))):
            p = c + sx * PROP_X * fx + PROP_Z * up
            t = d["tilt"][k][j]
            axis = math.sin(t) * fx + math.cos(t) * up  # direction the air is blown
            plane = math.cos(t) * fx - math.sin(t) * up
            disc.set_data([p[0] - PROP_R * plane[0], p[0] + PROP_R * plane[0]],
                          [p[1] - PROP_R * plane[1], p[1] + PROP_R * plane[1]])
            force = -axis * ARROW_LEN * float(d["thrust"][k][j])
            if np.linalg.norm(force) > 0.01:
                arrow = FancyArrow(p[0], p[1], force[0], force[1], width=0.012, head_width=0.05,
                                   head_length=0.05, length_includes_head=True, color=col, zorder=6)
                self.ax.add_patch(arrow)
                self.arrows.append(arrow)
        if d["on_wall"][k] == 4:
            facing = "belly to wall" if up[0] < -0.5 else ("BACK to wall (props on wall!)" if up[0] > 0.5 else "")
            where = f"on wall, {facing}"
        elif d["on_floor"][k] > 0 and d["on_wall"][k] == 0:
            where = "on floor"
        else:
            where = "flipping / partly on wall"
        status = "ENDED: fell or flew off" if (i >= n - 1 and bool(d["terminated"])) else where
        cmd = "HOLD" if d["command"][k] > 0 else "CLIMB"
        self.text.set_text(f"t={k * float(d['dt']):4.1f}s  cmd={cmd}\n{status}")
        return [self.body, self.trail, self.disc_f, self.disc_r, self.text, *self.wheels, *self.arrows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Folders written by record_rollout.py")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default="videos/side_view")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    data = {(r, s): dict(np.load(Path(r) / f"{s}.npz")) for r in args.runs for s, _ in STARTS}

    # ---- animation
    cols = len(args.runs)
    fig, axes = plt.subplots(2, cols, figsize=(4.2 * cols, 11), squeeze=False)
    fig.suptitle("Side view.  Arrows = force each propeller puts on the robot (orange = front, green = rear; "
                 "length = thrust)", fontsize=10)
    panels = [Panel(axes[row][col], data[(r, s)], f"{lab}  |  {stitle}")
              for row, (s, stitle) in enumerate(STARTS) for col, (r, lab) in enumerate(zip(args.runs, labels))]
    dt = float(next(iter(data.values()))["dt"])
    steps = max(len(d["x"]) for d in data.values())
    stride = max(1, round(1 / (args.fps * dt)))  # real time
    frames = list(range(0, steps, stride))
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def update(i):
        out = []
        for p in panels:
            out += p.draw(i)
        return out

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out.with_suffix(".mp4"), writer=FFMpegWriter(fps=args.fps, bitrate=2500))
    plt.close(fig)

    # ---- time plots
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    styles = ["-", "--", ":"]
    for col, (s, stitle) in enumerate(STARTS):
        for (r, lab), ls in zip(zip(args.runs, labels), styles):
            d = data[(r, s)]
            t = np.arange(len(d["x"])) * dt
            axes[0][col].plot(t, d["z"], ls, color="k", label=lab)
            hold = np.flatnonzero(d["command"] > 0)
            if len(hold) and col == 1:
                axes[0][col].axvline(t[hold[0]], color="gray", ls=ls, lw=0.8)
            if bool(d["terminated"]):
                axes[0][col].plot(t[-1], d["z"][-1], "rx", ms=9)
            push = push_angle_deg(d["tilt"])
            axes[1][col].plot(t, push[:, 0], ls, color=COLORS["front"])
            axes[1][col].plot(t, push[:, 1], ls, color=COLORS["rear"])
            axes[2][col].plot(t, 100 * d["thrust"][:, 0], ls, color=COLORS["front"])
            axes[2][col].plot(t, 100 * d["thrust"][:, 1], ls, color=COLORS["rear"])
        axes[0][col].set_title(stitle)
    axes[0][0].set_ylabel("body height [m]")
    axes[1][0].set_ylabel("push direction [deg]\n0 = into surface, +90 = forward/up")
    axes[2][0].set_ylabel("thrust [% of max]")
    for a in axes[1]:
        a.set_yticks([-180, -90, 0, 90, 180])
        a.axhline(0, color="gray", lw=0.5)
    for a in axes[2]:
        a.set_xlabel("time [s]")
    axes[0][0].legend(fontsize=8)
    axes[0][1].text(0.99, 0.02, "grey line = hold command issued;  red x = episode ended (fell/flew)",
                    transform=axes[0][1].transAxes, ha="right", fontsize=8, color="gray")
    axes[1][1].text(0.99, 0.02, "orange = front prop, green = rear prop", transform=axes[1][1].transAxes,
                    ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(f"{out}_plots.png", dpi=110)
    print(f"wrote {out.with_suffix('.mp4')} and {out}_plots.png")


if __name__ == "__main__":
    main()
