"""
Plot the racing line of the corner track, and compare the lines of different cars.

Figure 1: minimum-length, minimum-curvature and optimal lines; segment time against the weight
          ratio; speed profiles; g-g diagram of the optimal line.
Figure 2: optimal lines of a baseline, a grip-limited and a power-limited car.

    python scripts/plot_racing_line.py                          # hairpin (R = 15 m, 180 deg)
    python scripts/plot_racing_line.py --radius 40 --angle 90   # 90 deg corner
    python scripts/plot_racing_line.py --out results/racing_line.png   # also saves ..._cars.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.racing_line import centerline_plan, racing_line
from overtake.speed_profile import accelerations
from overtake.track import CornerTrack, arc_length, curvature
from overtake.types import VehicleParams


def draw_track(ax, track, zoom=60.0):
    """ Track edges, zoomed on the corner. """
    left, right = track.bounds()
    ax.plot(*left, "k", lw=1)
    ax.plot(*right, "k", lw=1)
    near = (track.s > track.corner_start - zoom) & (track.s < track.corner_end + zoom)
    xs, ys = np.concatenate((left[0][near], right[0][near])), np.concatenate((left[1][near], right[1][near]))
    ax.set(xlim=(xs.min() - 2, xs.max() + 2), ylim=(ys.min() - 2, ys.max() + 2),
           aspect="equal", xlabel="x [m]", ylabel="y [m]")


def path_accelerations(track, plan):
    """ Accelerations (a_x, a_y) along the line of a plan. """
    x, y = track.from_frenet(track.s, plan.d)
    return accelerations(arc_length(x, y), plan.trajectory.v, curvature(x, y))


def figure_lines(track, p, v_start, margin):
    """ Figure 1: the three lines of Lecture 7 and the weight search. """
    plans = {
        "centreline": (centerline_plan(track, p, v_start), "0.6", "--"),
        "minimum length": (racing_line(track, p, v_start, ratios=(np.inf,), margin=margin), "tab:green", "-"),
        "minimum curvature": (racing_line(track, p, v_start, ratios=(0.0,), margin=margin), "tab:blue", "-"),
        "optimal": (racing_line(track, p, v_start, margin=margin), "tab:red", "-"),
    }
    optimal = plans["optimal"][0]
    fig, ((ax_map, ax_time), (ax_v, ax_gg)) = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    draw_track(ax_map, track)
    for name, (plan, color, style) in plans.items():
        ax_map.plot(plan.trajectory.x, plan.trajectory.y, style, color=color, label=f"{name}: {plan.time:.2f} s")
        ax_v.plot(track.s, plan.trajectory.v, style, color=color, label=name)
    ax_map.set_title("Lines in the corner")
    ax_map.legend(loc="best")
    ax_v.axvspan(track.corner_start, track.corner_end, color="0.92")
    ax_v.set(xlabel="s along the centreline [m]", ylabel="speed [m/s]", title="Speed profiles")
    ax_v.legend(loc="lower left")

    ratios = np.array(sorted(r for r in optimal.candidates if r > 0))
    ax_time.semilogx(ratios, [optimal.candidates[r] for r in ratios], "o-", color="tab:red", label="tried ratios")
    ax_time.axhline(plans["minimum curvature"][0].time, color="tab:blue", ls=":", label="minimum curvature (ratio 0)")
    ax_time.axhline(plans["minimum length"][0].time, color="tab:green", ls=":", label="minimum length (ratio inf)")
    ax_time.plot(optimal.weight_ratio, optimal.time, "*", color="k", ms=14, zorder=3,
                 label=f"selected: {optimal.weight_ratio:.2g}")
    ax_time.set(xlabel="weight ratio k_l / k_rho", ylabel="segment time [s]", title="Outer search over the weights")
    ax_time.legend(loc="best")

    a_x, a_y = path_accelerations(track, optimal)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_gg.plot(p.a_y_max * np.cos(theta), p.a_x_max * np.sin(theta), "k", lw=1, label="friction ellipse")
    ax_gg.plot(a_y, a_x, ".", ms=3, color="tab:red", label="optimal line")
    ax_gg.set(xlabel="a_y [m/s²]", ylabel="a_x [m/s²]", aspect="equal", title="g-g diagram of the optimal line")
    ax_gg.legend(loc="upper right")
    fig.suptitle(f"Racing line: R = {track.radius:g} m, {track.angle_deg:g} deg, v_start = {v_start:g} m/s")
    return fig


def figure_cars(track, p, v_start, margin):
    """ Figure 2: does the optimal line depend on the car? """
    cars = {
        "baseline": (p, "tab:red"),
        "grip-limited (grip x 0.85)": (p.scaled(a_y_max=0.85, a_x_max=0.85), "tab:blue"),
        "power-limited (engine x 0.75)": (p.scaled(a_engine=0.75), "tab:green"),
    }
    fig, (ax_map, ax_d) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    draw_track(ax_map, track)
    reference = None
    print(f"{'car':32s} {'ratio':>9s} {'time [s]':>9s} {'v min':>7s} {'max |d - baseline| [m]':>24s}")
    for name, (params, color) in cars.items():
        plan = racing_line(track, params, v_start, margin=margin)
        reference = plan if reference is None else reference
        shift = np.abs(plan.d - reference.d).max()
        print(f"{name:32s} {plan.weight_ratio:9.2g} {plan.time:9.3f} {plan.trajectory.v.min():7.2f} {shift:24.2f}")
        label = f"{name}: ratio {plan.weight_ratio:.2g}, {plan.time:.2f} s"
        ax_map.plot(plan.trajectory.x, plan.trajectory.y, color=color, label=label)
        ax_d.plot(track.s, plan.d, color=color, label=name)
    ax_map.set_title("Optimal line of each car")
    ax_map.legend(loc="best")
    ax_d.axvspan(track.corner_start, track.corner_end, color="0.92")
    ax_d.set(xlabel="s along the centreline [m]", ylabel="lateral offset d [m] (> 0 = left)",
             title="Lateral position along the track")
    ax_d.legend(loc="best")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l-in", type=float, default=250.0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--angle", type=float, default=180.0, help="corner angle [deg], positive = left")
    parser.add_argument("--l-out", type=float, default=150.0)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--margin", type=float, default=0.3, help="safety margin to the track edge [m]")
    parser.add_argument("--v-start", type=float, default=60.0, help="speed at the start of the segment [m/s]")
    parser.add_argument("--out", type=Path, default=None, help="save the figures (second one gets '_cars')")
    args = parser.parse_args()

    p = VehicleParams()
    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out, args.width)
    figures = {"": figure_lines(track, p, args.v_start, args.margin),
               "_cars": figure_cars(track, p, args.v_start, args.margin)}
    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for suffix, fig in figures.items():
            path = args.out.with_name(args.out.stem + suffix + args.out.suffix)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
