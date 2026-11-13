"""
Plot the three passes of the speed profile on the centreline of the corner track.

    python scripts/plot_speed_profile.py                          # hairpin (R = 15 m, 180 deg)
    python scripts/plot_speed_profile.py --radius 40 --angle 90   # 90 deg corner
    python scripts/plot_speed_profile.py --out results/speed_profile.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.speed_profile import accelerations, three_pass, traversal_time
from overtake.track import CornerTrack
from overtake.types import VehicleParams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l-in", type=float, default=250.0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--angle", type=float, default=180.0, help="corner angle [deg], positive = left")
    parser.add_argument("--l-out", type=float, default=150.0)
    parser.add_argument("--v-start", type=float, default=60.0, help="speed at the start of the segment [m/s]")
    parser.add_argument("--out", type=Path, default=None, help="save the figure to this file")
    args = parser.parse_args()

    p = VehicleParams()
    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out)
    guess, forward, final = three_pass(track.s, track.kappa, p, args.v_start, return_passes=True)
    a_x, a_y = accelerations(track.s, final, track.kappa)

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax in (ax_v, ax_a):
        ax.axvspan(track.corner_start, track.corner_end, color="0.9", label="corner")

    ax_v.plot(track.s, guess, color="tab:blue", label="1. initial guess (lateral grip, top speed)")
    ax_v.plot(track.s, forward, color="tab:red", label="2. forward pass (acceleration)")
    ax_v.plot(track.s, final, color="tab:orange", lw=2.5, label="3. backward pass (braking) = final")
    ax_v.set(ylabel="speed [m/s]", ylim=(0, 1.1 * p.v_max),
             title=f"Three-pass speed profile on the centreline: R = {track.radius:g} m, "
                   f"{track.angle_deg:g} deg, v_start = {args.v_start:g} m/s, "
                   f"time = {traversal_time(track.s, final):.2f} s")
    ax_v.legend(loc="lower left")

    ax_a.plot(track.s, a_x, color="tab:green", label="longitudinal a_x")
    ax_a.plot(track.s, a_y, color="tab:purple", label="lateral a_y")
    ax_a.plot(track.s, p.a_engine * (1 - final / p.v_max), ":", color="tab:green", label="engine limit")
    for limit in (-p.a_x_max, p.a_y_max):
        ax_a.axhline(limit, color="0.5", lw=0.8, ls="--")
    ax_a.set(xlabel="s [m]", ylabel="acceleration [m/s²]")
    ax_a.legend(loc="lower left")

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
