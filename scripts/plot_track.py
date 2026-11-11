"""
Plot the corner track with its bounds, and check the geometry helpers on its centreline (T03, T04).

    python scripts/plot_track.py                          # hairpin (R = 15 m, 180 deg)
    python scripts/plot_track.py --radius 40 --angle 90   # 90 deg corner
    python scripts/plot_track.py --out results/track.png  # save instead of showing
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.track import CornerTrack, curvature, heading
from overtake.types import VehicleParams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l-in", type=float, default=250.0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--angle", type=float, default=180.0, help="corner angle [deg], positive = left")
    parser.add_argument("--l-out", type=float, default=150.0)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--margin", type=float, default=0.3, help="safety margin to the track edge [m]")
    parser.add_argument("--out", type=Path, default=None, help="save the figure to this file")
    args = parser.parse_args()

    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out, args.width)
    d_min, d_max = track.d_limits(VehicleParams().width, args.margin)
    fig, (ax_map, ax_psi, ax_kappa) = plt.subplots(
        3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [3, 1, 1]}, constrained_layout=True)

    # Track: edges, centreline, limits of the car centre, start and end of the arc.
    left, right = track.bounds()
    ax_map.plot(*left, "k", label="track edges")
    ax_map.plot(*right, "k")
    ax_map.plot(track.cx, track.cy, "--", color="tab:orange", label="centreline (d = 0)")
    ax_map.plot(*track.from_frenet(track.s, d_max), ":", color="tab:blue", label="limits of the car centre")
    ax_map.plot(*track.from_frenet(track.s, d_min), ":", color="tab:blue")
    for s, name in ((track.corner_start, "corner start"), (track.corner_end, "corner end")):
        (xl, yl), (xr, yr) = track.from_frenet(s, track.width / 2), track.from_frenet(s, -track.width / 2)
        ax_map.plot([xl, xr], [yl, yr], color="tab:gray", lw=1)
        ax_map.annotate(name, (xr, yr), textcoords="offset points", xytext=(6, -12), color="tab:gray")
    # Frenet frame at one station of the arc: tangent and left normal.
    x0, y0, psi0, _ = track.at_s((track.corner_start + track.corner_end) / 2)
    ax_map.quiver([x0, x0], [y0, y0], [np.cos(psi0), -np.sin(psi0)], [np.sin(psi0), np.cos(psi0)],
                  color=["tab:red", "tab:green"], scale=12, width=0.005, zorder=3)
    ax_map.plot([], [], color="tab:red", label="tangent (s grows)")
    ax_map.plot([], [], color="tab:green", label="left normal (d > 0)")
    ax_map.set(xlabel="x [m]", ylabel="y [m]", aspect="equal",
               title=f"Corner track: R = {track.radius:g} m, angle = {track.angle_deg:g} deg, "
                     f"width = {track.width:g} m, length = {track.length:.1f} m")
    ax_map.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # Heading and curvature: exact values against the helpers applied to the sampled centreline.
    ax_psi.plot(track.s, np.rad2deg(track.psi), label="exact")
    ax_psi.plot(track.s, np.rad2deg(heading(track.cx, track.cy)), "--", label="heading(x, y)")
    ax_psi.set(xlabel="s [m]", ylabel="heading [deg]")
    ax_psi.legend()
    ax_kappa.plot(track.s, track.kappa, label="exact")
    ax_kappa.plot(track.s, curvature(track.cx, track.cy), "--", label="curvature(x, y) = 4A / (abc)")
    ax_kappa.set(xlabel="s [m]", ylabel="curvature [1/m]")
    ax_kappa.legend()

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
