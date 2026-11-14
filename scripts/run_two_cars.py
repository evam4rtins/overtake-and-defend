"""
Two cars on their racing line: the follower uses the ACC.

    python scripts/run_two_cars.py                          # identical cars, 3 m gap
    python scripts/run_two_cars.py --gap 10 --model bicycle
    python scripts/run_two_cars.py --out results/two_cars.png

Shows the trajectories on the track, the speed along the track and the gap between the cars,
and prints the outcome of the success criterion.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.agents import RacingLineAgent
from overtake.racing_line import racing_line
from overtake.sim import Car, Simulator, outcome, start_state
from overtake.track import CornerTrack
from overtake.types import VehicleParams
from overtake.vehicles import KinematicBicycle, PointMass

MODELS = {"point_mass": PointMass, "bicycle": KinematicBicycle}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap", type=float, default=3.0, help="initial bumper-to-bumper gap [m]")
    parser.add_argument("--v0", type=float, default=50.0, help="initial speed of both cars [m/s]")
    parser.add_argument("--model", choices=MODELS, default="point_mass")
    parser.add_argument("--out", type=Path, default=None, help="save the figure to this file")
    args = parser.parse_args()

    track = CornerTrack(l_in=250, radius=40, angle_deg=90, l_out=150, width=12)
    params = VehicleParams()
    plan = racing_line(track, params, args.v0)
    Model = MODELS[args.model]
    s_opp = 60.0
    s_ego = s_opp - args.gap - params.length
    cars = [Car("opponent", params, Model(params), RacingLineAgent(), start_state(track, s_opp, plan.d_at(s_opp), args.v0), plan),
            Car("ego", params, Model(params), RacingLineAgent(), start_state(track, s_ego, plan.d_at(s_ego), args.v0), plan)]
    result = Simulator(track, cars).run()
    print(outcome(result))

    fig, (ax_map, ax_v, ax_gap) = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    left, right = track.bounds()
    ax_map.plot(*left, "k", lw=1)
    ax_map.plot(*right, "k", lw=1)
    ax_map.plot(*track.from_frenet(track.s, plan.d), ":", color="gray", label="racing line")
    for name, log in result.cars.items():
        ax_map.plot(log["x"], log["y"], label=name)
    ax_map.set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="Trajectories")
    ax_map.legend()

    ax_v.plot(plan.s, plan.trajectory.v, ":", color="gray", label="racing line (plan)")
    for name, log in result.cars.items():
        ax_v.plot(log["s"], log["v"], label=name)
    ax_v.set(xlabel="s [m]", ylabel="v [m/s]", title="Speed along the track")
    ax_v.legend()

    e, o = result.cars["ego"], result.cars["opponent"]
    gap = o["s"] - e["s"] - params.length
    ax_gap.plot(e["t"], gap, label="gap (bumper to bumper)")
    ax_gap.plot(e["t"], 0.6 * e["v"] + 5.0, "--", label="ACC desired gap")
    ax_gap.set(xlabel="t [s]", ylabel="[m]", title=f"Gap, min clearance = {result.min_clearance:.2f} m")
    ax_gap.legend()

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
