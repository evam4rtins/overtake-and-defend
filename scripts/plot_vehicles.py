"""

Plot the 2 vehicle models in 3 different scenarios:
1. constant lateral acceleration (point mass) and constant steering angle (bicycle) at 2 speeds
2. braking from 60 m/s with a constant deceleration
3. gg diagram of braking while turning

"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.types import VehicleParams, VehicleState
from overtake.vehicles import KinematicBicycle, PointMass

def simulate(model, state, action, dt, steps):
    states = [state]
    for _ in range(steps):
        state = model.step(state, np.asarray(action, dtype=float), dt)
        states.append(state)
    return states

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="save the figure to this file")
    args = parser.parse_args()

    p = VehicleParams()
    dt = 0.01
    fig, (ax_xy, ax_v, ax_gg) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # circles
    for v in (10.0, 40.0):
        pm = simulate(PointMass(p), VehicleState(0, 0, 0, v), [0.0, 10.0], dt, 1500)
        bi = simulate(KinematicBicycle(p), VehicleState(0, 0, 0, v), [0.15, 0.0], dt, 1500)
        ax_xy.plot([s.x for s in pm], [s.y for s in pm], label=f"point mass, a_y = 10, v = {v:g} m/s")
        ax_xy.plot([s.x for s in bi], [s.y for s in bi], label=f"bicycle, delta = 0.15 rad, v = {v:g} m/s")
    ax_xy.set(xlabel="x [m]", ylabel="y [m]", title="Constant input", aspect="equal")
    ax_xy.legend(fontsize=8)

    # braking
    for model, action, name in (
        (PointMass(p), [-50.0, 0.0], "point mass"),
        (KinematicBicycle(p), [0.0, -50.0], "bicycle"),
    ):
        states = simulate(model, VehicleState(0, 0, 0, 60.0), action, dt, 500)
        t = np.arange(len(states)) * dt
        ax_v.plot(t, [s.v for s in states], label=name)
    ax_v.set(xlabel="t [s]", ylabel="v [m/s]", title="Full braking (a_x_max = {p.a_x_max:g} m/s²)")
    ax_v.legend(fontsize=8)

    # g-g diagram: break while steering harder and harder
    model, state, ax_s, ay_s = KinematicBicycle(p), VehicleState(0, 0, 0, 50.0), [], []
    for k in range(300):
        state = model.step(state, np.array([p.delta_max * k / 300, -20.0]), dt)
        ax_s.append(state.a_x)
        ay_s.append(state.a_y)
    th = np.linspace(0, 2 * np.pi, 200)
    ax_gg.plot(p.a_y_max * np.cos(th), p.a_x_max * np.sin(th), "k", lw=1, label="friction ellipse")
    ax_gg.plot(ay_s, ax_s, ".", ms=3, label="bicycle")
    ax_gg.set(xlabel="a_y [m/s²]", ylabel="a_x [m/s²]", title="G-G diagram", aspect="equal")
    ax_gg.legend(fontsize=8)

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.out}")

if __name__ == "__main__":
    main()