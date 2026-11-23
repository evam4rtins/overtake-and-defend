"""
Ego with a local planner against an opponent on its racing line.

    python scripts/run_planner.py                                   # S0: identical cars, sigmoid planner
    python scripts/run_planner.py --opponent-scale 0.6 --l-out 300 --s-opp 40 --v0 40
    python scripts/run_planner.py --brakes 2.0 --gap 2              # braking advantage only
    python scripts/run_planner.py --planner follow --model bicycle --predictor coordinated_turn
    python scripts/run_planner.py --out results/planner.png         # save the figures

Figure 1: trajectories (with the planning steps where the ego decided to overtake), speeds along
the track, lateral offsets, gap between the cars and the decisions over time.
Figure 2 (sigmoid planner): the candidates of the first planning step that chose to overtake,
feasible or not, and the chosen one.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.agents import PlannerAgent, RacingLineAgent
from overtake.planners import PLANNERS, SigmoidPlanner
from overtake.prediction import PREDICTORS
from overtake.racing_line import racing_line
from overtake.sim import Car, Simulator, outcome, start_state
from overtake.track import CornerTrack
from overtake.types import VehicleParams
from overtake.vehicles import KinematicBicycle, PointMass

MODELS = {"point_mass": PointMass, "bicycle": KinematicBicycle}


class RecordingSigmoid(SigmoidPlanner):
    """ Sigmoid planner that keeps the candidates of its first decision to overtake. """

    def plan(self, obs, predictions):
        traj = super().plan(obs, predictions)
        if self.decision.kind == "overtake" and not hasattr(self, "first_overtake"):
            self.first_overtake = dict(self.last, t=obs.t)
        return traj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius", type=float, default=40.0)
    parser.add_argument("--angle", type=float, default=90.0, help="corner angle [deg], positive = left")
    parser.add_argument("--l-in", type=float, default=250.0)
    parser.add_argument("--l-out", type=float, default=150.0)
    parser.add_argument("--planner", choices=PLANNERS, default="sigmoid")
    parser.add_argument("--model", choices=MODELS, default="point_mass")
    parser.add_argument("--predictor", choices=PREDICTORS, default="exact")
    parser.add_argument("--brakes", type=float, default=1.0, help="braking grip ratio of the ego")
    parser.add_argument("--grip", type=float, default=1.0, help="lateral grip ratio of the ego")
    parser.add_argument("--power", type=float, default=1.0, help="engine ratio of the ego")
    parser.add_argument("--opponent-scale", type=float, default=1.0, help="all opponent parameters x this ratio")
    parser.add_argument("--gap", type=float, default=5.0, help="initial bumper-to-bumper gap [m]")
    parser.add_argument("--s-opp", type=float, default=80.0, help="initial station of the opponent [m]")
    parser.add_argument("--v0", type=float, default=50.0, help="entry speed used for the racing lines [m/s]")
    parser.add_argument("--out", type=Path, default=None, help="save the figures (second one gets '_candidates')")
    args = parser.parse_args()

    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out)
    base = VehicleParams()
    r = args.opponent_scale
    opp_params = base.scaled(a_y_max=r, a_x_max=r, a_engine=r, v_max=r)
    ego_params = base.scaled(a_x_max=args.brakes, a_y_max=args.grip, a_engine=args.power)
    plan_o, plan_e = racing_line(track, opp_params, args.v0), racing_line(track, ego_params, args.v0)
    s_ego = args.s_opp - args.gap - ego_params.length
    Model = MODELS[args.model]
    planner = RecordingSigmoid() if args.planner == "sigmoid" else PLANNERS[args.planner]()
    agent = PlannerAgent(planner, predictor=PREDICTORS[args.predictor](horizon=10.0))
    cars = [Car("opponent", opp_params, Model(opp_params), RacingLineAgent(),
                start_state(track, args.s_opp, plan_o.d_at(args.s_opp), float(np.interp(args.s_opp, plan_o.s, plan_o.trajectory.v))), plan_o),
            Car("ego", ego_params, Model(ego_params), agent,
                start_state(track, s_ego, plan_e.d_at(s_ego), float(np.interp(s_ego, plan_e.s, plan_e.trajectory.v))), plan_e)]
    result = Simulator(track, cars).run()
    o = outcome(result)
    print(o)
    kinds = [d.kind for _, d in agent.decisions]
    print(f"{len(kinds)} planning steps, {kinds.count('overtake')} chose to overtake")

    e, op = result.cars["ego"], result.cars["opponent"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax_map, ax_v, ax_d, ax_gap, ax_dec, ax_info = axes.ravel()
    left, right = track.bounds()
    ax_map.plot(*left, "k", lw=1)
    ax_map.plot(*right, "k", lw=1)
    ax_map.plot(*track.from_frenet(track.s, plan_e.d), ":", color="gray", label="ego racing line")
    ax_map.plot(op["x"], op["y"], color="tab:blue", label="opponent")
    ax_map.plot(e["x"], e["y"], color="tab:red", label="ego")
    t_ovt = [t for t, d in agent.decisions if d.kind == "overtake"]
    k_ovt = np.searchsorted(e["t"], t_ovt)
    ax_map.plot(e["x"][k_ovt], e["y"][k_ovt], ".", color="tab:red", ms=4, label="ego decides to overtake")
    ax_map.set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="Trajectories")
    ax_map.legend(fontsize=8)

    ax_v.plot(plan_e.s, plan_e.trajectory.v, ":", color="gray", label="ego racing line")
    ax_v.plot(op["s"], op["v"], color="tab:blue", label="opponent")
    ax_v.plot(e["s"], e["v"], color="tab:red", label="ego")
    ax_v.set(xlabel="s [m]", ylabel="v [m/s]", title="Speed along the track")
    ax_v.legend(fontsize=8)

    ax_d.plot(op["s"], op["d"], color="tab:blue", label="opponent")
    ax_d.plot(e["s"], e["d"], color="tab:red", label="ego")
    ax_d.plot(plan_e.s, plan_e.d, ":", color="gray", label="ego racing line")
    d_min, d_max = track.d_limits(ego_params.width, 0.3)
    ax_d.axhline(d_min, color="k", lw=0.8)
    ax_d.axhline(d_max, color="k", lw=0.8)
    ax_d.axvspan(track.corner_start, track.corner_end, color="0.92")
    ax_d.set(xlabel="s [m]", ylabel="d [m] (> 0 = left)", title="Lateral position")
    ax_d.legend(fontsize=8)

    ax_gap.plot(e["t"], e["s"] - op["s"], color="tab:purple", label="s_ego - s_opponent")
    ax_gap.axhline(0.0, color="k", lw=0.8)
    ax_gap.set(xlabel="t [s]", ylabel="[m]", title=f"Gap along the track, min clearance {result.min_clearance:.2f} m")
    ax_gap.legend(fontsize=8)

    t_dec = np.array([t for t, _ in agent.decisions])
    ax_dec.step(t_dec, [1 if d.kind == "overtake" else 0 for _, d in agent.decisions], where="post", color="tab:red")
    ax_dec.set(xlabel="t [s]", yticks=[0, 1], yticklabels=["follow", "overtake"], title="Decision at each planning step")
    ax_info.axis("off")
    ax_info.text(0.0, 1.0, "\n".join([
        f"planner: {args.planner}, model: {args.model}, predictor: {args.predictor}",
        f"ego: brakes x{args.brakes:g}, grip x{args.grip:g}, power x{args.power:g}; opponent x{r:g}",
        f"initial gap {args.gap:g} m, corner R = {args.radius:g} m, {args.angle:g} deg, L_out = {args.l_out:g} m", "",
        f"success: {o.success}   overtook: {o.overtook}   safe re-entry: {o.safe_reentry}",
        f"collision: {o.collision}   off track: {result.off_track}",
        f"margin t_opp - t_ego: {o.margin if o.margin is None else round(o.margin, 2)} s",
        f"finish times: ego {o.t_ego}, opponent {o.t_opponent}",
    ]), va="top", family="monospace", fontsize=10)
    figures = {"": fig}

    if isinstance(planner, RecordingSigmoid) and hasattr(planner, "first_overtake"):
        rec = planner.first_overtake
        full, choice, f = rec["full"], rec["choice"], rec["frame"]
        fig2, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        feasible_rows = {opt.row for opt in rec["options"]}
        for m in range(len(full)):
            ok = m in feasible_rows
            ax.plot(full.stations, full.d[m], color="tab:green" if ok else "0.8", lw=0.8 if ok else 0.5, zorder=2 if ok else 1)
        ax.plot(f.stations, plan_e.d_at(f.stations), ":", color="gray", label="racing line")
        if choice.kind == "overtake":
            ax.plot(full.stations, full.d[choice.row], color="tab:red", lw=2.5, label="chosen")
        ax.plot([], [], color="tab:green", label="feasible (first return point)")
        ax.plot([], [], color="0.8", label="not feasible")
        ax.axhline(d_min, color="k", lw=0.8)
        ax.axhline(d_max, color="k", lw=0.8)
        ax.axvspan(track.corner_start, track.corner_end, color="0.94")
        ax.set(xlabel="s [m]", ylabel="d [m]", title=f"Sigmoid candidates at t = {rec['t']:.2f} s ({len(full)} evaluated)")
        ax.legend(fontsize=8)
        figures["_candidates"] = fig2

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for suffix, figure in figures.items():
            path = args.out.with_name(args.out.stem + suffix + args.out.suffix)
            figure.savefig(path, dpi=140, bbox_inches="tight")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
