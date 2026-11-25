"""
Ego with a local planner against an opponent on its racing line.

    python scripts/run_planner.py                                   # S0: identical cars, sigmoid planner
    python scripts/run_planner.py --grip 1.15 --tau0 0.1            # S1: more lateral grip
    python scripts/run_planner.py --brakes 1.15 --tau0 0.1          # S2: more braking grip
    python scripts/run_planner.py --power 1.1 --top-speed 1.05 --l-out 300   # S3: power / top speed
    python scripts/run_planner.py --dv0 5 --tau0 0.3                # S4: initial speed difference
    python scripts/run_planner.py --predictor constant_velocity     # S5: imperfect prediction
    python scripts/run_planner.py --t-hw 0.2 --d-offset 2           # one spacing rule for both cars
    python scripts/run_planner.py --opponent-scale 0.6 --l-out 300 --s-opp 40 --v0 40   # clear pass (test)
    python scripts/run_planner.py --planner follow --model bicycle --radius 15 --angle 180
    python scripts/run_planner.py --out results/planner.png         # save the figures
    python scripts/run_planner.py --opponent-scale 0.6 --l-out 300 --s-opp 40 --v0 40 --gif results/pass.gif

Ego ratios (--grip, --brakes, --power, --top-speed) multiply the baseline car; the opponent keeps
the baseline unless --opponent-scale is used. The initial gap is given in metres (--gap) or in
seconds (--tau0). --v0 is only the entry speed of the racing lines; both cars start at the speed
of their line at their station (+ --dv0 for the ego).

Figure 1: trajectories (with the planning steps where the ego decided to overtake), speeds along
the track, lateral offsets, gap between the cars and the decisions over time.
Figure 2 (sigmoid planner): the candidates of the first planning step that chose to overtake,
feasible or not, and the chosen one.
--gif: animation of the race (whole track, and a view that follows the ego with both cars to
scale and the ego's current plan), played in real time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overtake.agents import PlannerAgent, RacingLineAgent
from overtake.planners import PLANNERS, SigmoidPlanner
from overtake.planners.base import REENTRY_OFFSET, REENTRY_T_HW
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


def keep_plans(planner):
    """ Wrap planner.plan so that every published plan is kept as (t, trajectory, kind), for the gif. """
    plans, plan = [], planner.plan

    def recording_plan(obs, predictions):
        traj = plan(obs, predictions)
        plans.append((obs.t, traj, planner.decision.kind if planner.decision else "follow"))
        return traj

    planner.plan = recording_plan
    return plans


def car_outline(x, y, psi, params):
    """ Corners of the car rectangle (closed polygon), for drawing. """
    hl, hw = params.length / 2, params.width / 2
    corners = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]])
    c, s = np.cos(psi), np.sin(psi)
    return x + c * corners[:, 0] - s * corners[:, 1], y + s * corners[:, 0] + c * corners[:, 1]


def save_gif(path, track, result, plans, ego_params, opp_params, fps=10, window=70.0):
    """
    Animation of the race: the whole track on the left, a view that follows the ego on the right
    (both cars drawn to scale, their recent path and the ego's current plan). One frame every
    1 / fps seconds of simulation, so the gif plays in real time.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    e, op = result.cars["ego"], result.cars["opponent"]
    n = min(len(e["t"]), len(op["t"]))
    dt = float(e["t"][1] - e["t"][0])
    frames = np.arange(0, n, max(round(1.0 / (fps * dt)), 1))
    left, right = track.bounds()
    plan_times = np.array([t for t, _, _ in plans])

    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax in (ax_all, ax_zoom):
        ax.plot(*left, "k", lw=1)
        ax.plot(*right, "k", lw=1)
        ax.set_aspect("equal")
    ax_all.set_title("Track")
    ax_all.set_xticks([])
    ax_all.set_yticks([])
    dot_o, = ax_all.plot([], [], "o", color="tab:blue", ms=5)
    dot_e, = ax_all.plot([], [], "o", color="tab:red", ms=5)

    trail_o, = ax_zoom.plot([], [], color="tab:blue", lw=1, alpha=0.5)
    trail_e, = ax_zoom.plot([], [], color="tab:red", lw=1, alpha=0.5)
    plan_line, = ax_zoom.plot([], [], "--", color="tab:red", lw=1.2, label="ego plan")
    body_o, = ax_zoom.fill([], [], color="tab:blue", label="opponent")
    body_e, = ax_zoom.fill([], [], color="tab:red", label="ego")
    ax_zoom.legend(loc="upper right", fontsize=8)
    info = ax_zoom.text(0.02, 0.98, "", transform=ax_zoom.transAxes, va="top", family="monospace", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.tight_layout()

    def draw(k):
        dot_o.set_data([op["x"][k]], [op["y"][k]])
        dot_e.set_data([e["x"][k]], [e["y"][k]])
        start = max(k - int(3.0 / dt), 0)                      # last 3 s of path
        trail_o.set_data(op["x"][start:k + 1], op["y"][start:k + 1])
        trail_e.set_data(e["x"][start:k + 1], e["y"][start:k + 1])
        body_o.set_xy(np.column_stack(car_outline(op["x"][k], op["y"][k], op["psi"][k], opp_params)))
        body_e.set_xy(np.column_stack(car_outline(e["x"][k], e["y"][k], e["psi"][k], ego_params)))
        kind = ""
        if len(plans):
            i = max(int(np.searchsorted(plan_times, e["t"][k], side="right")) - 1, 0)
            _, traj, kind = plans[i]
            plan_line.set_data(traj.x, traj.y)
        ax_zoom.set_xlim(e["x"][k] - window / 2, e["x"][k] + window / 2)
        ax_zoom.set_ylim(e["y"][k] - window / 3, e["y"][k] + window / 3)
        info.set_text(f"t = {e['t'][k]:5.2f} s   decision: {kind}\n"
                      f"v ego {e['v'][k]:5.1f} m/s   v opp {op['v'][k]:5.1f} m/s\n"
                      f"s_ego - s_opp = {e['s'][k] - op['s'][k]:6.1f} m")
        return dot_o, dot_e, trail_o, trail_e, body_o, body_e, plan_line, info

    path.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Saved {path} ({len(frames)} frames)")


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
    parser.add_argument("--top-speed", type=float, default=1.0, help="top speed ratio of the ego")
    parser.add_argument("--opponent-scale", type=float, default=1.0, help="all opponent parameters x this ratio")
    parser.add_argument("--gap", type=float, default=5.0, help="initial bumper-to-bumper gap [m]")
    parser.add_argument("--tau0", type=float, default=None,
                        help="initial time gap [s]; replaces --gap (gap = tau0 x opponent speed)")
    parser.add_argument("--dv0", type=float, default=0.0, help="the ego starts this much faster than its racing line [m/s]")
    parser.add_argument("--s-opp", type=float, default=80.0, help="initial station of the opponent [m]")
    parser.add_argument("--v0", type=float, default=50.0, help="entry speed used for the racing lines [m/s]")
    parser.add_argument("--t-hw", type=float, default=None,
                        help="ACC and re-entry time headway [s], used by both cars, the planner and the success criterion")
    parser.add_argument("--d-offset", type=float, default=None, help="ACC and re-entry standstill distance [m], same")
    parser.add_argument("--out", type=Path, default=None, help="save the figures (second one gets '_candidates')")
    parser.add_argument("--gif", type=Path, default=None, help="also save an animation of the race to this .gif file")
    parser.add_argument("--gif-fps", type=float, default=10.0, help="frames per second of the gif (real time)")
    args = parser.parse_args()

    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out)
    base = VehicleParams()
    r = args.opponent_scale
    opp_params = base.scaled(a_y_max=r, a_x_max=r, a_engine=r, v_max=r)
    ego_params = base.scaled(a_x_max=args.brakes, a_y_max=args.grip, a_engine=args.power, v_max=args.top_speed)
    plan_o, plan_e = racing_line(track, opp_params, args.v0), racing_line(track, ego_params, args.v0)

    # start: the opponent on its racing line at s_opp, the ego behind it (gap in metres or in seconds)
    v_opp = float(np.interp(args.s_opp, plan_o.s, plan_o.trajectory.v))
    gap = args.gap if args.tau0 is None else args.tau0 * v_opp
    s_ego = args.s_opp - gap - ego_params.length
    v_ego = float(np.interp(s_ego, plan_e.s, plan_e.trajectory.v)) + args.dv0

    # one spacing rule for everything if --t-hw or --d-offset is given, otherwise the code defaults
    # (opponent ACC 0.6 s + 5 m; ego, planner and success criterion 0.3 s + 2 m)
    if args.t_hw is None and args.d_offset is None:
        opp_rule, ego_rule = {}, {}
        t_hw, d_offset = REENTRY_T_HW, REENTRY_OFFSET
    else:
        t_hw = REENTRY_T_HW if args.t_hw is None else args.t_hw
        d_offset = REENTRY_OFFSET if args.d_offset is None else args.d_offset
        opp_rule = ego_rule = {"t_hw": t_hw, "d_offset": d_offset}

    Model = MODELS[args.model]
    planner = RecordingSigmoid(**ego_rule) if args.planner == "sigmoid" else PLANNERS[args.planner](**ego_rule)
    plans = keep_plans(planner) if args.gif else []
    agent = PlannerAgent(planner, predictor=PREDICTORS[args.predictor](horizon=10.0), **ego_rule)
    cars = [Car("opponent", opp_params, Model(opp_params), RacingLineAgent(**opp_rule),
                start_state(track, args.s_opp, plan_o.d_at(args.s_opp), v_opp), plan_o),
            Car("ego", ego_params, Model(ego_params), agent,
                start_state(track, s_ego, plan_e.d_at(s_ego), v_ego), plan_e)]
    result = Simulator(track, cars).run()
    o = outcome(result, reentry_t_hw=t_hw, reentry_offset=d_offset)
    print(o)
    kinds = [d.kind for _, d in agent.decisions]
    print(f"{len(kinds)} planning steps, {kinds.count('overtake')} chose to overtake")
    if args.gif:
        save_gif(args.gif, track, result, plans, ego_params, opp_params, fps=args.gif_fps)

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
        (f"ego: brakes x{args.brakes:g}, grip x{args.grip:g}, power x{args.power:g}, "
         f"top speed x{args.top_speed:g}; opponent x{r:g}"),
        (f"initial gap {gap:.1f} m ({gap / v_opp:.2f} s), dv0 {args.dv0:g} m/s; "
         f"spacing {'0.6 s + 5 m / 0.3 s + 2 m (defaults)' if not ego_rule else f'{t_hw:g} s + {d_offset:g} m'}"),
        f"corner R = {args.radius:g} m, {args.angle:g} deg, L_out = {args.l_out:g} m", "",
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
