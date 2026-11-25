import numpy as np
import pytest

from overtake.agents import PlannerAgent, RacingLineAgent
from overtake.planners import FollowPlanner, SigmoidPlanner
from overtake.planners.base import (acc_cap, clearances, predicted_cars, curvature_limit, earliest_reentry,
                                    max_steepness, offsets_to_candidates, passing_distance, planning_horizon,
                                    race_speed, reentry_ok, rows_geometry, sigmoid_steps, stations_ahead)
from overtake.racing_line import centerline_plan, racing_line
from overtake.sim import Car, Simulator, outcome, start_state, trajectory_clearance
from overtake.track import CornerTrack, arc_length, curvature
from overtake.types import CarView, Observation, Trajectory, VehicleParams, VehicleState
from overtake.vehicles import KinematicBicycle, PointMass

P = VehicleParams()
SLOW = P.scaled(a_y_max=0.6, a_x_max=0.6, a_engine=0.6, v_max=0.6)    # a clearly slower car
TRACK = CornerTrack()                                                  # R = 40 m, 90 deg
HAIRPIN = CornerTrack(radius=15, angle_deg=180)
LONG_EXIT = CornerTrack(l_out=300)


@pytest.fixture(scope="module")
def plan():
    return racing_line(TRACK, P, 50.0)


def race(track, ego, opponent, gap, planner=None, s_opp=80.0, v0=50.0, Model=PointMass):
    """ Opponent on its racing line (RacingLineAgent), ego behind it with a local planner. """
    plan_o, plan_e = racing_line(track, opponent, v0), racing_line(track, ego, v0)
    s_ego = s_opp - gap - ego.length
    agent = PlannerAgent(planner or SigmoidPlanner())
    cars = [Car("opponent", opponent, Model(opponent), RacingLineAgent(),
                start_state(track, s_opp, plan_o.d_at(s_opp), float(race_speed(plan_o, s_opp))), plan_o),
            Car("ego", ego, Model(ego), agent,
                start_state(track, s_ego, plan_e.d_at(s_ego), float(race_speed(plan_e, s_ego))), plan_e)]
    result = Simulator(track, cars).run()
    return outcome(result), agent, result


def observation(track, plan, s, d_offset=0.0, v=None, others=(), t=0.0):
    v = float(race_speed(plan, s)) if v is None else v
    me = CarView("ego", start_state(track, s, plan.d_at(s) + d_offset, v), s, float(plan.d_at(s)) + d_offset, P)
    return Observation(t, track, me, list(others))


def straight_car(name, s0, v, d=0.0, t0=0.0, horizon=10.0, params=P):
    """ A car predicted at constant speed along the entry straight (which lies on the x axis). """
    t = t0 + np.arange(0.0, horizon + 1e-9, 0.05)
    traj = Trajectory(t=t, x=s0 + v * (t - t0), y=np.full_like(t, d), psi=np.zeros_like(t), v=np.full_like(t, v))
    view = CarView(name, VehicleState(s0, d, 0.0, v), s0, d, params)
    return view, traj


# ------------------------------------ helpers ---------------------------------------
def test_planning_horizon_scales_with_speed():
    assert planning_horizon(30.0, P, 100.0) == pytest.approx(30.0 ** 2 / 30.0 + 100.0 + 20.0)
    assert planning_horizon(60.0, P, 100.0) > planning_horizon(30.0, P, 100.0)
    assert planning_horizon(1.0, P, 0.0) == 50.0                          # minimum


def test_passing_distance_is_measured_on_the_racing_line(plan):
    assert passing_distance(plan, 100.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    t0 = float(np.interp(100.0, plan.s, plan.trajectory.t))
    assert 100.0 + passing_distance(plan, 100.0, 2.0) == pytest.approx(np.interp(t0 + 2.0, plan.trajectory.t, plan.s))
    # In the corner the car is slow, but the horizon counts the acceleration on the exit.
    apex = 0.5 * (TRACK.corner_start + TRACK.corner_end)
    assert passing_distance(plan, apex, 4.0) > 4.0 * race_speed(plan, apex)


def test_stations_ahead():
    s = stations_ahead(TRACK, 100.0, 150.0, ds=1.0)
    assert s[0] == 100.0 and s[-1] == pytest.approx(250.0) and np.allclose(np.diff(s), 1.0)
    end = stations_ahead(TRACK, TRACK.length - 5.0, 150.0)
    assert end[-1] == pytest.approx(TRACK.length + 15.0)                 # at least 20 m long


def test_steepest_sigmoid_uses_the_curvature_limit():
    v, w = 40.0, 4.0
    a = max_steepness(w, v, P)
    s = np.linspace(0.0, 200.0, 2001)
    y = w / (1.0 + np.exp(-a * (s - 100.0)))
    kappa = np.abs(curvature(s, y)).max()
    assert kappa == pytest.approx(curvature_limit(v, P), rel=0.02)       # slides 6a: rho_hat = rho_max
    assert max_steepness(w, v, P, share=0.5) < a
    assert curvature_limit(1.0, P) == pytest.approx(1.0 / np.hypot(P.l_r, P.wheelbase / np.tan(P.delta_max)))


def test_sigmoid_steps_start_at_the_car():
    s = np.linspace(0.0, 100.0, 101)
    steps = sigmoid_steps(s, [30.0, np.inf, -1e3], [0.2, 0.2, 0.2], s0=0.0)
    assert steps[0, 0] == 0.0 and steps[0, -1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(steps[0]) >= 0)
    assert np.all(steps[1] == 0.0) and np.all(steps[2] == 1.0)          # no step / step already done


def test_rows_geometry_matches_the_track_helpers():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.uniform(0.5, 1.5, (3, 50)), axis=1)
    y = np.cumsum(rng.normal(0.0, 0.3, (3, 50)), axis=1)
    s, kappa = rows_geometry(x, y)
    for m in range(3):
        assert np.allclose(s[m], arc_length(x[m], y[m]))
        assert np.allclose(kappa[m], curvature(x[m], y[m]))


def test_zero_offset_gives_the_racing_line(plan):
    stations = stations_ahead(TRACK, 60.0, 200.0)
    v0 = float(race_speed(plan, 60.0))
    terminal = np.full(len(stations), np.inf)
    terminal[-1] = race_speed(plan, stations[-1])                        # as LocalPlanner.frame does
    c = offsets_to_candidates(TRACK, plan, stations, np.zeros((2, len(stations))), P, v0, 3.0, terminal)
    assert len(c) == 2 and np.allclose(c.d[0], plan.d_at(stations))
    assert np.max(np.abs(c.v[0] - race_speed(plan, stations))) < 0.5
    traj = c.trajectory(0)
    assert traj.t[0] == 3.0 and np.allclose(traj.s, stations)


def test_batch_clearance_is_trajectory_clearance():
    plan = centerline_plan(TRACK, P, 30.0)
    stations = stations_ahead(TRACK, 20.0, 150.0)
    offsets = np.vstack([np.zeros(len(stations)), 4.0 * sigmoid_steps(stations, 30.0, 0.3, 20.0)[0]])
    c = offsets_to_candidates(TRACK, plan, stations, offsets, P, 30.0, 0.0)
    view, traj = straight_car("other", 35.0, 20.0)
    obs = observation(TRACK, plan, 20.0, v=30.0, others=[view])
    cars = predicted_cars(TRACK, obs, {"other": traj})
    batch = clearances(c, P, cars)
    for m in range(2):
        assert batch[m] == pytest.approx(trajectory_clearance(c.trajectory(m), P, cars[0].trajectory, P), abs=0.05)
    assert batch[0] < 0 < batch[1]                      # staying behind a slower car hits it, moving out does not


def test_reentry_rule_and_earliest_reentry():
    plan = centerline_plan(CornerTrack(l_in=600), P, 30.0)
    track = CornerTrack(l_in=600)
    stations = stations_ahead(track, 0.0, 400.0)
    view, traj = straight_car("other", 20.0, 20.0)
    obs = observation(track, plan, 0.0, v=30.0, others=[view])
    cars = predicted_cars(track, obs, {"other": traj})
    out = 4.0 * sigmoid_steps(stations, 20.0, 0.2, 0.0)                 # move out and stay out
    early = out - 4.0 * sigmoid_steps(stations, 75.0, 0.3, 0.0)        # come back right after passing
    late = out - 4.0 * sigmoid_steps(stations, 200.0, 0.3, 0.0)
    c = offsets_to_candidates(track, plan, stations, np.vstack([out, early, late]), P, 30.0, 0.0)
    ok = reentry_ok(c, P, cars)
    assert ok[0] and not ok[1] and ok[2]
    k = earliest_reentry(c, P, cars)[0]
    s_k, t_k = stations[k], c.t[0, k]
    gap = s_k - cars[0].station_at(t_k) - P.length
    assert gap >= 0.3 * 20.0 + 2.0 - 0.5 and gap < 0.3 * 20.0 + 2.0 + 3.0


def test_acc_cap():
    me = CarView("ego", VehicleState(0, 0, 0, 30.0), 100.0, 0.0, P)
    close = CarView("a", VehicleState(0, 0, 0, 25.0), 100.0 + P.length + 3.0, 0.5, P)
    beside = CarView("b", VehicleState(0, 0, 0, 25.0), 105.0, 4.0, P)
    assert acc_cap(me, [close]) < 25.0                                   # too close: slower than the leader
    assert acc_cap(me, [beside]) is None                                 # other lane
    assert acc_cap(me, []) is None


# ------------------------------------ follow planner ---------------------------------------
def test_follow_alone_is_the_racing_line(plan):
    planner = FollowPlanner()
    planner.setup(TRACK, P, plan)
    traj = planner.plan(observation(TRACK, plan, 100.0), {})
    assert planner.decision.kind == "follow"
    assert np.max(np.abs(TRACK.to_frenet(traj.x, traj.y)[1] - plan.d_at(traj.s))) < 0.05
    assert np.max(np.abs(traj.v - race_speed(plan, traj.s))) < 0.5


def test_follow_slows_down_behind_a_slower_car(plan):
    planner = FollowPlanner()
    planner.setup(TRACK, P, plan)
    view, traj = straight_car("lead", 140.0, 30.0, d=float(plan.d_at(140.0)))
    obs = observation(TRACK, plan, 100.0, v=40.0, others=[view])
    mine = planner.plan(obs, {"lead": traj})
    assert planner.decision.n_feasible > 0
    assert trajectory_clearance(mine, P, traj, P) >= 0.5
    # it may close the gap first, but then it is never faster than the leader (slower in the
    # corner, where its own grip limits it) and stays behind it
    end = mine.t > mine.t[0] + 3.0
    assert np.all(mine.v[end] <= 30.0 + 0.5)
    gap = 140.0 + 30.0 * (mine.t - mine.t[0]) - mine.s - P.length
    assert np.all(gap[end] >= planner.t_hw * 30.0 + planner.d_offset)


def test_faster_car_far_behind_keeps_its_speed(plan):
    """ A much faster car 50 m behind is not slowed to the leader's speed straight away. """
    planner = FollowPlanner()
    planner.setup(TRACK, P, plan)
    view, traj = straight_car("lead", 150.0, 30.0, d=float(plan.d_at(150.0)))
    obs = observation(TRACK, plan, 100.0, v=45.0, others=[view])
    mine = planner.plan(obs, {"lead": traj})
    assert mine.v[10] > 44.0                                  # no braking at the start
    assert trajectory_clearance(mine, P, traj, P) >= 0.5
    assert abs(mine.v[-1] - 30.0) < 0.5                       # still ends behind, at its speed


def test_follow_returns_to_its_line(plan):
    planner = FollowPlanner()
    planner.setup(TRACK, P, plan)
    traj = planner.plan(observation(TRACK, plan, 60.0, d_offset=4.0), {})
    d = TRACK.to_frenet(traj.x, traj.y)[1]
    assert d[0] == pytest.approx(plan.d_at(60.0) + 4.0, abs=1e-6)
    assert abs(d[-1] - plan.d_at(traj.s[-1])) < 0.05


def test_follow_agent_stays_behind_an_identical_car():
    o, _, _ = race(TRACK, P, P, 5.0, planner=FollowPlanner())
    assert not o.collision and not o.overtook and not o.off_track


# ------------------------------------ sigmoid planner ---------------------------------------
def test_no_car_ahead_means_following(plan):
    planner = SigmoidPlanner()
    planner.setup(TRACK, P, plan)
    planner.plan(observation(TRACK, plan, 100.0), {})
    assert planner.decision.kind == "follow"


@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_s0_identical_cars_no_overtake_no_collision(Model):
    o, agent, _ = race(TRACK, P, P, 5.0, Model=Model)
    assert not o.collision and not o.off_track and o.safe_reentry
    assert not o.overtook and not o.success
    assert agent.decisions                                               # the planner replanned


def test_clearly_faster_car_overtakes_safely():
    """ Opponent much slower: the ego moves out, passes, and returns at the earliest safe point. """
    o, agent, _ = race(LONG_EXIT, P, SLOW, 5.0, s_opp=40.0, v0=40.0)
    assert o.success and o.overtook and o.safe_reentry and not o.collision
    assert o.margin > 0.5
    returns = [d.c2 for _, d in agent.decisions if d.kind == "overtake" and np.isfinite(d.c2)]
    assert returns                                                       # a return to the line was planned


@pytest.mark.parametrize("track", [TRACK, HAIRPIN])
def test_braking_advantage_alone_stays_safe(track):
    """
    Better brakes alone do not give enough gap for a safe re-entry before the apex (Approach 0):
    the planner must then stay behind, without collision or unsafe cut-in.
    """
    o, _, _ = race(track, P.scaled(a_x_max=2.0), P, 2.0)
    assert not o.collision and o.safe_reentry and not o.off_track


class RecordingPlanner(SigmoidPlanner):
    """ Keeps the inputs and the plan of the first planning step that chose to overtake. """

    def plan(self, obs, predictions):
        traj = super().plan(obs, predictions)
        if self.decision.kind == "overtake" and not hasattr(self, "first"):
            self.first = (obs, predictions, traj, self.decision)
        return traj


def test_chosen_overtake_is_feasible_and_kept():
    planner = RecordingPlanner()
    _, agent, result = race(LONG_EXIT, P, SLOW, 5.0, planner=planner, s_opp=40.0, v0=40.0)
    _, predictions, traj, first = planner.first
    assert first.cost < first.follow_cost                                # faster than staying behind
    d = LONG_EXIT.to_frenet(traj.x, traj.y, s_hint=traj.s)[1]
    d_min, d_max = LONG_EXIT.d_limits(P.width, 0.3)
    assert d.min() >= d_min - 0.05 and d.max() <= d_max + 0.05          # within the track limits
    assert trajectory_clearance(traj, P, predictions["opponent"], SLOW) >= 0.5   # collision free
    # Hysteresis: once the ego has left its line to pass, it never aborts before being ahead.
    ego, opp, line = result.cars["ego"], result.cars["opponent"], planner.global_plan
    moved_out = 0
    for t, d in agent.decisions:
        k = int(np.searchsorted(ego["t"], t - 1e-9))
        out = abs(ego["d"][k] - line.d_at(ego["s"][k])) > 1.0
        behind = ego["s"][k] < opp["s"][k] + P.length
        if out and behind:
            moved_out += 1
            assert d.kind == "overtake"
    assert moved_out > 0
