import numpy as np
import pytest

from overtake.agents import RacingLineAgent
from overtake.racing_line import racing_line
from overtake.sim import Car, EpisodeResult, Simulator, footprint_distance, outcome, start_state, trajectory_clearance
from overtake.track import CornerTrack
from overtake.types import Trajectory, VehicleParams, VehicleState
from overtake.vehicles import KinematicBicycle, PointMass

P = VehicleParams()
TRACK = CornerTrack(l_in=250, radius=40, angle_deg=90, l_out=150, width=12)
PLAN = racing_line(TRACK, P, v_start=50.0)


def two_cars(Model, gap, s_opponent=60.0, v=50.0):
    s_ego = s_opponent - gap - P.length
    return [Car("opponent", P, Model(P), RacingLineAgent(), start_state(TRACK, s_opponent, PLAN.d_at(s_opponent), v), PLAN),
            Car("ego", P, Model(P), RacingLineAgent(), start_state(TRACK, s_ego, PLAN.d_at(s_ego), v), PLAN)]


# ------------------------------------ collision checks ---------------------------------------
def test_footprint_distance():
    side_by_side = footprint_distance(0, 0, 0, P, 0, P.width + 1.0, 0, P)
    assert 0 < side_by_side < 1.0                         # 1 m apart, circles a little conservative
    assert footprint_distance(0, 0, 0, P, 2.0, 0, 0, P) < 0          # overlapping
    # 3 m between the bumpers: the end circles stick out ~0.45 m at each end, so the clearance is
    # a little smaller (conservative), but positive.
    behind = footprint_distance(0, 0, 0, P, -(P.length + 3.0), 0, 0, P)
    assert 2.0 < behind < 3.0


def test_trajectory_clearance_is_space_time():
    """ Two cars crossing the same point at different times do not collide. """
    t = np.linspace(0, 4, 81)
    a = Trajectory(t=t, x=10 * t, y=np.zeros_like(t), psi=np.zeros_like(t), v=np.full_like(t, 10))
    b = Trajectory(t=t, x=np.full_like(t, 20.0), y=10 * t - 30, psi=np.full_like(t, np.pi / 2), v=np.full_like(t, 10))
    assert trajectory_clearance(a, P, b, P) > 0           # b reaches (20, 0) at t = 3, a was there at t = 2
    b_late = Trajectory(t=t, x=np.full_like(t, 20.0), y=10 * t - 20, psi=b.psi, v=b.v)
    assert trajectory_clearance(a, P, b_late, P) < 0      # both at (20, 0) at t = 2


# ------------------------------------ simulator ---------------------------------------
@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_single_car_follows_its_racing_line(Model):
    car = Car("car", P, Model(P), RacingLineAgent(), start_state(TRACK, 0.0, PLAN.d_at(0.0), 50.0), PLAN)
    result = Simulator(TRACK, [car]).run()
    log = result.cars["car"]
    assert result.off_track == [] and result.finish_times["car"] is not None
    assert np.max(np.abs(log["d"] - PLAN.d_at(log["s"]))) < 0.8
    assert result.finish_times["car"] == pytest.approx(PLAN.time, rel=0.05)


@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_s0_identical_cars_follower_stays_behind(Model):
    """ Scenario S0: identical cars, small gap -> no collision, no overtake. """
    result = Simulator(TRACK, two_cars(Model, gap=3.0)).run()
    o = outcome(result)
    assert not o.collision and not o.off_track
    assert not o.overtook and not o.success
    assert np.all(result.cars["ego"]["s"] < result.cars["opponent"]["s"])


def test_collision_is_detected():
    cars = [Car("a", P, PointMass(P), None, VehicleState(10, 0, 0, 20.0)),
            Car("b", P, PointMass(P), None, VehicleState(13, 0, 0, 10.0))]
    sim = Simulator(TRACK, cars)
    assert sim.collision                                   # overlapping at t = 0
    done = sim.step({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    assert done                                            # stops at the first collision


def test_external_actions_drive_cars_without_agent():
    car = Car("rl", P, PointMass(P), None, VehicleState(0, 0, 0, 20.0))
    sim = Simulator(TRACK, [car], dt=0.1)
    sim.step({"rl": [2.0, 0.0]})
    assert car.state.v == pytest.approx(20.2)


# ------------------------------------ success criterion ---------------------------------------
def fake_result(s_ego, s_opp, d_ego, d_opp, t_ego, t_opp, collision=False):
    n = len(s_ego)
    log = lambda s, d: {"s": np.array(s, float), "d": np.array(d, float), "v": np.full(n, 30.0)}
    return EpisodeResult(cars={"ego": log(s_ego, d_ego), "opponent": log(s_opp, d_opp)},
                         finish_times={"ego": t_ego, "opponent": t_opp}, collision=collision, off_track=[],
                         min_clearance=1.0, t_end=10.0, params={"ego": P, "opponent": P})


def test_outcome_clean_overtake():
    r = fake_result([0, 50, 120], [10, 55, 90], [0, 3, 0], [0, 0, 0], t_ego=9.0, t_opp=9.5)
    o = outcome(r)
    assert o.success and o.overtook and o.safe_reentry and o.margin == pytest.approx(0.5)


def test_outcome_cut_in_too_early():
    """ Ego back in the opponent's lane only 1 m ahead: not a safe re-entry. """
    r = fake_result([0, 50, 61.7], [10, 55, 56], [0, 3, 0], [0, 0, 0], t_ego=9.0, t_opp=9.5)
    o = outcome(r)
    assert o.overtook and not o.safe_reentry and not o.success


def test_outcome_needs_margin_and_no_collision():
    assert not outcome(fake_result([0, 1], [10, 11], [0, 0], [0, 0], t_ego=9.45, t_opp=9.5)).success
    assert not outcome(fake_result([0, 120], [10, 90], [3, 3], [0, 0], 9.0, 9.5, collision=True)).success
