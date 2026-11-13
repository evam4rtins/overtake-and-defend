import numpy as np
import pytest
from overtake.speed_profile import accelerations, path_to_trajectory, three_pass, time_stamps, traversal_time
from overtake.track import CornerTrack, arc_length, curvature
from overtake.types import Trajectory, VehicleParams

P = VehicleParams()
V_START = 60.0
TRACKS = {
    "corner": dict(l_in=250, radius=40, angle_deg=90, l_out=150, width=12),
    "hairpin": dict(l_in=250, radius=15, angle_deg=180, l_out=150, width=12),
}


@pytest.fixture(params=TRACKS)
def track(request):
    return CornerTrack(**TRACKS[request.param])


def ellipse_use(s, v, kappa, p=P):
    a_x, a_y = accelerations(s, v, kappa)
    return np.hypot(a_x / p.a_x_max, a_y / p.a_y_max)


def braking_point(track, v):
    """ Station where the car starts to brake for the corner: highest speed before the arc. """
    before = track.s < track.corner_start
    return track.s[before][np.argmax(v[before])]


# ------------------------------------ three passes ---------------------------------------
def test_apex_speed(track):
    v = three_pass(track.s, track.kappa, P, V_START)
    apex = np.sqrt(P.a_y_max * track.radius)
    in_arc = (track.s > track.corner_start + 1) & (track.s < track.corner_end - 1)
    assert np.allclose(v[in_arc], apex)
    assert v.min() == pytest.approx(apex)


def test_slow_at_turn_in(track):
    v = three_pass(track.s, track.kappa, P, V_START)
    apex = np.sqrt(P.a_y_max * track.radius)
    turn_in = np.searchsorted(track.s, track.corner_start)
    assert v[0] == pytest.approx(V_START)                   # the entry speed is respected
    assert v[turn_in] <= apex + 1e-9                        # the braking is finished at turn-in
    assert v[turn_in - 20] > apex                           # and it happened on the straight
    assert v[-1] > apex                                     # the car accelerates out of the corner


def test_passes_only_lower_the_speed(track):
    guess, forward, final = three_pass(track.s, track.kappa, P, V_START, return_passes=True)
    assert np.all(forward <= guess + 1e-12) and np.all(final <= forward + 1e-12)
    assert np.all(np.diff(forward[track.s < track.corner_start]) >= -1e-12)   # forward pass never brakes
    assert np.array_equal(final, three_pass(track.s, track.kappa, P, V_START))


def test_accelerations_within_limits(track):
    v = three_pass(track.s, track.kappa, P, V_START)
    assert ellipse_use(track.s, v, track.kappa).max() <= 1.001
    a_x, _ = accelerations(track.s, v, track.kappa)
    assert np.all(a_x <= P.a_engine * (1 - v / P.v_max) + 0.02)   # engine limit
    assert a_x.min() == pytest.approx(-P.a_x_max)                 # full braking on the straight


def test_accelerations_within_limits_on_a_winding_path(track):
    """ Braking and cornering at the same time: the combined acceleration stays in the ellipse. """
    x, y = track.from_frenet(track.s, 3.0 * np.sin(track.s / 25.0))
    s, kappa = arc_length(x, y), curvature(x, y)
    v = three_pass(s, kappa, P, V_START)
    assert ellipse_use(s, v, kappa).max() <= 1.02


def test_better_brakes_brake_later(track):
    v = three_pass(track.s, track.kappa, P, V_START)
    strong = P.scaled(a_x_max=1.2)
    v_strong = three_pass(track.s, track.kappa, strong, V_START)
    assert braking_point(track, v_strong) > braking_point(track, v) + 5.0
    assert traversal_time(track.s, v_strong) < traversal_time(track.s, v)
    assert v_strong.min() == pytest.approx(v.min())               # same apex speed


def test_more_grip_is_faster_in_the_corner(track):
    v = three_pass(track.s, track.kappa, P, V_START)
    v_grip = three_pass(track.s, track.kappa, P.scaled(a_y_max=1.1), V_START)
    assert v_grip.min() == pytest.approx(np.sqrt(1.1) * v.min())


def test_entry_speed_too_high_is_reported():
    """ Straight too short to brake: the profile starts below the requested entry speed. """
    track = CornerTrack(l_in=30, radius=15, angle_deg=180, l_out=50)
    v = three_pass(track.s, track.kappa, P, V_START)
    assert v[0] < V_START - 10


def test_speed_limit_and_top_speed():
    s = np.linspace(0, 2000, 2001)
    v = three_pass(s, np.zeros_like(s), P, 70.0)
    assert np.all(np.diff(v) >= 0) and v[-1] <= P.v_max and v[-1] > 74.0
    assert three_pass(s, np.zeros_like(s), P, 70.0, v_limit=50.0).max() == pytest.approx(50.0)


def test_three_pass_rejects_bad_input():
    with pytest.raises(ValueError):
        three_pass([0, 1, 2], [0, 0], P, 10.0)
    with pytest.raises(ValueError):
        three_pass([0, 1, 1], [0, 0, 0], P, 10.0)


# ------------------------------------ time stamps ---------------------------------------
def test_time_stamps_at_constant_speed():
    s = np.linspace(0, 100, 11)
    assert np.allclose(time_stamps(s, np.full(11, 20.0), t0=2.0), 2.0 + s / 20.0)
    assert traversal_time(s, np.full(11, 20.0)) == pytest.approx(5.0)


def test_time_stamps_are_exact_for_uniform_acceleration():
    v0, a = 10.0, 3.0
    s = np.sort(np.random.default_rng(0).uniform(0, 200, 40))
    s[0] = 0.0
    v = np.sqrt(v0 ** 2 + 2 * a * s)
    assert np.allclose(time_stamps(s, v), (v - v0) / a)


def test_time_stamps_reject_a_stopped_car():
    with pytest.raises(ValueError):
        time_stamps([0, 1, 2], [0, 0, 5])


# ------------------------------------ path to trajectory ---------------------------------------
def test_path_to_trajectory(track):
    traj = path_to_trajectory(track.cx, track.cy, P, V_START, t0=1.5)
    assert isinstance(traj, Trajectory) and len(traj) == len(track.s)
    assert traj.t[0] == 1.5 and np.all(np.diff(traj.t) > 0)
    assert np.allclose(traj.s, track.s, atol=1e-3)
    v = three_pass(track.s, track.kappa, P, V_START)
    assert traj.duration == pytest.approx(traversal_time(track.s, v), rel=5e-3)
    assert traj.v.min() == pytest.approx(np.sqrt(P.a_y_max * track.radius), rel=1e-3)
