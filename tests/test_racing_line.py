import numpy as np
import pytest
from scipy.optimize import approx_fprime
from overtake import racing_line as rl
from overtake.racing_line import GlobalPlan, centerline_plan, racing_line
from overtake.speed_profile import accelerations, three_pass, traversal_time
from overtake.track import CornerTrack, arc_length, curvature
from overtake.types import VehicleParams

P = VehicleParams()
V_START = 50.0
MARGIN = 0.3
RATIOS = (0.0, 1e-5, 1e-4, 1e-3)


@pytest.fixture(scope="module")
def track():
    return CornerTrack(l_in=150, radius=40, angle_deg=90, l_out=120, width=12)


@pytest.fixture(scope="module")
def plan(track):
    return racing_line(track, P, V_START, ratios=RATIOS, margin=MARGIN)


def test_faster_than_the_centreline(track, plan):
    assert plan.time < centerline_plan(track, P, V_START).time - 0.3


def test_inside_the_track(track, plan):
    d_min, d_max = track.d_limits(P.width, MARGIN)
    assert plan.d.min() >= d_min - 1e-9 and plan.d.max() <= d_max + 1e-9
    assert len(plan.d) == len(track.s) and np.array_equal(plan.s, track.s)


def test_outside_inside_outside(track, plan):
    """ Left turn: the outside is the right side (d < 0), the inside the left side (d > 0). """
    d_max = track.d_limits(P.width, MARGIN)[1]
    apex = (track.corner_start + track.corner_end) / 2
    assert plan.d_at(track.corner_start - 40) < -0.5 * d_max
    assert plan.d_at(apex) > 0.9 * d_max
    assert plan.d_at(track.corner_end + 40) < -0.5 * d_max


def test_right_turn_is_the_mirror_image(track, plan):
    mirror = CornerTrack(track.l_in, track.radius, -track.angle_deg, track.l_out, track.width)
    other = racing_line(mirror, P, V_START, ratios=RATIOS, margin=MARGIN)
    assert other.time == pytest.approx(plan.time, abs=0.01)
    assert other.weight_ratio == plan.weight_ratio
    assert np.allclose(other.d, -plan.d, atol=0.05)


def test_selected_ratio_is_the_fastest(plan):
    assert set(plan.candidates) == set(RATIOS)
    assert plan.candidates[plan.weight_ratio] == pytest.approx(plan.time)
    assert plan.time == pytest.approx(min(plan.candidates.values()))


def test_trajectory_of_the_plan(track, plan):
    traj = plan.trajectory
    assert np.array_equal(traj.s, track.s)                        # indexed by track station
    assert traj.v[0] == pytest.approx(V_START)                    # entry speed compatible
    x, y = track.from_frenet(track.s, plan.d)
    assert np.allclose(traj.x, x) and np.allclose(traj.y, y)
    # The speed profile of the line stays inside the friction ellipse.
    s_path, kappa = arc_length(x, y), curvature(x, y)
    a_x, a_y = accelerations(s_path, traj.v, kappa)
    assert np.hypot(a_x / P.a_x_max, a_y / P.a_y_max).max() <= 1.02
    assert plan.time == pytest.approx(traversal_time(s_path, traj.v))


def test_d_at(plan):
    assert plan.d_at(plan.s[10]) == pytest.approx(plan.d[10])
    mid = 0.5 * (plan.s[10] + plan.s[11])
    assert plan.d_at(mid) == pytest.approx(0.5 * (plan.d[10] + plan.d[11]))
    assert plan.d_at(plan.s[:5]).shape == (5,)


def test_centerline_plan(track):
    centre = centerline_plan(track, P, V_START)
    assert isinstance(centre, GlobalPlan) and np.all(centre.d == 0)
    assert np.isnan(centre.weight_ratio)
    v = three_pass(track.s, track.kappa, P, V_START)
    assert centre.time == pytest.approx(traversal_time(track.s, v), rel=5e-3)


def test_better_car_is_faster_on_its_line(track, plan):
    assert racing_line(track, P.scaled(a_y_max=1.1), V_START, ratios=RATIOS, margin=MARGIN).time < plan.time
    assert racing_line(track, P.scaled(a_x_max=1.1), V_START, ratios=RATIOS, margin=MARGIN).time < plan.time


def test_pure_criteria(track):
    """ Ratio 0 gives the smoothest line, infinity the shortest one. """
    smooth = racing_line(track, P, V_START, ratios=(0.0,), margin=MARGIN)
    short = racing_line(track, P, V_START, ratios=(np.inf,), margin=MARGIN)

    def path(plan):
        x, y = track.from_frenet(track.s, plan.d)
        return arc_length(x, y)[-1], np.abs(curvature(x, y)).max()

    (len_smooth, kappa_smooth), (len_short, kappa_short) = path(smooth), path(short)
    assert len_short < len_smooth < track.length
    assert kappa_smooth < 1 / track.radius < kappa_short


def test_hairpin_apex_is_on_the_inside():
    track = CornerTrack(l_in=120, radius=15, angle_deg=180, l_out=100)
    plan = racing_line(track, P, V_START, margin=MARGIN)
    d_max = track.d_limits(P.width, MARGIN)[1]
    assert plan.d_at((track.corner_start + track.corner_end) / 2) > 0.9 * d_max
    assert plan.d.max() <= d_max + 1e-9
    assert plan.time < centerline_plan(track, P, V_START).time


def test_cost_gradient(track):
    s_nodes = rl._node_stations(track, 4.0)
    cx, cy, psi, _ = track.at_s(s_nodes)
    nx, ny = -np.sin(psi), np.cos(psi)
    d = np.random.default_rng(0).uniform(-3, 3, len(s_nodes))
    for k_l, k_rho in ((1.0, 0.0), (0.0, 1.0), (1e-4, 1.0)):
        cost, gradient = rl._cost(d, cx, cy, nx, ny, k_l, k_rho)
        numeric = approx_fprime(d, lambda z: rl._cost(z, cx, cy, nx, ny, k_l, k_rho)[0], 1e-6)
        assert np.allclose(gradient, numeric, rtol=1e-4, atol=1e-6 * np.abs(numeric).max())
