import numpy as np
import pytest
from overtake.track import CornerTrack, arc_length, curvature, heading


def circle(radius, turn_left=True, n=200, angle=np.pi):
    """ Points on a circle of the given radius, starting at the origin and heading along +x. """
    phi = np.linspace(0, angle, n)
    sign = 1 if turn_left else -1
    return radius * np.sin(phi), sign * radius * (1 - np.cos(phi))


# ------------------------------------ T03: path geometry ---------------------------------------
def test_arc_length():
    assert np.allclose(arc_length([0, 3, 6], [0, 4, 8]), [0, 5, 10])
    x, y = circle(20.0)
    assert arc_length(x, y)[-1] == pytest.approx(np.pi * 20.0, rel=1e-4)


def test_curvature_of_a_circle():
    x, y = circle(25.0, turn_left=True)
    assert np.allclose(curvature(x, y), 1 / 25.0)
    x, y = circle(25.0, turn_left=False)
    assert np.allclose(curvature(x, y), -1 / 25.0)


def test_curvature_of_a_straight_line():
    x = np.linspace(0, 50, 30)
    assert np.allclose(curvature(x, 2 * x + 1), 0.0)
    assert np.allclose(curvature([0, 1], [0, 1]), 0.0)   # fewer than three points


def test_curvature_with_uneven_spacing():
    phi = np.sort(np.random.default_rng(0).uniform(0, np.pi, 50))
    assert np.allclose(curvature(10 * np.sin(phi), 10 * (1 - np.cos(phi))), 1 / 10.0)


def test_heading():
    x = np.linspace(0, 10, 5)
    assert np.allclose(heading(x, x), np.pi / 4)
    # On a circle the heading equals the angle travelled, also at the two end points.
    phi = np.linspace(0, 1.5 * np.pi, 100)
    x, y = circle(30.0, n=100, angle=1.5 * np.pi)
    assert np.allclose(heading(x, y), phi)
    x, y = circle(30.0, turn_left=False, n=100, angle=1.5 * np.pi)
    assert np.allclose(heading(x, y), -phi)


def test_geometry_rejects_bad_input():
    with pytest.raises(ValueError):
        arc_length([0, 1, 2], [0, 1])
    with pytest.raises(ValueError):
        curvature([0, 1, 1, 2], [0, 0, 0, 0])   # repeated point
    with pytest.raises(ValueError):
        heading([0], [0])


# ------------------------------------ T04: corner track ---------------------------------------
TRACKS = {
    "left": dict(l_in=250, radius=40, angle_deg=90, l_out=150, width=12),
    "right": dict(l_in=100, radius=40, angle_deg=-90, l_out=80, width=12),
    "hairpin": dict(l_in=250, radius=15, angle_deg=180, l_out=150, width=12),
}


@pytest.fixture(params=TRACKS)
def track(request):
    return CornerTrack(**TRACKS[request.param])


def test_track_length_and_sampling(track):
    expected = track.l_in + track.radius * abs(track.angle) + track.l_out
    assert track.length == pytest.approx(expected)
    assert track.s[0] == 0 and track.s[-1] == pytest.approx(expected)
    assert np.all(np.abs(np.diff(track.s) - 0.5) < 0.01)
    assert track.corner_end - track.corner_start == pytest.approx(track.radius * abs(track.angle))


def test_baseline_track_end_point():
    track = CornerTrack(l_in=250, radius=40, angle_deg=90, l_out=150, width=12)
    assert track.length == pytest.approx(250 + 40 * np.pi / 2 + 150)
    x, y, psi, kappa = track.at_s(track.length)
    assert (x, y, psi, kappa) == pytest.approx((290.0, 190.0, np.pi / 2, 0.0))
    assert track.at_s(track.corner_start + 1.0)[3] == pytest.approx(1 / 40)


def test_sampled_centreline_matches_geometry_helpers(track):
    """ The T03 helpers, applied to the sampled centreline, give back s, psi and kappa. """
    assert np.allclose(arc_length(track.cx, track.cy), track.s, atol=1e-3)
    assert np.allclose(heading(track.cx, track.cy), track.psi, atol=0.02)
    # Curvature jumps at the two ends of the arc, so compare away from them.
    away = (np.abs(track.s - track.corner_start) > 1.0) & (np.abs(track.s - track.corner_end) > 1.0)
    assert np.allclose(curvature(track.cx, track.cy)[away], track.kappa[away], atol=1e-6)


def test_frenet_round_trip(track):
    rng = np.random.default_rng(1)
    s = rng.uniform(-5, track.length + 5, 500)
    d = rng.uniform(-track.width / 2, track.width / 2, 500)
    s_back, d_back = track.to_frenet(*track.from_frenet(s, d))
    assert np.allclose(s_back, s, atol=1e-9)
    assert np.allclose(d_back, d, atol=1e-9)
    # Scalars in, floats out.
    s0, d0 = track.to_frenet(*track.from_frenet(track.corner_start + 3.0, 1.5))
    assert isinstance(s0, float) and (s0, d0) == pytest.approx((track.corner_start + 3.0, 1.5))


def test_positive_d_is_left():
    track = CornerTrack(angle_deg=90)
    assert track.from_frenet(10.0, 2.0) == pytest.approx((10.0, 2.0))    # heading +x: left is +y
    assert track.to_frenet(10.0, -3.0) == pytest.approx((10.0, -3.0))
    # In a left turn the left side is the inside: closer to the centre of the arc.
    s_mid = (track.corner_start + track.corner_end) / 2
    centre = np.array([track.l_in, track.radius])
    inside = np.array(track.from_frenet(s_mid, 2.0))
    assert np.linalg.norm(inside - centre) == pytest.approx(track.radius - 2.0)


def test_s_hint_selects_the_branch():
    """ In a hairpin a point between the two straights is as close to one as to the other. """
    track = CornerTrack(**TRACKS["hairpin"])
    s_exit = track.corner_end + 150.0
    s, d = track.to_frenet(100.0, 15.0, s_hint=98.0)
    assert (s, d) == pytest.approx((100.0, 15.0))
    s, d = track.to_frenet(100.0, 15.0, s_hint=s_exit - 2.0)
    assert (s, d) == pytest.approx((s_exit, 15.0))


def test_d_limits_and_bounds(track):
    d_min, d_max = track.d_limits(car_width=1.9, margin=0.3)
    assert (d_min, d_max) == pytest.approx((-4.75, 4.75))
    with pytest.raises(ValueError):
        track.d_limits(car_width=13.0)
    (xl, yl), (xr, yr) = track.bounds()
    assert len(xl) == len(track.s)
    assert np.allclose(np.hypot(xl - xr, yl - yr), track.width)
    # The inside edge of the corner is shorter than the outside edge.
    left, right = arc_length(xl, yl)[-1], arc_length(xr, yr)[-1]
    assert (left < right) == (track.angle > 0)


def test_track_rejects_bad_parameters():
    with pytest.raises(ValueError):
        CornerTrack(radius=5, width=12)
    with pytest.raises(ValueError):
        CornerTrack(angle_deg=0)
