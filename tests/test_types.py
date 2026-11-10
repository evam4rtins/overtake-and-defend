import numpy as np
import pytest
from overtake.types import CarView, Observation, Trajectory, VehicleParams, VehicleState

def test_scaled_params():
    p = VehicleParams()
    q = p.scaled(a_x_max=1.1)
    assert q.a_x_max == pytest.approx(1.1*p.a_x_max)
    assert q.a_y_max == p.a_y_max
    assert p.a_x_max == 15.0
    with pytest.raises(ValueError):
        p.scaled(grip=1.1)

def test_trajectory_arc_legth_and_interpolation():
    tr = Trajectory(t=[0, 1, 2], x=[0, 3, 6], y=[0, 4, 8], psi=[0.9, 0.9, 0.9], v=[5, 5, 5])
    assert np.allclose(tr.s, [0, 5, 10])
    x, y, _, v = tr.at_time(0.5)
    assert (x, y, v) == pytest.approx((1.5, 2.0, 5.0))
    assert tr.duration == 2.0

def test_trajectory_rejects_bad_input():
    with pytest.raises(ValueError):
        Trajectory(t=[0, 0, 1], x=[0, 1, 2], y=[0, 0, 0], psi=[0, 0, 0], v=[1, 1, 1])
    with pytest.raises(ValueError):
        Trajectory(t=[0, 1], x=[0, 1, 2], y=[0, 0], psi=[0, 0], v=[1, 1])


def test_observation_car_ahead():
    p, st = VehicleParams(), VehicleState(0, 0, 0, 30)
    me = CarView("ego", st, 100.0, 0.0, p)
    a, b = CarView("a", st, 120.0, 0.0, p), CarView("b", st, 90.0, 0.0, p)
    obs = Observation(0.0, None, me, [a, b])
    assert obs.car_ahead().name == "a"
