import numpy as np
import pytest

from overtake.control import PurePursuit, acc_speed, curvature_to_steering
from overtake.speed_profile import path_to_trajectory
from overtake.types import VehicleParams, VehicleState
from overtake.vehicles import KinematicBicycle, PointMass

P = VehicleParams()


def track_reference(Model, ref, state, steps, dt=0.02):
    model, ctrl = Model(P), PurePursuit()
    errors = []
    for _ in range(steps):
        action = ctrl.control(state, ref, P, model.action_names)
        state = model.step(state, action, dt)
        errors.append(np.min(np.hypot(ref.x - state.x, ref.y - state.y)))
    return state, np.array(errors)


def test_steering_for_curvature_matches_the_bicycle():
    """ The steering given by curvature_to_steering makes the bicycle drive that curvature. """
    kappa = 1 / 30.0
    delta = curvature_to_steering(kappa, P)
    model = KinematicBicycle(P)
    state = model.step(VehicleState(0, 0, 0, 5.0), np.array([delta, 0.0]), 0.01)
    assert state.omega / state.v == pytest.approx(kappa, rel=1e-6)


@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_converges_to_a_straight_line_from_an_offset(Model):
    x = np.linspace(0, 400, 801)
    ref = path_to_trajectory(x, np.zeros_like(x), P, v_start=20.0)
    ref.v[:] = 20.0                                       # constant reference speed
    state, err = track_reference(Model, ref, VehicleState(0.0, 2.0, 0.0, 20.0), steps=400)
    assert abs(state.y) < 0.05 and abs(state.psi) < 0.01  # back on the line, aligned with it
    assert state.v == pytest.approx(20.0, abs=0.1)


@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_tracks_a_circle_at_moderate_speed(Model):
    th = np.linspace(0, 1.5 * np.pi, 600)
    x, y = 40 * np.sin(th), 40 * (1 - np.cos(th))
    ref = path_to_trajectory(x, y, P, v_start=15.0)
    ref.v[:] = 15.0                                       # well below the grip limit (24.5 m/s)
    _, err = track_reference(Model, ref, VehicleState(0.0, 0.0, 0.0, 15.0), steps=300)
    assert err.max() < 0.5


def test_acc_speed():
    # At the desired spacing, follow the leader's speed.
    d_des = 0.6 * 30.0 + 5.0
    assert acc_speed(d_des, 30.0, 25.0) == pytest.approx(25.0)
    assert acc_speed(d_des - 10, 30.0, 25.0) < 25.0      # too close: slower than the leader
    assert acc_speed(d_des + 10, 30.0, 25.0) > 25.0      # too far: faster
    assert acc_speed(-50.0, 30.0, 0.0) == 0.0            # never negative
