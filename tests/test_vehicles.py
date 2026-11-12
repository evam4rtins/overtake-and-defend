import numpy as np
import pytest

from overtake.types import VehicleParams, VehicleState
from overtake.vehicles import KinematicBicycle, PointMass, max_longitudinal, project_to_limits

P = VehicleParams()


def inside_ellipse(a_x, a_y, p=P):
    return (a_x / p.a_x_max) ** 2 + (a_y / p.a_y_max) ** 2 <= 1 + 1e-9


def run(model, state, action, dt=0.01, steps=100):
    for _ in range(steps):
        state = model.step(state, np.asarray(action, dtype=float), dt)
    return state


# ------------------------------------ friction ellipse  ---------------------------------------
def test_residual_braking():
    assert max_longitudinal(30.0, 0.0, P, accelerate=False) == pytest.approx(P.a_x_max)
    assert max_longitudinal(30.0, P.a_y_max, P, accelerate=False) == pytest.approx(0.0)
    # Half the lateral grip leaves sqrt(1 - 0.25) of the braking grip.
    assert max_longitudinal(30.0, 0.5 * P.a_y_max, P, accelerate=False) == pytest.approx(P.a_x_max * np.sqrt(0.75))


def test_engine_limit():
    assert max_longitudinal(0.0, 0.0, P, accelerate=True) == pytest.approx(P.a_engine)
    assert max_longitudinal(P.v_max / 2, 0.0, P, accelerate=True) == pytest.approx(P.a_engine / 2)
    assert max_longitudinal(P.v_max, 0.0, P, accelerate=True) == pytest.approx(0.0)
    # Near the grip limit the ellipse, not the engine, is the limit.
    assert max_longitudinal(10.0, 0.99 * P.a_y_max, P, accelerate=True) < P.a_engine * (1 - 10 / P.v_max)


def test_max_longitudinal_works_on_arrays():
    v = np.array([0.0, 37.5, 75.0])
    assert np.allclose(max_longitudinal(v, np.zeros(3), P, accelerate=True), [8.0, 4.0, 0.0])


@pytest.mark.parametrize("a_x,a_y,v", [(20, 20, 30), (-30, 5, 30), (5, -40, 30), (-15, 15, 10), (8, 0, 74)])
def test_projection_is_admissible(a_x, a_y, v):
    bx, by = project_to_limits(a_x, a_y, v, P)
    assert inside_ellipse(bx, by)
    assert bx <= max_longitudinal(v, by, P, accelerate=True) + 1e-9


def test_projection_keeps_admissible_requests():
    assert project_to_limits(-5.0, 6.0, 30.0, P) == pytest.approx((-5.0, 6.0))


def test_projection_keeps_lateral_priority():
    a_x, a_y = project_to_limits(-15.0, 15.0, 30.0, P)   # full braking + full cornering requested
    assert a_y == pytest.approx(15.0) and a_x == pytest.approx(0.0)


# ------------------------------------ vehicle models  ---------------------------------------
@pytest.mark.parametrize("Model,left", [(PointMass, [0.0, 5.0]), (KinematicBicycle, [0.05, 0.0])])
def test_left_input_turns_left_within_limits(Model, left):
    model, state = Model(P), VehicleState(0.0, 0.0, 0.0, 20.0)
    for _ in range(50):
        state = model.step(state, np.array(left), 0.02)
        assert inside_ellipse(state.a_x, state.a_y)
    assert state.y > 0 and state.psi > 0


@pytest.mark.parametrize("Model,brake", [(PointMass, [-50.0, 0.0]), (KinematicBicycle, [0.0, -50.0])])
def test_braking_is_limited_and_stops_at_zero(Model, brake):
    model = Model(P)
    state = model.step(VehicleState(0.0, 0.0, 0.0, 30.0), np.array(brake), 0.1)
    assert state.v == pytest.approx(30.0 - P.a_x_max * 0.1)
    state = run(model, state, brake, dt=0.1, steps=50)
    assert state.v == 0.0 and state.a_x == 0.0                       # stops, never reverses


@pytest.mark.parametrize("Model,throttle", [(PointMass, [20.0, 0.0]), (KinematicBicycle, [0.0, 20.0])])
def test_top_speed_is_never_exceeded(Model, throttle):
    state = run(Model(P), VehicleState(0.0, 0.0, 0.0, 70.0), throttle, dt=0.1, steps=500)
    assert state.v <= P.v_max + 1e-9


def test_point_mass_drives_a_circle_of_radius_v2_over_ay():
    v, a_y = 20.0, 10.0
    radius = v ** 2 / a_y
    period = 2 * np.pi * radius / v
    model, state = PointMass(P), VehicleState(0.0, 0.0, 0.0, v)
    n = 1000
    for _ in range(n):
        state = model.step(state, np.array([0.0, a_y]), period / n)
        assert np.hypot(state.x, state.y - radius) == pytest.approx(radius, rel=1e-3)
    assert (state.x, state.y) == pytest.approx((0.0, 0.0), abs=0.05)   # back at the start


def test_bicycle_steady_state_yaw_rate():
    """ At low speed (grip not saturated) psi_dot = v / l_r * sin(beta). """
    model, v, delta = KinematicBicycle(P), 10.0, 0.1
    state = model.step(VehicleState(0.0, 0.0, 0.0, v), np.array([delta, 0.0]), 0.01)
    beta = model.slip_angle(delta)
    assert state.omega == pytest.approx(v / P.l_r * np.sin(beta))
    assert state.a_y == pytest.approx(v * state.omega)


def test_bicycle_understeers_when_grip_is_exceeded():
    model, v = KinematicBicycle(P), 40.0
    state = model.step(VehicleState(0.0, 0.0, 0.0, v), np.array([P.delta_max, 0.0]), 0.05)
    assert state.a_y == pytest.approx(P.a_y_max)                     # saturated, not more
    assert state.omega == pytest.approx(P.a_y_max / v)               # yaw rate reduced accordingly


def test_steering_is_clipped():
    model = KinematicBicycle(P)
    s0 = VehicleState(0.0, 0.0, 0.0, 5.0)
    a = model.step(s0, np.array([10.0, 0.0]), 0.05)
    b = model.step(s0, np.array([P.delta_max, 0.0]), 0.05)
    assert (a.x, a.y, a.psi) == pytest.approx((b.x, b.y, b.psi))


@pytest.mark.parametrize("Model", [PointMass, KinematicBicycle])
def test_action_bounds(Model):
    low, high = Model(P).action_bounds()
    assert low.shape == high.shape == (2,) and np.all(low < high)
