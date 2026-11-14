import numpy as np
import pytest

from overtake.prediction import ConstantVelocity, CoordinatedTurn, ExactPlan
from overtake.types import CarView, Trajectory, VehicleParams, VehicleState

P = VehicleParams()


def view(state, plan=None):
    return CarView("other", state, 0.0, 0.0, P, plan)


def test_constant_velocity_goes_straight():
    pred = ConstantVelocity(horizon=2.0).predict(view(VehicleState(0, 0, np.pi / 4, 10.0, omega=0.3)), t=5.0)
    assert pred.t[0] == 5.0 and pred.t[-1] == pytest.approx(7.0)
    assert (pred.x[-1], pred.y[-1]) == pytest.approx((20 / np.sqrt(2), 20 / np.sqrt(2)))


def test_coordinated_turn_follows_a_circle():
    v, omega = 10.0, 0.25                                   # radius 40 m, centre (0, 40)
    pred = CoordinatedTurn(horizon=4.0).predict(view(VehicleState(0, 0, 0, v, omega=omega)), t=0.0)
    assert np.allclose(np.hypot(pred.x, pred.y - 40.0), 40.0)
    assert pred.psi[-1] == pytest.approx(omega * 4.0)


def test_exact_plan_and_fallback():
    t = np.linspace(0, 10, 101)
    plan = Trajectory(t=t, x=5 * t, y=np.zeros_like(t), psi=np.zeros_like(t), v=np.full_like(t, 5.0))
    pred = ExactPlan(horizon=2.0).predict(view(VehicleState(0, 0, 0, 5.0), plan), t=3.0)
    assert pred.x[0] == pytest.approx(15.0) and pred.x[-1] == pytest.approx(25.0)
    no_plan = ExactPlan(horizon=2.0).predict(view(VehicleState(0, 0, 0, 5.0, omega=0.0)), t=0.0)
    assert no_plan.x[-1] == pytest.approx(10.0)            # falls back to the coordinated turn
