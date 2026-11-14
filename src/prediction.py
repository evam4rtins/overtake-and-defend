"""
Prediction of other cars' future trajectories, given their current state and plan.
"""

from __future__ import annotations
from dataclasses import replace
import numpy as np
from .types import CarView, Trajectory

class _Predictor:
    def __init__(self, horizon: float = 4.0, dt: float = 0.05):
        self.horizon = horizon
        self.dt = dt

    def times(self, t: float) -> np.ndarray:
        n = int(round(self.horizon / self.dt)) + 1
        return t + np.arange(n) * self.dt
    
class CoordinatedTurn(_Predictor):
    """
    Constant speed v and constant yaw rate omega, integrated exactly:
        psi(tau) = psi0 + omega tau
        x(tau)   = x0 + v/omega (sin(psi(tau)) - sin(psi0))     
        y(tau)   = y0 - v/omega (cos(psi(tau)) - cos(psi0))

    Args:
        horizon: float. Prediction horizon [s].
        dt:      float. Time step of the predicted trajectory [s].
    """

    def predict(self, car: CarView, t: float) -> Trajectory:
        st = car.state
        times = self.times(t)
        tau = times - t
        psi = st.psi + st.omega * tau
        if abs(st.omega) < 1e-6:
            x = st.x + st.v * tau * np.cos(st.psi)
            y = st.y + st.v * tau * np.sin(st.psi)
        else:
            r = st.v / st.omega
            x = st.x + r * (np.sin(psi) - np.sin(st.psi))
            y = st.y - r * (np.cos(psi) - np.cos(st.psi))
        return Trajectory(
            times=times,
            x=x,
            y=y,
            psi=psi,
            v=np.full_like(times, st.v)
        )

class ConstantVelocity(CoordinatedTurn):
    """ Straight line at constant speed: the coordinated turn with omega = 0. """

    def predict(self, car: CarView, t: float) -> Trajectory:
        return super().predict(replace(car, state=replace(car.state, omega=0.0)), t)
    
class ExactPlan(_Predictor):
    """ The other car's current plan, sampled on the prediction grid (perfect knowledge). """
    
    def predict(self, car: CarView, t: float) -> Trajectory:
        if car.plan is None:
            return CoordinatedTurn(self.horizon, self.dt).predict(car, t)
        times = self.times(t)
        x, y, psi, v = car.plan.sample(times)
        return Trajectory(
            times=times,
            x=x,
            y=y,
            psi=psi,
            v=v
        )
    
PREDICTORS = {"constant_velocity": ConstantVelocity, "coordinated_turn": CoordinatedTurn, "exact": ExactPlan}

