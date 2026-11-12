"""
Point Mass model, Kinematic Bicycle model, and Friction Ellipse constraints for vehicle dynamics.

Conventions from slides:
- accelerations in the car frame
- friction ellipse : (a_x / a_x_max)^2 + (a_y / a_y_max)^2 <= 1
- engine limit : when accelerating a_x <= a_engine * (1 - v / v_max)
- point mass action : (a_x, a_y) [m/s^2, m/s^2]
- kinmatic bicycle action : (a, delta) [m/s^2, rad]
- every acceleration is projected to the friction ellipse and engine limit 
before being applied, so no model can use more grip than the car has

"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from .types import VehicleState, VehicleParams

# -------------------------- Friction Ellipse --------------------------------

def max_longitudinal(v, a_y, params: VehicleParams, accelerate: bool):
    ratio = np.clip(np.abs(a_y) / params.a_y_max, 0.0, 1.0)
    residual = params.a_x_max * np.sqrt(1.0 - ratio**2)
    if accelerate:
        engine = params.a_engine * np.clip(1.0 - np.asarray(v, dtype=np.float64) / params.v_max, 0.0, None)
        residual = np.minimum(residual, engine)
    return float(residual) if np.ndim(residual) == 0 else residual


def project_to_limits(a_x: float, a_y: float, v: float, params: VehicleParams) -> Tuple[float, float]:
    a_y = float(np.clip(a_y, -params.a_y_max, params.a_y_max))
    limit = max_longitudinal(v, a_y, params, accelerate=a_x > 0)
    a_x = float(np.clip(a_x, -limit, limit))
    return a_x, a_y


# ----------------------------- Point Mass -------------------------------------

class PointMass:

    action_names = ("a_x", "a_y")

    def __init__(self, params: VehicleParams, v_min: float = 0.5):
        self.params = params
        self.v_min = v_min
    
    def step(self, state: VehicleState, action, dt: float) -> VehicleState:
        """ Step the point mass dynamics forward by dt seconds. """
        p = self.params
        a_x, a_y = project_to_limits(float(action[0]), float(action[1]), state.v, p)
        omega = a_y / max(state.v, self.v_min)
        v_new = float(np.clip(state.v + a_x * dt, 0.0, p.v_max))
        if dt > 0:
            a_x = (v_new - state.v) / dt   # acceleration really applied
        # mid-point integration: mean speed and mean heading over the step.
        v_mid = 0.5 * (state.v + v_new)
        psi_mid = state.psi + 0.5 * omega * dt
        return VehicleState(
            x = state.x + v_mid * np.cos(psi_mid) * dt,
            y = state.y + v_mid * np.sin(psi_mid) * dt,
            psi = state.psi + omega * dt,
            v = v_new,
            omega = omega,
            a_x = a_x,
            a_y = a_y
        )
    
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the lower and upper bounds on the action space. """
        p = self.params
        return (np.array([-p.a_x_max, -p.a_y_max]), np.array([p.a_engine, p.a_y_max]))


# -------------------------- Kinematic Bicycle Model ---------------------------

class KinematicBicycle:

    action_names = ("delta", "a")

    def __init__(self, params: VehicleParams, substeps: int = 4):
        self.params = params
        self.substeps = substeps

    def slip_angle(self, delta): 
        p = self.params
        return np.arctan(p.l_r / p.wheelbase * np.tan(delta))

    def step(self, state: VehicleState, action, dt: float) -> VehicleState:
        """ Advance the car by dt seconds """
        p = self.params
        delta = float(np.clip(action[0], -p.delta_max, p.delta_max))
        a_request = float(action[1])
        h = dt / self.substeps
        x, y, psi, v = state.x, state.y, state.psi, state.v
        omega = a_x = a_y = 0.0
        for _ in range(self.substeps):
            beta = self.slip_angle(delta)
            omega = v / p.l_r * np.sin(beta)
            a_y_needed = v * omega
            a_x, a_y = project_to_limits(a_request, a_y_needed, v, p)
            if abs(a_y) < abs(a_y_needed):
                omega *= a_y / a_y_needed
                beta = np.arcsin(np.clip(omega * p.l_r / v, -1.0, 1.0))
            x += v * np.cos(psi + beta) * h
            y += v * np.sin(psi + beta) * h
            psi += omega * h
            v_new = float(np.clip(v + a_x * h, 0.0, p.v_max))
            a_x = (v_new - v) / h if h > 0 else a_x
            v = v_new
        return VehicleState(x = x, y = y, psi = psi, v = v, omega = omega, a_x = a_x, a_y = a_y)
    
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the lower and upper bounds on the action space. """
        p = self.params
        return (np.array([-p.delta_max, -p.a_x_max]), np.array([p.delta_max, p.a_engine]))