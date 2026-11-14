"""
Path tracking (pure pursuit + speed control) and adaptive cruise control
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from .track import arc_length
from .types import Trajectory, VehicleState, VehicleParams

def curvature_to_steering(kappa: float, vehicle_params: VehicleParams) -> float:
    """
    Steering angle that makes the kinematic bicycle follow a path of curvature kappa.
    """
    beta = np.arcsin(np.clip(kappa * vehicle_params.l_r, -1.0, 1.0))
    delta = np.arctan(vehicle_params.wheelbase / vehicle_params.l_r * np.tan(beta))
    return float(np.clip(delta, -vehicle_params.delta_max, vehicle_params.delta_max))

def curvature_to_action(kappa: float, a: float, v: float, 
                        vehicle_params: VehicleParams, action_names: Sequence[str]) -> np.ndarray:
    """
    Convert (desired curvature, desired longitudinal acceleration) into a model action.
    Args:
        kappa:        float. Desired path curvature [1/m].
        a:            float. Desired longitudinal acceleration [m/s^2].
        v:            float. Current speed [m/s].
        params:       VehicleParams.
        action_names: tuple. The model's action_names: ("a_x", "a_y") or ("delta", "a").

    Returns:
        The action for that model.
    """
    names = tuple(action_names)
    if names == ("a_x", "a_y"):
        # Convert curvature to lateral acceleration
        a_y = kappa * v**2
        return np.array([a, a_y], dtype=np.float32)
    if names == ("delta", "a"):
        delta = curvature_to_steering(kappa, vehicle_params)
        return np.array([delta, a], dtype=np.float32)
    raise ValueError(f"Unknown action_names: {action_names}")

# ------------------------------ Pure Pursuit ------------------------------

class PurePursuit:
    """
    pure pursuit path tracking with proportional speed control.
    
    Args:
        lookahead_min:  Look-ahead distance at zero speed [m].
        lookahead_gain: Extra look-ahead per m/s of speed [s].
        k_v:            Gain of the speed correction [1/s]: a = a_ref + k_v (v_ref - v).
        preview_time:   The reference is read this far ahead [s] (anticipates braking).
    """

    def __init__(self, lookahed_min: float = 6.0, lookahed_gain: float = 0.35, 
                 k_v: float = 2.0, preview_time: float = 0.2):
        self.lookahead_min = lookahed_min
        self.lookahead_gain = lookahed_gain
        self.k_v = k_v
        self.preview_time = preview_time
        self._chache_key = None
        self._chache_s = None

    def _reference_arc_length(self, ref: Trajectory) -> np.ndarray:
        """
        Compute the arc length of the reference trajectory.
        """
        key = (id(ref), len(ref))
        if key != self._chache_key:
            self._chache_key = key
            self._chache_s = arc_length(ref.x, ref.y)
        return self._chache_s
    
    def control(self, state: VehicleState, ref: Trajectory, params: VehicleParams,
                action_names: Sequence[str],  v_cap: Optional[float] = None) -> np.ndarray:
        """
        Compute the action to follow the reference trajectory.
        """
        s_ref = self._reference_arc_length(ref)
        i = int(np.argmin((ref.x - state.x)**2 + (ref.y - state.y)**2))  # closest point on the reference

        # Lateral: pure pursuit on the look-ahead point.
        lookahead = self.lookahead_min + self.lookahead_gain * state.v
        j = min(int(np.searchsorted(s_ref, s_ref[i] + lookahead)), len(s_ref) - 1)
        dx = ref.x[j] - state.x
        dy = ref.y[j] - state.y
        dist = max(np.hypot(dx, dy), 1e-3)
        alpha = np.arctan2(dy, dx) - state.psi
        kappa = 2 * np.sin(alpha) / dist  # curvature of the arc to the lookahead point

        # Longitudinal: feedforward of the reference acceleration + proportional correction,
        k = min(int(np.searchsorted(s_ref, s_ref[i] + self.preview_time * state.v)), len(s_ref) - 2)
        v_ref = float(ref.v[k])
        a_ff = (ref.v[k + 1]**2 - ref.v[k]**2) / (2 * max(s_ref[k + 1] - s_ref[k], 1e-6))
        if v_cap is not None and v_ref > v_cap:    # ACC is the limit: track its speed, no feedforward
            v_ref = v_cap
            a_ff = 0.0
        a = a_ff + self.k_v * (v_ref - state.v)
        return curvature_to_action(kappa, a, state.v, params, action_names)
    
# ------------------------------ ACC ------------------------------ 

def acc_speed(gap: float, v_ego: float, v_lead: float, t_hw: float = 0.6, 
              d_offset: float = 5.0, k_d: float = 0.5) -> float:
    
    """
    Compute the desired speed for adaptive cruise control (ACC).
    
    Args:
        gap:      Bumper-to-bumper distance to the car ahead, along the track [m].
        v_ego:    Own speed [m/s].
        v_lead:   Speed of the car ahead [m/s].
        t_hw:     Time headway [s].
        d_offset: Standstill distance [m].
        k_d:      Gain on the spacing error [1/s].

    Returns:
        v_acc:    Reference speed [m/s], >= 0.
    """

    d_des = d_offset + t_hw * v_ego
    v_acc = v_lead + k_d * (gap - d_des)
    return max(v_acc, 0.0)