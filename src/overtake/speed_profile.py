"""

Three-pass speed profile: the fastest speed a car can hold at every point of a given path.

Trapezoidal velocity planner:
1. Initial guess: the highest speed each point allows on its own, v = sqrt(a_y_max / |kappa|),
   capped by the top speed and by an optional speed limit.
2. Forward pass: starting from the entry speed, the car cannot gain speed faster than the
   acceleration left by the friction ellipse and the engine.
3. Backward pass: starting from the end, the car cannot lose speed faster than the braking
   left by the friction ellipse.

Between two stations the motion is uniformly accelerated: v_next^2 = v^2 + 2 a_x ds.

Conventions:
- s is the arc length along the path itself (not along the track centreline) [m].
- kappa is the signed curvature of the path [1/m]; only its absolute value matters here.

"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from .track import arc_length, curvature, heading
from .types import Trajectory, VehicleParams
from .vehicles import max_longitudinal


def three_pass(s, kappa, params: VehicleParams, v_start: float, v_limit=None, return_passes: bool = False):
    """
    Fastest feasible speed at every station of a path.

    Open-segment boundary condition (proposal, Section 4): the entry speed v_start is imposed.
    If the returned v[0] is lower than v_start, the car arrives too fast to make the path:
    the backward pass is not compatible with the entry speed.

    Args:
        s: array-like of float, shape (N,). Arc length of the stations along the path [m],
           strictly increasing.
        kappa: array-like of float, shape (N,). Curvature of the path at each station [1/m].
        params: VehicleParams. Grip, engine and top speed of the car.
        v_start: float. Speed of the car at the first station [m/s].
        v_limit: None, float or array-like (N,). Extra speed cap [m/s] (e.g. the ACC speed).
        return_passes: bool. If True, also return the two intermediate passes (for plots).

    Returns:
        v: np.ndarray of float, shape (N,). Speed at each station [m/s].
        If return_passes is True: (v_guess, v_forward, v), three arrays of shape (N,).

    Raises:
        ValueError: if s and kappa have different lengths or s is not strictly increasing.
    """
    s, k = np.asarray(s, dtype=float), np.abs(np.asarray(kappa, dtype=float))
    if s.ndim != 1 or s.shape != k.shape or len(s) == 0:
        raise ValueError("s and kappa must be non-empty 1-D arrays of the same length.")
    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("s must be strictly increasing.")
    n = len(s)

    # Pass 1: all the grip goes to cornering.
    with np.errstate(divide="ignore"):
        v_guess = np.minimum(np.sqrt(params.a_y_max / k), params.v_max)   # straight: only the top speed
    if v_limit is not None:
        v_guess = np.minimum(v_guess, v_limit)

    # Pass 2: forward, acceleration limited by what the ellipse and the engine leave.
    v_forward = np.empty(n)
    v_forward[0] = min(v_start, v_guess[0])
    for i in range(n - 1):
        v = v_forward[i]
        a_x = max_longitudinal(v, v * v * k[i], params, accelerate=True)
        v_forward[i + 1] = min(v_guess[i + 1], np.sqrt(v * v + 2.0 * a_x * ds[i]))

    # Pass 3: backward, braking limited by what the ellipse leaves.
    v_final = v_forward.copy()
    for i in range(n - 2, -1, -1):
        v = v_final[i + 1]
        a_x = max_longitudinal(v, v * v * k[i + 1], params, accelerate=False)
        v_final[i] = min(v_forward[i], np.sqrt(v * v + 2.0 * a_x * ds[i]))

    return (v_guess, v_forward, v_final) if return_passes else v_final


def time_stamps(s, v, t0: float = 0.0) -> np.ndarray:
    """
    Time at which the car reaches each station, for uniformly accelerated motion between stations.

    Proposal eq. 3: dt = (-v_i + sqrt(v_i^2 + 2 a ds)) / a. With a = (v_next^2 - v_i^2) / (2 ds)
    the square root is v_next, and the formula becomes dt = 2 ds / (v_i + v_next): the distance
    divided by the mean speed. This form also works when a = 0.

    Args:
        s: array-like of float, shape (N,). Arc length of the stations along the path [m].
        v: array-like of float, shape (N,). Speed at each station [m/s]; two consecutive
           speeds cannot both be zero.
        t0: float. Time at the first station [s].

    Returns:
        t: np.ndarray of float, shape (N,). Time at each station [s]; t[0] = t0.
    """
    s, v = np.asarray(s, dtype=float), np.asarray(v, dtype=float)
    if s.shape != v.shape:
        raise ValueError("s and v must have the same length.")
    mean_speed = 0.5 * (v[:-1] + v[1:])
    if np.any(mean_speed <= 0):
        raise ValueError("The car is stopped on a whole segment: the time is not defined.")
    return t0 + np.concatenate(([0.0], np.cumsum(np.diff(s) / mean_speed)))


def traversal_time(s, v) -> float:
    """
    Total time needed to drive the path with the given speed profile.

    Args:
        s: array-like of float, shape (N,). Arc length of the stations along the path [m].
        v: array-like of float, shape (N,). Speed at each station [m/s].

    Returns:
        float. Time from the first to the last station [s].
    """
    return float(time_stamps(s, v)[-1])


def accelerations(s, v, kappa) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accelerations of a speed profile, to compare with the friction ellipse (g-g diagram).

    Args:
        s: array-like of float, shape (N,). Arc length of the stations along the path [m].
        v: array-like of float, shape (N,). Speed at each station [m/s].
        kappa: array-like of float, shape (N,). Signed curvature at each station [1/m].

    Returns:
        a_x: np.ndarray of float, shape (N,). Longitudinal acceleration between each station
             and the next one [m/s^2]; the last station copies the previous one.
        a_y: np.ndarray of float, shape (N,). Lateral acceleration v^2 kappa at each station
             [m/s^2], > 0 when turning left.
    """
    s, v, kappa = (np.asarray(a, dtype=float) for a in (s, v, kappa))
    a_x = np.zeros(len(s))
    if len(s) > 1:
        a_x[:-1] = np.diff(v ** 2) / (2.0 * np.diff(s))
        a_x[-1] = a_x[-2]
    return a_x, v ** 2 * kappa


def path_to_trajectory(x, y, params: VehicleParams, v_start: float, t0: float = 0.0, v_limit=None) -> Trajectory:
    """
    Turn a path into a time-stamped trajectory driven as fast as the car allows.

    Args:
        x: array-like of float, shape (N,), N >= 3. x coordinates of the ordered points of the path [m].
        y: array-like of float, shape (N,). y coordinates of the same points [m].
        params: VehicleParams. Grip, engine and top speed of the car.
        v_start: float. Speed of the car at the first point [m/s].
        t0: float. Time at the first point [s].
        v_limit: None, float or array-like (N,). Extra speed cap [m/s].

    Returns:
        Trajectory with t, x, y, psi, v at each point; its s is the arc length along the path.
    """
    s = arc_length(x, y)
    v = three_pass(s, curvature(x, y), params, v_start, v_limit)
    return Trajectory(t=time_stamps(s, v, t0), x=x, y=y, psi=heading(x, y), v=v, s=s)
