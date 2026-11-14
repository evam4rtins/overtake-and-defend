"""

Racing line: the global plan of one car.

Geometric approach:
1. Path: the lateral offsets d_i at coarse nodes along the track minimise
       J = k_l * sum(ds_i^2) + k_rho * sum(kappa_i^2)
   (length against curvature), inside the track limits. Solved with L-BFGS-B.
2. Speed: the path is driven with the three-pass speed profile, which gives the segment time.
3. Weights: step 1 is repeated for several ratios k_l / k_rho; the ratio with the lowest
   segment time of this car is kept.

Conventions:
- weight_ratio = k_l / k_rho. 0 is the minimum-curvature line, infinity the minimum-length line.
- d > 0 to the left of the centreline, as everywhere.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from .speed_profile import path_to_trajectory
from .track import CornerTrack
from .types import Trajectory, VehicleParams

# Ratios k_l / k_rho tried by default: curvature only, then 4 values per decade from 1e-6 to 1e-2.
# The useful range depends on the node spacing; these values suit a spacing of a few metres.
DEFAULT_RATIOS = (0.0,) + tuple(float(r) for r in np.logspace(-6.0, -2.0, 17))


# ------------------------------------ Global plan ---------------------------------------
@dataclass
class GlobalPlan:
    """
    Racing line of one car on one track, with its speed profile.

    Attributes:
        s:            np.ndarray, shape (n,). Track stations (= track.s) [m].
        d:            np.ndarray, shape (n,). Lateral offset of the line at each station [m].
        trajectory:   Trajectory. The line driven as fast as the car allows; its s holds the
                      track stations, so trajectory.v[i] is the speed at station s[i].
        weight_ratio: float. Selected k_l / k_rho (nan for a line that was not optimised).
        candidates:   dict. Segment time [s] of every ratio that was tried.
    """
    s: np.ndarray
    d: np.ndarray
    trajectory: Trajectory
    weight_ratio: float = float("nan")
    candidates: Dict[float, float] = field(default_factory=dict)

    @property
    def time(self) -> float:
        """ Time to drive the whole segment [s]. """
        return float(self.trajectory.duration)

    def d_at(self, s):
        """
        Lateral offset of the line at any arc length.

        Args:
            s: float or array-like of float. Arc length along the track centreline [m].

        Returns:
            float or np.ndarray. Offset of the racing line [m], > 0 to the left.
        """
        d = np.interp(s, self.s, self.d)
        return float(d) if np.ndim(d) == 0 else d


def _plan(track: CornerTrack, d: np.ndarray, params: VehicleParams, v_start: float,
          weight_ratio: float = float("nan")) -> GlobalPlan:
    """ Build the plan of a car that follows the offsets d (one per track station). """
    x, y = track.from_frenet(track.s, d)
    trajectory = path_to_trajectory(x, y, params, v_start)
    trajectory.s = track.s.copy()          # index the trajectory by track station
    return GlobalPlan(s=track.s.copy(), d=np.array(d, dtype=float), trajectory=trajectory,
                      weight_ratio=weight_ratio)


def centerline_plan(track: CornerTrack, params: VehicleParams, v_start: float) -> GlobalPlan:
    """
    Baseline plan: the car follows the centreline of the track.

    Args:
        track: CornerTrack.
        params: VehicleParams of the car.
        v_start: float. Speed at the start of the segment [m/s].

    Returns:
        GlobalPlan with d = 0 everywhere.
    """
    return _plan(track, np.zeros(len(track.s)), params, v_start)


def racing_line(track: CornerTrack, params: VehicleParams, v_start: float, node_spacing: float = 4.0,
                ratios: Optional[Sequence[float]] = None, margin: float = 0.3) -> GlobalPlan:
    """
    Minimum-time racing line of one car.

    Args:
        track: CornerTrack.
        params: VehicleParams of the car (its width sets the track limits, the rest the speed).
        v_start: float. Speed at the start of the segment [m/s].
        node_spacing: float. Approximate distance between the optimisation nodes [m].
        ratios: sequence of float, or None. Ratios k_l / k_rho to try (0 = minimum curvature,
                inf = minimum length). None uses DEFAULT_RATIOS.
        margin: float. Safety margin between the side of the car and the track edge [m].

    Returns:
        GlobalPlan of the fastest line; plan.candidates gives the time of every ratio.
    """
    ratios = tuple(float(r) for r in (DEFAULT_RATIOS if ratios is None else ratios))
    d_max = track.d_limits(params.width, margin)[1]
    family = _line_family(_track_key(track), round(d_max, 6), float(node_spacing), ratios)
    plans = [_plan(track, d, params, v_start, ratio) for ratio, d in zip(ratios, family)]
    best = min(plans, key=lambda plan: plan.time)
    best.candidates = {plan.weight_ratio: plan.time for plan in plans}
    return best


# ------------------------------------ Path optimisation ---------------------------------------
def _track_key(track: CornerTrack) -> tuple:
    """ Values that define a track, used as cache key. """
    return (track.l_in, track.radius, track.angle_deg, track.l_out, track.width, float(np.diff(track.s).mean()))


@lru_cache(maxsize=64)
def _line_family(track_key: tuple, d_max: float, node_spacing: float, ratios: tuple) -> Tuple[np.ndarray, ...]:
    """ Offsets at the track stations for every ratio (independent of the car, so cached). """
    track = CornerTrack(*track_key)
    order = sorted(range(len(ratios)), key=lambda i: ratios[i])      # smoothest line first
    lines, d_nodes = [None] * len(ratios), None
    for i in order:
        d_nodes = _optimise_nodes(track, d_max, ratios[i], node_spacing, d_nodes)   # warm start
        lines[i] = _nodes_to_stations(track, d_nodes, d_max, node_spacing)
    return tuple(lines)


def _node_stations(track: CornerTrack, node_spacing: float) -> np.ndarray:
    """ Arc lengths of the optimisation nodes: evenly spaced, both ends of the track included. """
    return np.linspace(0.0, track.length, max(int(round(track.length / node_spacing)), 3) + 1)


def _optimise_nodes(track: CornerTrack, d_max: float, ratio: float, node_spacing: float,
                    d_init: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Step 1: offsets at the nodes that minimise k_l * sum(ds^2) + k_rho * sum(kappa^2).

    Args:
        track: CornerTrack.
        d_max: float. Largest admissible |d| [m].
        ratio: float. k_l / k_rho (0 = curvature only, inf = length only).
        node_spacing: float. Approximate distance between nodes [m].
        d_init: None or np.ndarray. Starting offsets; None starts from the centreline.

    Returns:
        np.ndarray. Optimal offset at each node [m].
    """
    cx, cy, psi, _ = track.at_s(_node_stations(track, node_spacing))
    nx, ny = -np.sin(psi), np.cos(psi)                       # left normal at each node
    k_l, k_rho = (1.0, 0.0) if np.isinf(ratio) else (ratio, 1.0)
    d0 = np.zeros(len(cx)) if d_init is None else d_init
    scale = 1.0 / _cost(np.zeros(len(cx)), cx, cy, nx, ny, k_l, k_rho)[0]   # cost of the centreline = 1

    def objective(d):
        cost, gradient = _cost(d, cx, cy, nx, ny, k_l, k_rho)
        return scale * cost, scale * gradient

    lower, upper = _node_bounds(track, d_max, node_spacing)
    result = minimize(objective, np.clip(d0, lower, upper), jac=True, method="L-BFGS-B",
                      bounds=list(zip(lower, upper)),
                      options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-14, "gtol": 1e-10})
    return result.x


def _node_bounds(track: CornerTrack, d_max: float, node_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lower and upper bound of the offset at each node.

    The limits are only imposed at the nodes. Between two nodes that both sit on the inside
    limit of the corner, the line is flatter than the limit (a circle) and would cut inside it.
    The inside bound of the corner nodes is therefore reduced by the sagitta of that circle
    between two nodes, so that the whole line stays inside the limits.

    Args:
        track: CornerTrack.
        d_max: float. Largest admissible |d| [m].
        node_spacing: float. Approximate distance between nodes [m].

    Returns:
        lower, upper: np.ndarray, shape (n,). Bounds of the offset at each node [m].
    """
    s_nodes = _node_stations(track, node_spacing)
    step = s_nodes[1] - s_nodes[0]
    sagitta = (track.radius - d_max) * (1.0 - np.cos(0.5 * step / track.radius))
    in_corner = (s_nodes > track.corner_start - step) & (s_nodes < track.corner_end + step)
    lower, upper = np.full(len(s_nodes), -d_max), np.full(len(s_nodes), d_max)
    if track.angle > 0:                    # left turn: the inside is the left side (d > 0)
        upper[in_corner] -= sagitta
    else:
        lower[in_corner] += sagitta
    return lower, upper


def _cost(d, cx, cy, nx, ny, k_l: float, k_rho: float) -> Tuple[float, np.ndarray]:
    """
    Cost J = k_l * sum(ds^2) + k_rho * sum(kappa^2) of the path through the nodes, and its gradient.

    The nodes are p_i = c_i + d_i n_i. The curvature at an interior node is the one of the circle
    through the node and its two neighbours (slides 6b): kappa = 2 (u x w) / (|u| |w| |u + w|),
    with u = p_i - p_(i-1) and w = p_(i+1) - p_i.

    Args:
        d: np.ndarray, shape (n,). Offset at each node [m].
        cx, cy: np.ndarray, shape (n,). Centreline point of each node [m].
        nx, ny: np.ndarray, shape (n,). Left normal of each node.
        k_l, k_rho: float. Weights of the length and curvature terms.

    Returns:
        cost: float.
        gradient: np.ndarray, shape (n,). Derivative of the cost with respect to each offset.
    """
    px, py = cx + d * nx, cy + d * ny
    sx, sy = np.diff(px), np.diff(py)                        # segments between nodes
    ux, uy, wx, wy = sx[:-1], sy[:-1], sx[1:], sy[1:]        # segment before / after each interior node
    hx, hy = ux + wx, uy + wy                                # previous node -> next node
    a2, b2, c2 = ux ** 2 + uy ** 2, wx ** 2 + wy ** 2, hx ** 2 + hy ** 2
    abc = np.sqrt(a2 * b2 * c2)
    kappa = 2.0 * (ux * wy - uy * wx) / abc
    cost = k_l * np.sum(sx ** 2 + sy ** 2) + k_rho * np.sum(kappa ** 2)

    # Gradient with respect to the node positions, then projected on the normals.
    gx, gy = np.zeros(len(d)), np.zeros(len(d))
    gx[:-1] -= 2.0 * k_l * sx
    gx[1:] += 2.0 * k_l * sx
    gy[:-1] -= 2.0 * k_l * sy
    gy[1:] += 2.0 * k_l * sy
    f = 2.0 * k_rho * kappa                                  # d(kappa^2) = 2 kappa d(kappa)
    dux = f * (2.0 * wy / abc - kappa * (ux / a2 + hx / c2))   # d(kappa^2) / du
    duy = f * (-2.0 * wx / abc - kappa * (uy / a2 + hy / c2))
    dwx = f * (-2.0 * uy / abc - kappa * (wx / b2 + hx / c2))  # d(kappa^2) / dw
    dwy = f * (2.0 * ux / abc - kappa * (wy / b2 + hy / c2))
    gx[:-2] -= dux                                           # u = p_i - p_(i-1), w = p_(i+1) - p_i
    gx[1:-1] += dux - dwx
    gx[2:] += dwx
    gy[:-2] -= duy
    gy[1:-1] += duy - dwy
    gy[2:] += dwy
    return float(cost), gx * nx + gy * ny


def _nodes_to_stations(track: CornerTrack, d_nodes: np.ndarray, d_max: float, node_spacing: float) -> np.ndarray:
    """
    Offsets at the nodes -> offsets at every track station.

    A cubic spline is drawn through the node points in the plane (x, y), then read back in the
    Frenet frame. Interpolating d(s) directly would copy the curvature jump of the centreline
    (straight -> arc) into the line.

    Args:
        track: CornerTrack.
        d_nodes: np.ndarray. Offset at each node [m].
        d_max: float. Largest admissible |d| [m].
        node_spacing: float. Approximate distance between nodes [m].

    Returns:
        np.ndarray, shape (len(track.s),). Offset at each track station [m], inside the limits.
    """
    x, y = track.from_frenet(_node_stations(track, node_spacing), d_nodes)
    chord = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    fine = np.linspace(0.0, chord[-1], 8 * len(track.s))
    s_fine, d_fine = track.to_frenet(CubicSpline(chord, x)(fine), CubicSpline(chord, y)(fine))
    order = np.argsort(s_fine)
    return np.clip(np.interp(track.s, s_fine[order], d_fine[order]), -d_max, d_max)
