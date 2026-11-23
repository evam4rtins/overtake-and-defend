"""
Shared code for the local planners.

A local plan is the car's racing line shifted sideways: at each station s ahead of the car we
add an offset o(s), so d(s) = d_race(s) + o(s). Each candidate path gets a three-pass speed
profile and time stamps, which lets us check it against the predicted cars in space and time
and use its travel time as the cost. All candidates of one step are handled together as
(M, N) arrays (M candidates, N stations).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
from scipy.special import expit
from ..control import acc_speed
from ..racing_line import GlobalPlan
from ..sim import footprint_distance
from ..speed_profile import three_pass, time_stamps
from ..track import CornerTrack, heading
from ..types import CarView, Observation, Trajectory, VehicleParams

D_SAFE = 0.5                 # smallest clearance between two footprints [m] (proposal 7.1)
REENTRY_T_HW = 0.3           # re-entry and ACC spacing: t_hw * v + offset (same values as sim.outcome)
REENTRY_OFFSET = 2.0         # [m]
_SIGMA2_MAX = 1.0 / (6.0 * np.sqrt(3.0))   # largest |sigma''(x)| of the logistic function


# ------------------------------------ Horizon and stations ---------------------------------------
def planning_horizon(v: float, params: VehicleParams, passing_distance: float, margin: float = 20.0,
                     minimum: float = 50.0) -> float:
    """
    Length of the planning horizon, scaled with the speed (proposal 5.3).

    H = v^2 / (2 a_x_max) + passing distance + margin: the braking distance to a stop, the
    distance driven while passing, and a margin.

    Args:
        v: float. Current speed [m/s].
        params: VehicleParams of the car (its braking grip a_x_max).
        passing_distance: float. Distance driven during the pass [m] (see passing_distance).
        margin: float. Extra distance [m].
        minimum: float. Shortest horizon [m].

    Returns:
        float. Horizon length along the track [m].
    """
    return float(max(minimum, v * v / (2.0 * params.a_x_max) + passing_distance + margin))


def passing_distance(plan: GlobalPlan, s0: float, passing_time: float) -> float:
    """
    Distance the car drives along its racing line in passing_time seconds from station s0 [m].

    Measured on the racing line rather than at the current speed, so that the horizon does not
    collapse in the corner, where the car is slow but is about to accelerate again. Beyond the
    end of the track, the last speed of the line is kept.
    """
    t_line = plan.trajectory.t
    t_end = float(np.interp(s0, plan.s, t_line)) + passing_time
    if t_end <= t_line[-1]:
        return float(np.interp(t_end, t_line, plan.s) - s0)
    return float(plan.s[-1] - s0 + plan.trajectory.v[-1] * (t_end - t_line[-1]))


def passing_time_estimate(me: CarView, others: Sequence[CarView], t_hw: float = REENTRY_T_HW,
                          d_offset: float = REENTRY_OFFSET, look_ahead: float = 150.0) -> Optional[float]:
    """
    Time needed to pass the closest car ahead at the current speed difference (Approach 0).

    tau = (g + L_O + L_E + m) / dv (proposal 5.1), with g the bumper-to-bumper gap and m the
    re-entry spacing t_hw * v_other + d_offset that the car must have once ahead.

    Returns:
        float [s], or None if there is no car ahead within look_ahead or the car is not faster.
    """
    ahead = [o for o in others if me.s < o.s < me.s + look_ahead]
    if not ahead:
        return None
    other = min(ahead, key=lambda o: o.s)
    dv = me.state.v - other.state.v
    if dv <= 0.5:
        return None
    gap = other.s - me.s - 0.5 * (me.params.length + other.params.length)
    margin = t_hw * other.state.v + d_offset
    return float((max(gap, 0.0) + me.params.length + other.params.length + margin) / dv)


def stations_ahead(track: CornerTrack, s0: float, horizon: float, ds: float = 1.0,
                   min_length: float = 20.0) -> np.ndarray:
    """
    Track stations of a plan: from the car to the end of the horizon.

    The plan stops at the end of the track, but is always at least min_length long (beyond the
    end, CornerTrack extends the exit straight).

    Args:
        track: CornerTrack.
        s0: float. Station of the car [m].
        horizon: float. Horizon length [m].
        ds: float. Approximate spacing of the stations [m].
        min_length: float. Shortest plan [m].

    Returns:
        np.ndarray, shape (N,), N >= 3. Stations from s0 [m].
    """
    end = max(min(s0 + horizon, track.length), s0 + min_length)
    n = max(int(np.ceil((end - s0) / ds)), 2) + 1
    return np.linspace(s0, end, n)


def race_speed(plan: GlobalPlan, s) -> np.ndarray:
    """ Speed of the racing line at track stations s [m/s] (clamped outside the track). """
    return np.interp(s, plan.s, plan.trajectory.v)


# ------------------------------------ Sigmoids ---------------------------------------
def curvature_limit(v: float, params: VehicleParams) -> float:
    """
    Largest curvature the car can drive at speed v [1/m] (slides 6a and 6b).

    rho_max = min(rho_max_steer, a_y_max / v^2), with the steering limit of the kinematic
    bicycle rho_max_steer = 1 / sqrt(l_r^2 + L^2 / tan^2(delta_max)).
    """
    steer = 1.0 / np.hypot(params.l_r, params.wheelbase / np.tan(params.delta_max))
    return float(min(steer, params.a_y_max / max(v, 1e-3) ** 2))


def max_steepness(width: float, v: float, params: VehicleParams, share: float = 1.0) -> float:
    """
    Steepest sigmoid y = w / (1 + exp(-a (s - c))) whose curvature stays feasible (slides 6a).

    On a straight reference the curvature of the shifted path is close to y'', and
    max |y''| = |w| a^2 max|sigma''| = |w| a^2 / (6 sqrt(3)). Setting it to share * rho_max(v)
    gives the steepness a_max of the slides (rho_hat = rho_max).

    Args:
        width: float. Lateral shift w of the sigmoid [m].
        v: float. Speed at which the shift is driven [m/s].
        params: VehicleParams.
        share: float. Part of the curvature limit the shift may use (1 = the whole lateral grip).

    Returns:
        float. Steepness a [1/m]. The shift takes about 8 / a metres (from 2 % to 98 %).
    """
    width = max(abs(width), 1e-3)
    return float(np.sqrt(share * curvature_limit(v, params) / (width * _SIGMA2_MAX)))


def sigmoid_steps(s, centres, steepness, s0: float) -> np.ndarray:
    """
    Logistic steps from 0 to 1, rescaled to start exactly at 0 at the car.

    S(s) = (sigma(a (s - c)) - sigma(a (s0 - c))) / (1 - sigma(a (s0 - c))), so that a plan
    always starts at the current position of the car. A centre c = +inf gives S = 0 everywhere
    (no shift), and a step already completed at s0 gives S = 1 everywhere.

    Args:
        s: array-like, shape (N,). Track stations [m].
        centres: float or array-like (M,). Centres c of the steps [m].
        steepness: float or array-like (M,). Steepness a of the steps [1/m].
        s0: float. Station of the car [m].

    Returns:
        np.ndarray, shape (M, N). Value of each step at each station, in [0, 1].
    """
    s = np.asarray(s, dtype=float)[None, :]
    c = np.atleast_1d(np.asarray(centres, dtype=float))[:, None]
    a = np.broadcast_to(np.atleast_1d(np.asarray(steepness, dtype=float)), (c.shape[0],))[:, None]
    with np.errstate(invalid="ignore"):
        raw = expit(a * (s - c))
        start = expit(a * (s0 - c))
    done = (1.0 - start) < 1e-9
    steps = np.clip((raw - start) / np.where(done, 1.0, 1.0 - start), 0.0, 1.0)
    steps = np.where(done, 1.0, steps)
    return np.where(np.isinf(c) & (c > 0), 0.0, steps)


# ------------------------------------ Offsets -> trajectories ---------------------------------------
def rows_geometry(x: np.ndarray, y: np.ndarray):
    """
    Arc length and signed curvature of a batch of paths, along the last axis.

    Same formulas as track.arc_length and track.curvature (kappa = 4A / (abc) for three
    consecutive points), written for arrays of shape (M, N).

    Returns:
        s: np.ndarray (M, N). Arc length along each path [m].
        kappa: np.ndarray (M, N). Signed curvature [1/m].
    """
    dx, dy = np.diff(x, axis=-1), np.diff(y, axis=-1)
    h = np.maximum(np.hypot(dx, dy), 1e-9)
    s = np.concatenate((np.zeros(x.shape[:-1] + (1,)), np.cumsum(h, axis=-1)), axis=-1)
    c = np.maximum(np.hypot(x[..., 2:] - x[..., :-2], y[..., 2:] - y[..., :-2]), 1e-9)
    twice_area = dx[..., :-1] * dy[..., 1:] - dy[..., :-1] * dx[..., 1:]
    kappa = np.zeros_like(x)
    kappa[..., 1:-1] = 2.0 * twice_area / (h[..., :-1] * h[..., 1:] * c)
    kappa[..., 0], kappa[..., -1] = kappa[..., 1], kappa[..., -2]
    return s, kappa


@dataclass
class Candidates:
    """
    A batch of M candidate plans on the same N track stations, with their speed profile.

    Attributes:
        stations: np.ndarray (N,). Track stations [m].
        offset:   np.ndarray (M, N). Offset from the racing line [m], > 0 to the left.
        d:        np.ndarray (M, N). Frenet d of each path [m].
        x, y:     np.ndarray (M, N). Global position of each path [m].
        s:        np.ndarray (M, N). Arc length along each path [m].
        kappa:    np.ndarray (M, N). Curvature of each path [1/m].
        v:        np.ndarray (M, N). Three-pass speed [m/s].
        t:        np.ndarray (M, N). Time at each station [s].
    """
    stations: np.ndarray
    offset: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    kappa: np.ndarray
    v: np.ndarray
    t: np.ndarray

    def __len__(self) -> int:
        return self.offset.shape[0]

    @property
    def cost(self) -> np.ndarray:
        """ Time each candidate needs to reach the last station [s] (proposal 5.2: time is the cost). """
        return self.t[:, -1] - self.t[:, 0]

    def trajectory(self, m: int) -> Trajectory:
        """ Candidate m as a Trajectory; its s holds the track stations (as in GlobalPlan). """
        x, y = self.x[m], self.y[m]
        return Trajectory(t=self.t[m], x=x, y=y, psi=heading(x, y), v=self.v[m], s=self.stations.copy())

    def within_limits(self, d_min: float, d_max: float, tol: float = 0.02) -> np.ndarray:
        """ (M,) bool: the whole path stays inside the admissible offsets of the car centre. """
        return np.all((self.d >= d_min - tol) & (self.d <= d_max + tol), axis=1)


def offsets_to_candidates(track: CornerTrack, plan: GlobalPlan, stations: np.ndarray, offsets,
                          params: VehicleParams, v_start: float, t0: float, v_limit=None) -> Candidates:
    """
    Offsets from the racing line -> paths -> trajectories, for a batch of candidates.

    d = d_race(s) + offset; (x, y) = track.from_frenet(s, d); arc length and curvature
    along each path; three-pass speed profile and time stamps.

    Args:
        track: CornerTrack.
        plan: GlobalPlan of the car (its racing line).
        stations: np.ndarray (N,). Track stations [m].
        offsets: array-like (N,) or (M, N). Offset of each candidate at each station [m].
        params: VehicleParams of the car.
        v_start: float. Current speed of the car [m/s].
        t0: float. Current time [s].
        v_limit: None or array-like broadcastable to (M, N). Speed cap [m/s].

    Returns:
        Candidates.
    """
    offsets = np.atleast_2d(np.asarray(offsets, dtype=float))
    d = plan.d_at(stations)[None, :] + offsets
    x, y = track.from_frenet(stations, d)
    s, kappa = rows_geometry(x, y)
    v = three_pass(s, kappa, params, v_start, v_limit)
    t = time_stamps(s, v, t0)
    return Candidates(stations=np.asarray(stations, dtype=float), offset=offsets, d=d, x=x, y=y,
                      s=s, kappa=kappa, v=v, t=t)


# ------------------------------------ Predicted cars ---------------------------------------
@dataclass
class PredictedCar:
    """
    Another car seen through its predicted trajectory (prediction.py), also expressed in the track frame.

    Attributes:
        name, params: of the other car.
        trajectory: Trajectory. Prediction, cut where the car reaches the end of the track.
        s, d: np.ndarray. Track station and lateral offset at each predicted time [m].
        s_now, d_now, v_now: float. Current station, offset and speed.
    """
    name: str
    params: VehicleParams
    trajectory: Trajectory
    s: np.ndarray
    d: np.ndarray
    s_now: float
    d_now: float
    v_now: float

    def __post_init__(self):
        n = len(self.s)
        self._s_mono = np.maximum.accumulate(self.s) + 1e-9 * np.arange(n)   # for the inverse s -> t

    def station_at(self, t) -> np.ndarray:
        """ Station at times t [m]; after the prediction the car keeps its last speed. """
        t = np.asarray(t, dtype=float)
        tp = self.trajectory.t
        s = np.interp(t, tp, self.s)
        return np.where(t > tp[-1], self.s[-1] + self.trajectory.v[-1] * (t - tp[-1]), s)

    def lateral_at(self, t) -> np.ndarray:
        return np.interp(t, self.trajectory.t, self.d)

    def speed_at(self, t) -> np.ndarray:
        return np.interp(t, self.trajectory.t, self.trajectory.v)

    def speed_at_station(self, s) -> np.ndarray:
        """ Predicted speed of the car when it is at station s [m/s]. """
        return np.interp(s, self._s_mono, self.trajectory.v)

    def time_at_station(self, s) -> np.ndarray:
        """ Time at which the car reaches station s [s]; extrapolated at its last speed. """
        s = np.asarray(s, dtype=float)
        tp = self.trajectory.t
        t = np.interp(s, self._s_mono, tp)
        v_last = max(float(self.trajectory.v[-1]), 0.1)
        return np.where(s > self._s_mono[-1], tp[-1] + (s - self._s_mono[-1]) / v_last, t)


def predicted_cars(track: CornerTrack, obs: Observation, predictions: Dict[str, Trajectory]) -> List[PredictedCar]:
    """
    Convert the predictions of the other cars to the track frame.

    Each predicted point gets its (s, d) with track.to_frenet, using as hint the current station
    of the car plus the distance along the prediction (removes the ambiguity of the hairpin).
    The exact plan of a car stops at the end of the track (and then stays there), so the
    prediction is cut where the car reaches it.

    Args:
        track: CornerTrack.
        obs: Observation of the planning car.
        predictions: dict name -> predicted Trajectory of that car.

    Returns:
        list of PredictedCar, one per other car that has a prediction.
    """
    cars = []
    for other in obs.others:
        traj = predictions.get(other.name)
        if traj is None or len(traj) < 2:
            continue
        hint = other.s + (traj.s - traj.s[0])
        s, d = track.to_frenet(traj.x, traj.y, s_hint=hint)
        s, d = np.atleast_1d(s), np.atleast_1d(d)
        beyond = np.flatnonzero(s >= track.length - 0.5)
        n = max(int(beyond[0]) + 1 if len(beyond) else len(s), 2)
        cut = Trajectory(t=traj.t[:n], x=traj.x[:n], y=traj.y[:n], psi=traj.psi[:n], v=traj.v[:n])
        cars.append(PredictedCar(other.name, other.params, cut, s[:n], d[:n], float(other.s),
                                 float(other.d), float(other.state.v)))
    return cars


def clearances(c: Candidates, params: VehicleParams, cars: Sequence[PredictedCar]) -> np.ndarray:
    """
    Smallest footprint clearance of each candidate to the predicted cars [m].

    This is sim.trajectory_clearance for a batch: both cars are compared at the same
    instants (the time grid of the prediction) over their common time window, with the
    three-circle footprints of footprint_distance (< 0 = collision).

    Returns:
        np.ndarray (M,). inf when no car overlaps in time.
    """
    out = np.full(len(c), np.inf)
    if not cars:
        return out
    psi = np.unwrap(np.arctan2(np.gradient(c.y, axis=-1), np.gradient(c.x, axis=-1)), axis=-1)
    for car in cars:
        tp, pred = car.trajectory.t, car.trajectory
        for m in range(len(c)):
            inside = (tp >= c.t[m, 0]) & (tp <= c.t[m, -1])
            if not np.any(inside):
                continue
            tt = tp[inside]
            xe, ye = np.interp(tt, c.t[m], c.x[m]), np.interp(tt, c.t[m], c.y[m])
            pe = np.interp(tt, c.t[m], psi[m])
            gap = footprint_distance(xe, ye, pe, params, pred.x[inside], pred.y[inside], pred.psi[inside], car.params)
            out[m] = min(out[m], float(np.min(gap)))
    return out


def _gaps(c: Candidates, params: VehicleParams, car: PredictedCar, t_hw: float, d_offset: float):
    """ Gap ahead of the car (ego rear -> other front), lateral overlap and required re-entry gap. """
    s_o = car.station_at(c.t)
    gap = c.stations[None, :] - s_o - 0.5 * (params.length + car.params.length)
    overlap = np.abs(c.d - car.lateral_at(c.t)) < 0.5 * (params.width + car.params.width)
    required = t_hw * car.speed_at(c.t) + d_offset
    return gap, overlap, required


def reentry_ok(c: Candidates, params: VehicleParams, cars: Sequence[PredictedCar],
               t_hw: float = REENTRY_T_HW, d_offset: float = REENTRY_OFFSET) -> np.ndarray:
    """
    (M,) bool: the candidate never cuts in too close in front of a predicted car.

    Same rule as sim.outcome (proposal 3.5): whenever the car is ahead of the other one and
    laterally overlaps it, the gap must be at least t_hw * v_other + d_offset. A car that is
    already in front, in the same lane, at the start of the plan is not checked: it cannot undo
    its earlier re-entry.
    """
    ok = np.ones(len(c), dtype=bool)
    for car in cars:
        gap, overlap, required = _gaps(c, params, car, t_hw, d_offset)
        if gap[0, 0] > 0 and overlap[0, 0]:
            continue
        ok &= ~np.any((gap > 0) & overlap & (gap < required), axis=1)
    return ok


def earliest_reentry(c: Candidates, params: VehicleParams, cars: Sequence[PredictedCar],
                     t_hw: float = REENTRY_T_HW, d_offset: float = REENTRY_OFFSET) -> np.ndarray:
    """
    First station index where the car is far enough ahead of every predicted car to re-enter.

    For each candidate (a path that has moved out and stays out), the re-entry may start at
    index k if the gap ahead of each car is at least t_hw * v_other + d_offset there
    (proposal 3.5). Only the first such station is looked for: the out path is not driven after
    the return, so its later part (often slow, or off the track in the corner) does not matter.
    The complete path, with its return, is checked afterwards by reentry_ok.

    Returns:
        np.ndarray (M,) of int. Index of the earliest safe re-entry, or -1 if there is none.
    """
    k_star = np.zeros(len(c), dtype=int)
    for car in cars:
        gap, _, required = _gaps(c, params, car, t_hw, d_offset)
        good = gap >= required
        first = np.where(good.any(axis=1), np.argmax(good, axis=1), -1)
        k_star = np.where((k_star < 0) | (first < 0), -1, np.maximum(k_star, first))
    return k_star


# ------------------------------------ ACC ---------------------------------------
def acc_cap(me: CarView, others: Sequence[CarView], t_hw: float = REENTRY_T_HW,
            d_offset: float = REENTRY_OFFSET) -> Optional[float]:
    """
    ACC speed behind the closest car ahead in the same lane, or None (slides 6a, control.acc_speed).

    A car is in the same lane when the lateral distance is smaller than the half widths plus
    0.5 m. This is the cap min(v_opt, v_ACC) that every attempt keeps (proposal 5.4), so that
    an aborted attempt degrades into following instead of a collision.
    """
    caps = []
    for other in others:
        if other.s <= me.s or abs(other.d - me.d) > 0.5 * (me.params.width + other.params.width) + 0.5:
            continue
        gap = other.s - me.s - 0.5 * (me.params.length + other.params.length)
        caps.append(acc_speed(gap, me.state.v, other.state.v, t_hw, d_offset))
    return min(caps) if caps else None


# ------------------------------------ Planner interface ---------------------------------------
@dataclass
class Decision:
    """
    What a planner chose at one planning step (to log the commit points, proposal 5.4).

    Attributes:
        t: float. Time of the planning step [s].
        kind: str. "follow" (racing line + ACC) or "overtake".
        width: float. Target offset w from the racing line [m] (> 0 left), 0 when following.
        c1, c2: float. Centres of the lane change and of the return [m] (inf: no return planned).
        cost: float. Time to the end of the horizon of the chosen plan [s].
        follow_cost: float. Same for the follow option [s].
        n_candidates, n_feasible: int. Number of candidates evaluated and of feasible ones.
    """
    t: float
    kind: str
    width: float = 0.0
    c1: float = float("nan")
    c2: float = float("nan")
    cost: float = float("nan")
    follow_cost: float = float("nan")
    n_candidates: int = 1
    n_feasible: int = 1


@dataclass
class Frame:
    """ Quantities shared by all the candidates of one planning step. """
    t0: float
    s0: float
    v0: float
    o0: float                 # current offset of the car from its racing line [m]
    stations: np.ndarray      # (N,) track stations of the plan [m]
    v_limit: np.ndarray       # (N,) terminal cap: racing-line speed at the last station [m/s]


class LocalPlanner:
    """
    Interface of the local planners: plan(obs, predictions) -> Trajectory.

    A planner is attached to one car with setup(track, params, global_plan). At every planning
    step it receives the observation of the car and the predicted trajectories of the others
    (dict name -> Trajectory, from prediction.py), and returns its plan as a Trajectory in
    absolute time, starting at the car. Its last choice is in self.decision.

    Args:
        ds: float. Spacing of the plan stations [m].
        passing_time, horizon_margin: float. Horizon parameters (see planning_horizon and
            passing_distance). passing_time is the smallest time allowed for a pass; when the car
            is faster than the car ahead, the estimate of Approach 0 is used instead, up to
            max_passing_time.
        max_passing_time: float. Longest passing time used in the horizon [s].
        track_margin: float. Safety margin e to the track edge [m] (proposal eq. 1).
        d_safe: float. Smallest clearance to another car [m].
        t_hw, d_offset: float. ACC and re-entry spacing t_hw * v + d_offset [s], [m].
        curvature_share: float. Part of the curvature limit a lane change may use.
        offset_deadband: float. A car closer than this to its racing line is planned from the
                         line itself [m]: correcting small tracking errors is the job of the
                         controller, and a steep sigmoid for a few centimetres would use the grip
                         the racing line already needs in the corner.
    """

    name = "local"

    def __init__(self, ds: float = 1.0, passing_time: float = 6.0, horizon_margin: float = 20.0,
                 max_passing_time: float = 8.0,
                 track_margin: float = 0.3, d_safe: float = D_SAFE, t_hw: float = REENTRY_T_HW,
                 d_offset: float = REENTRY_OFFSET, curvature_share: float = 1.0,
                 offset_deadband: float = 0.75):
        self.ds = ds
        self.passing_time = passing_time
        self.max_passing_time = max_passing_time
        self.horizon_margin = horizon_margin
        self.track_margin = track_margin
        self.d_safe = d_safe
        self.t_hw = t_hw
        self.d_offset = d_offset
        self.curvature_share = curvature_share
        self.offset_deadband = offset_deadband
        self.track: Optional[CornerTrack] = None
        self.decision: Optional[Decision] = None

    def options(self) -> dict:
        """ Constructor arguments of the base class (to build a sub-planner with the same settings). """
        return dict(ds=self.ds, passing_time=self.passing_time, horizon_margin=self.horizon_margin,
                    max_passing_time=self.max_passing_time,
                    track_margin=self.track_margin, d_safe=self.d_safe, t_hw=self.t_hw,
                    d_offset=self.d_offset, curvature_share=self.curvature_share,
                    offset_deadband=self.offset_deadband)

    def setup(self, track: CornerTrack, params: VehicleParams, global_plan: GlobalPlan) -> None:
        """ Attach the planner to a car: its track, its parameters and its racing line. """
        if global_plan is None:
            raise ValueError("A local planner needs the global plan (racing line) of its car.")
        self.track, self.params, self.global_plan = track, params, global_plan
        self.d_min, self.d_max = track.d_limits(params.width, self.track_margin)
        self.reset()

    def reset(self) -> None:
        """ Forget the previous decisions (new run). """
        self.decision = None

    def plan(self, obs: Observation, predictions: Dict[str, Trajectory]) -> Trajectory:
        raise NotImplementedError

    # ---- helpers for the subclasses ----
    def frame(self, obs: Observation) -> Frame:
        """ Current station, speed and offset of the car, and the stations of the horizon. """
        me = obs.me
        v0 = max(float(me.state.v), 0.5)
        tau = passing_time_estimate(me, obs.others, self.t_hw, self.d_offset)
        passing = self.passing_time if tau is None else float(np.clip(tau, self.passing_time, self.max_passing_time))
        horizon = planning_horizon(v0, self.params, passing_distance(self.global_plan, float(me.s), passing),
                                   self.horizon_margin)
        stations = stations_ahead(self.track, float(me.s), horizon, self.ds)
        o0 = float(me.d) - float(self.global_plan.d_at(me.s))
        if abs(o0) < self.offset_deadband:
            o0 = 0.0
        v_limit = np.full(len(stations), np.inf)
        v_limit[-1] = race_speed(self.global_plan, stations[-1])
        return Frame(float(obs.t), float(me.s), v0, o0, stations, v_limit)

    def evaluate(self, f: Frame, offsets, v_limit=None) -> Candidates:
        """ Offsets -> Candidates, with the terminal cap of the frame and an optional extra cap. """
        cap = f.v_limit if v_limit is None else np.minimum(f.v_limit, v_limit)
        return offsets_to_candidates(self.track, self.global_plan, f.stations, offsets, self.params,
                                     f.v0, f.t0, cap)

    def relevant_cars(self, obs: Observation, predictions: Dict[str, Trajectory], f: Frame) -> List[PredictedCar]:
        """
        Predicted cars that matter for this plan: ahead within the horizon, alongside, or just
        behind (closer than the re-entry spacing, so that the car does not cut in front of them).
        """
        cars = []
        for car in predicted_cars(self.track, obs, predictions):
            behind_zone = 0.5 * (self.params.length + car.params.length) + self.t_hw * car.v_now + self.d_offset + 5.0
            if f.s0 - behind_zone <= car.s_now <= f.stations[-1]:
                cars.append(car)
        return cars

    def checked_cars(self, cars: Sequence[PredictedCar], f: Frame) -> List[PredictedCar]:
        """
        Cars to check for clearance: all relevant cars except those entirely behind in the same
        lane, which follow this car and keep their own distance (slides 6a: ACC).
        """
        out = []
        for car in cars:
            behind = car.s_now < f.s0 - 0.5 * (self.params.length + car.params.length)
            same_lane = abs(car.d_now - (float(self.global_plan.d_at(f.s0)) + f.o0)) < \
                0.5 * (self.params.width + car.params.width)
            if not (behind and same_lane):
                out.append(car)
        return out

    def feasible(self, c: Candidates, cars: Sequence[PredictedCar], f: Frame, check_start: bool = True,
                 v_cap=None) -> np.ndarray:
        """
        (M,) bool: within the track limits, collision free and with a safe re-entry
        (slides 6a: "within limits and collision free").

        check_start also requires that the path does not need to brake instantly at the start:
        the three-pass speed at the car is not lower than its speed, or than the speed cap v_cap
        (ACC behind a leader) when there is one, since that braking is the job of the ACC.
        """
        ok = c.within_limits(self.d_min, self.d_max)
        if check_start:
            allowed = np.full(len(c), f.v0)
            if v_cap is not None:
                allowed = np.minimum(allowed, np.broadcast_to(v_cap, c.v.shape)[:, 0])
            ok &= c.v[:, 0] >= allowed - 0.5
        checked = self.checked_cars(cars, f)
        ok &= clearances(c, self.params, checked) >= self.d_safe
        ok &= reentry_ok(c, self.params, checked, self.t_hw, self.d_offset)
        return ok
