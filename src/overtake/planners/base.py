"""
Shared code for the local planners.

The idea: a local plan is just our racing line pushed sideways. At each station s in front of
the car we add an offset o(s), so the path is d(s) = d_race(s) + o(s). For every candidate path
we compute a speed profile and the time at which we reach each station. With those times we can
check the path against where the other cars will be, and the time to the end is the cost.

Several candidates are evaluated at once: they all share the same stations, so we keep them in
arrays of shape (M, N) - M candidates, N stations.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from ..control import acc_speed
from ..sim import footprint_distance
from ..speed_profile import three_pass, time_stamps
from ..track import heading
from ..types import Trajectory

# Safety settings (same values as sim.outcome for the re-entry rule)
D_SAFE = 0.5            # minimum distance between two cars [m]
REENTRY_T_HW = 0.3      # once ahead, we need t_hw * v + offset in front of the other car
REENTRY_OFFSET = 2.0    # [m]

# The logistic function sigma(x) bends the most where |sigma''| = 1 / (6 sqrt(3))
_SIGMA2_MAX = 1.0 / (6.0 * np.sqrt(3.0))


# ---------------------------------------------------------------------------
# How far ahead do we plan?
# ---------------------------------------------------------------------------

def planning_horizon(v, params, passing_distance, margin=20.0, minimum=50.0):
    """
    How far ahead we plan [m]: enough to brake to a stop, plus the distance needed to pass,
    plus a margin. At 60 m/s the braking alone is already ~120 m, so a fixed 40 m horizon
    (like in the 6c examples) would not work here.
    """
    braking = v * v / (2.0 * params.a_x_max)
    return float(max(minimum, braking + passing_distance + margin))


def passing_distance(plan, s0, passing_time):
    """
    Distance we would drive on our racing line during `passing_time` seconds, starting at s0.

    We use the racing line rather than the current speed: in the corner the car is slow, but it
    is about to accelerate, and we don't want the horizon to shrink right there.
    """
    t_line = plan.trajectory.t
    t_start = float(np.interp(s0, plan.s, t_line))
    t_end = t_start + passing_time

    if t_end <= t_line[-1]:
        return float(np.interp(t_end, t_line, plan.s) - s0)

    # past the end of the track: just keep going at the last speed
    extra_time = t_end - t_line[-1]
    return float(plan.s[-1] - s0 + plan.trajectory.v[-1] * extra_time)


def passing_time_estimate(me, others, t_hw=REENTRY_T_HW, d_offset=REENTRY_OFFSET, look_ahead=150.0):
    """
    Rough time to pass the car in front if the speed difference stays the same (Approach 0):

        tau = (gap + both car lengths + re-entry spacing) / dv

    Returns None if nobody is ahead (within look_ahead) or if we are not really faster.
    """
    ahead = [o for o in others if me.s < o.s < me.s + look_ahead]
    if not ahead:
        return None

    other = min(ahead, key=lambda o: o.s)     # the closest one
    dv = me.state.v - other.state.v
    if dv <= 0.5:
        return None                           # not faster: no pass to plan for

    gap = other.s - me.s - 0.5 * (me.params.length + other.params.length)
    spacing = t_hw * other.state.v + d_offset
    distance_to_gain = max(gap, 0.0) + me.params.length + other.params.length + spacing
    return float(distance_to_gain / dv)


def stations_ahead(track, s0, horizon, ds=1.0, min_length=20.0):
    """
    Stations of the plan, from the car up to the end of the horizon (about one every ds metres).
    We stop at the end of the track, but always keep at least min_length metres of plan.
    """
    end = min(s0 + horizon, track.length)
    end = max(end, s0 + min_length)
    n_points = max(int(np.ceil((end - s0) / ds)), 2) + 1
    return np.linspace(s0, end, n_points)


def race_speed(plan, s):
    """ Speed of our racing line at stations s. """
    return np.interp(s, plan.s, plan.trajectory.v)


# ---------------------------------------------------------------------------
# Sigmoids (slides 6a)
# ---------------------------------------------------------------------------

def curvature_limit(v, params):
    """
    Tightest curve the car can drive at speed v [1/m]: limited by the steering angle at low
    speed, and by the lateral grip (a_y_max / v^2) at high speed.
    """
    steering_limit = 1.0 / np.hypot(params.l_r, params.wheelbase / np.tan(params.delta_max))
    grip_limit = params.a_y_max / max(v, 1e-3) ** 2
    return float(min(steering_limit, grip_limit))


def max_steepness(width, v, params, share=1.0):
    """
    Steepest sigmoid y = w / (1 + exp(-a (s - c))) the car can still drive at speed v.

    On a straight, the curvature of the shifted path is about y'', whose peak is
    |w| a^2 / (6 sqrt(3)). We set that peak equal to the curvature limit (times `share`) and
    solve for a. A lane change then takes roughly 8 / a metres.
    """
    width = max(abs(width), 1e-3)
    rho_max = share * curvature_limit(v, params)
    return float(np.sqrt(rho_max / (width * _SIGMA2_MAX)))


def sigmoid_steps(s, centres, steepness, s0):
    """
    One logistic step (0 -> 1) per candidate, evaluated at the stations s. Returns (M, N).

    A plain sigmoid is never exactly 0, so the plan would not start exactly at the car. We
    rescale it so that it is 0 at s0 and still goes to 1:

        S(s) = (sigma(a (s - c)) - sigma(a (s0 - c))) / (1 - sigma(a (s0 - c)))

    Special cases: a centre at +inf means "no step" (S = 0), and a step that is already over at
    the car gives S = 1 everywhere.
    """
    s = np.asarray(s, dtype=float)[None, :]
    c = np.atleast_1d(np.asarray(centres, dtype=float))[:, None]
    a = np.atleast_1d(np.asarray(steepness, dtype=float))
    a = np.broadcast_to(a, (c.shape[0],))[:, None]     # one steepness per step

    with np.errstate(invalid="ignore"):                # inf - inf when the centre is infinite
        raw = expit(a * (s - c))
        at_car = expit(a * (s0 - c))

    already_done = (1.0 - at_car) < 1e-9
    denominator = np.where(already_done, 1.0, 1.0 - at_car)
    steps = np.clip((raw - at_car) / denominator, 0.0, 1.0)
    steps = np.where(already_done, 1.0, steps)

    no_step = np.isinf(c) & (c > 0)
    return np.where(no_step, 0.0, steps)


# ---------------------------------------------------------------------------
# From offsets to trajectories
# ---------------------------------------------------------------------------

def rows_geometry(x, y):
    """
    Arc length and curvature of several paths at once (one path per row).
    Same formulas as track.arc_length / track.curvature, just on (M, N) arrays.
    """
    dx = np.diff(x, axis=-1)
    dy = np.diff(y, axis=-1)
    seg = np.maximum(np.hypot(dx, dy), 1e-9)                 # length of each segment

    first = np.zeros(x.shape[:-1] + (1,))
    s = np.concatenate((first, np.cumsum(seg, axis=-1)), axis=-1)

    # curvature through 3 points: kappa = 4 * area / (a * b * c)
    chord = np.hypot(x[..., 2:] - x[..., :-2], y[..., 2:] - y[..., :-2])
    chord = np.maximum(chord, 1e-9)
    twice_area = dx[..., :-1] * dy[..., 1:] - dy[..., :-1] * dx[..., 1:]   # signed: > 0 to the left
    kappa = np.zeros_like(x)
    kappa[..., 1:-1] = 2.0 * twice_area / (seg[..., :-1] * seg[..., 1:] * chord)
    kappa[..., 0] = kappa[..., 1]          # the end points just copy their neighbour
    kappa[..., -1] = kappa[..., -2]
    return s, kappa


@dataclass
class Candidates:
    """
    A bunch of candidate plans (M of them), all on the same N track stations.
    Every array is (M, N) except `stations` (N,).
    """
    stations: np.ndarray   # track stations [m]
    offset: np.ndarray     # offset from our racing line [m], > 0 = left
    d: np.ndarray          # lateral position on the track [m]
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray          # distance along each path [m]
    kappa: np.ndarray      # curvature of each path [1/m]
    v: np.ndarray          # speed profile [m/s]
    t: np.ndarray          # time at each station [s]

    def __len__(self):
        return self.offset.shape[0]

    @property
    def cost(self):
        """ The cost is simply the time to reach the end of the plan. """
        return self.t[:, -1] - self.t[:, 0]

    def trajectory(self, m):
        """ Candidate m as a Trajectory (its s are the track stations, like the racing line). """
        x, y = self.x[m], self.y[m]
        return Trajectory(t=self.t[m], x=x, y=y, psi=heading(x, y), v=self.v[m], s=self.stations.copy())

    def within_limits(self, d_min, d_max, tol=0.02):
        """ Which candidates stay on the track the whole way. """
        inside = (self.d >= d_min - tol) & (self.d <= d_max + tol)
        return np.all(inside, axis=1)


def offsets_to_candidates(track, plan, stations, offsets, params, v_start, t0, v_limit=None):
    """
    Turn offsets into full candidates:
    offset -> lateral position d -> (x, y) -> length and curvature -> speed -> times.
    """
    offsets = np.atleast_2d(np.asarray(offsets, dtype=float))
    d = plan.d_at(stations)[None, :] + offsets
    x, y = track.from_frenet(stations, d)
    s, kappa = rows_geometry(x, y)
    v = three_pass(s, kappa, params, v_start, v_limit)
    t = time_stamps(s, v, t0)
    return Candidates(stations=np.asarray(stations, dtype=float), offset=offsets, d=d,
                      x=x, y=y, s=s, kappa=kappa, v=v, t=t)


# ---------------------------------------------------------------------------
# The other cars
# ---------------------------------------------------------------------------

@dataclass
class PredictedCar:
    """
    Another car, as we think it will move: its predicted trajectory, plus the same thing in
    track coordinates (s, d) so we can compare it with our stations.
    """
    name: str
    params: object
    trajectory: Trajectory
    s: np.ndarray
    d: np.ndarray
    s_now: float
    d_now: float
    v_now: float

    def __post_init__(self):
        # To go from a station back to a time we need s to be strictly increasing.
        self._s_mono = np.maximum.accumulate(self.s) + 1e-9 * np.arange(len(self.s))

    def station_at(self, t):
        """ Where it is at time t. After the end of the prediction it keeps its last speed. """
        t = np.asarray(t, dtype=float)
        t_pred = self.trajectory.t
        s = np.interp(t, t_pred, self.s)
        s_after = self.s[-1] + self.trajectory.v[-1] * (t - t_pred[-1])
        return np.where(t > t_pred[-1], s_after, s)

    def lateral_at(self, t):
        return np.interp(t, self.trajectory.t, self.d)

    def speed_at(self, t):
        return np.interp(t, self.trajectory.t, self.trajectory.v)

    def speed_at_station(self, s):
        """ How fast it goes when it passes station s. """
        return np.interp(s, self._s_mono, self.trajectory.v)

    def time_at_station(self, s):
        """ When it reaches station s (extrapolated at its last speed). """
        s = np.asarray(s, dtype=float)
        t_pred = self.trajectory.t
        t = np.interp(s, self._s_mono, t_pred)
        v_last = max(float(self.trajectory.v[-1]), 0.1)
        t_after = t_pred[-1] + (s - self._s_mono[-1]) / v_last
        return np.where(s > self._s_mono[-1], t_after, t)


def predicted_cars(track, obs, predictions):
    """
    Put the prediction of every other car in track coordinates.

    In the hairpin a point can be close to both straights, so we give to_frenet a hint: where
    the car is now, plus how far it travels along its prediction. We also cut the prediction
    where the car reaches the end of the track (the exact plan just stops there).
    """
    cars = []
    for other in obs.others:
        traj = predictions.get(other.name)
        if traj is None or len(traj) < 2:
            continue

        hint = other.s + (traj.s - traj.s[0])
        s, d = track.to_frenet(traj.x, traj.y, s_hint=hint)
        s, d = np.atleast_1d(s), np.atleast_1d(d)

        at_the_end = np.flatnonzero(s >= track.length - 0.5)
        n = int(at_the_end[0]) + 1 if len(at_the_end) else len(s)
        n = max(n, 2)

        cut = Trajectory(t=traj.t[:n], x=traj.x[:n], y=traj.y[:n], psi=traj.psi[:n], v=traj.v[:n])
        cars.append(PredictedCar(other.name, other.params, cut, s[:n], d[:n],
                                 float(other.s), float(other.d), float(other.state.v)))
    return cars


def clearances(c, params, cars):
    """
    For each candidate, the smallest distance to any predicted car [m] (< 0 means a crash).

    Same idea as sim.trajectory_clearance: we compare both cars at the same moments (the time
    steps of the prediction), with the 3-circle footprints. inf if the times never overlap.
    """
    out = np.full(len(c), np.inf)
    if not cars:
        return out

    # heading of our candidates, needed for the footprints
    psi = np.unwrap(np.arctan2(np.gradient(c.y, axis=-1), np.gradient(c.x, axis=-1)), axis=-1)

    for car in cars:
        pred = car.trajectory
        for m in range(len(c)):
            # only the moments covered by both the candidate and the prediction
            common = (pred.t >= c.t[m, 0]) & (pred.t <= c.t[m, -1])
            if not np.any(common):
                continue
            tt = pred.t[common]
            x_me = np.interp(tt, c.t[m], c.x[m])
            y_me = np.interp(tt, c.t[m], c.y[m])
            psi_me = np.interp(tt, c.t[m], psi[m])
            dist = footprint_distance(x_me, y_me, psi_me, params,
                                      pred.x[common], pred.y[common], pred.psi[common], car.params)
            out[m] = min(out[m], float(np.min(dist)))
    return out


def _gaps(c, params, car, t_hw, d_offset):
    """
    Along each candidate: gap between our rear and the other car's front, whether we overlap
    sideways, and the gap the re-entry rule asks for.
    """
    s_other = car.station_at(c.t)
    gap = c.stations[None, :] - s_other - 0.5 * (params.length + car.params.length)
    overlap = np.abs(c.d - car.lateral_at(c.t)) < 0.5 * (params.width + car.params.width)
    required = t_hw * car.speed_at(c.t) + d_offset
    return gap, overlap, required


def reentry_ok(c, params, cars, t_hw=REENTRY_T_HW, d_offset=REENTRY_OFFSET):
    """
    Which candidates never cut in too close in front of another car.

    The rule is the one of sim.outcome: whenever we are in front and in the same lane, we need
    at least t_hw * v_other + d_offset of gap. If we are already in front in its lane when the
    plan starts, there is nothing left to decide, so that car is skipped.
    """
    ok = np.ones(len(c), dtype=bool)
    for car in cars:
        gap, overlap, required = _gaps(c, params, car, t_hw, d_offset)
        already_in_front = gap[0, 0] > 0 and overlap[0, 0]
        if already_in_front:
            continue
        too_close = (gap > 0) & overlap & (gap < required)
        ok &= ~np.any(too_close, axis=1)
    return ok


def earliest_reentry(c, params, cars, t_hw=REENTRY_T_HW, d_offset=REENTRY_OFFSET):
    """
    For each "out" path (moved out and staying out), the first station where we are far enough
    ahead of every car to come back. Returns an index per candidate, or -1 if it never happens.

    We only look for the first good station: after that the car comes back, so what the out
    path does later (often slow, or off the track in the corner) does not matter. The full path
    with its return is checked afterwards with reentry_ok.
    """
    k_star = np.zeros(len(c), dtype=int)
    for car in cars:
        gap, _, required = _gaps(c, params, car, t_hw, d_offset)
        far_enough = gap >= required
        first = np.where(far_enough.any(axis=1), np.argmax(far_enough, axis=1), -1)
        # we need to be clear of *every* car: take the latest index, and -1 wins
        never = (k_star < 0) | (first < 0)
        k_star = np.where(never, -1, np.maximum(k_star, first))
    return k_star


# ---------------------------------------------------------------------------
# ACC
# ---------------------------------------------------------------------------

def acc_cap(me, others, t_hw=REENTRY_T_HW, d_offset=REENTRY_OFFSET):
    """
    ACC speed behind the closest car ahead in our lane (slides 6a), or None if there is none.

    This cap stays on during an overtaking attempt, so if the attempt goes wrong the car
    ends up following instead of crashing.
    """
    caps = []
    for other in others:
        if other.s <= me.s:
            continue                                         # behind us
        same_lane = abs(other.d - me.d) <= 0.5 * (me.params.width + other.params.width) + 0.5
        if not same_lane:
            continue
        gap = other.s - me.s - 0.5 * (me.params.length + other.params.length)
        caps.append(acc_speed(gap, me.state.v, other.state.v, t_hw, d_offset))
    return min(caps) if caps else None


# ---------------------------------------------------------------------------
# What every local planner has in common
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """ What the planner decided at one step (kept for the logs and the plots). """
    t: float
    kind: str                        # "follow" or "overtake"
    width: float = 0.0               # where we move to, from the racing line [m] (> 0 left)
    c1: float = float("nan")         # centre of the lane change [m]
    c2: float = float("nan")         # centre of the return [m] (inf = no return planned)
    cost: float = float("nan")       # time of the chosen plan [s]
    follow_cost: float = float("nan")   # time if we just followed [s]
    n_candidates: int = 1
    n_feasible: int = 1


@dataclass
class Frame:
    """ Everything about "now" that all candidates of one step share. """
    t0: float
    s0: float
    v0: float
    o0: float                # how far we currently are from our racing line [m]
    stations: np.ndarray     # stations of the plan [m]
    v_limit: np.ndarray      # speed cap: only the last station, at the racing-line speed


class LocalPlanner:
    """
    Base class of the local planners. A planner is attached to a car with setup(), then at each
    step plan(obs, predictions) returns the plan to follow as a Trajectory. What it chose ends up
    in self.decision.

    Settings:
        ds: spacing of the plan stations [m]
        passing_time, max_passing_time: time we allow for a pass, used for the horizon [s]
        horizon_margin: extra horizon [m]
        track_margin: distance kept from the track edge [m]
        d_safe: minimum distance to another car [m]
        t_hw, d_offset: ACC and re-entry spacing (t_hw * v + d_offset)
        curvature_share: how much of the grip a lane change may use
        offset_deadband: below this distance from the racing line we just plan from the line
            itself [m]. Small tracking errors are the controller's job, and a steep sigmoid to
            fix a few centimetres would eat grip the car needs in the corner.
    """

    name = "local"

    def __init__(self, ds=1.0, passing_time=6.0, horizon_margin=20.0, max_passing_time=8.0,
                 track_margin=0.3, d_safe=D_SAFE, t_hw=REENTRY_T_HW, d_offset=REENTRY_OFFSET,
                 curvature_share=1.0, offset_deadband=0.75):
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
        self.track = None
        self.decision = None

    def options(self):
        """ The settings above, to create a sub-planner that behaves the same way. """
        return dict(ds=self.ds, passing_time=self.passing_time, horizon_margin=self.horizon_margin,
                    max_passing_time=self.max_passing_time, track_margin=self.track_margin,
                    d_safe=self.d_safe, t_hw=self.t_hw, d_offset=self.d_offset,
                    curvature_share=self.curvature_share, offset_deadband=self.offset_deadband)

    def setup(self, track, params, global_plan):
        if global_plan is None:
            raise ValueError("A local planner needs the racing line of its car.")
        self.track = track
        self.params = params
        self.global_plan = global_plan
        self.d_min, self.d_max = track.d_limits(params.width, self.track_margin)
        self.reset()

    def reset(self):
        self.decision = None

    def plan(self, obs, predictions):
        raise NotImplementedError     # each planner writes its own

    # ---- tools for the planners ----

    def frame(self, obs):
        """ Where we are now, and which stations the plan will cover. """
        me = obs.me
        v0 = max(float(me.state.v), 0.5)

        # passing time: our default, or the Approach 0 estimate if we are catching someone
        tau = passing_time_estimate(me, obs.others, self.t_hw, self.d_offset)
        if tau is None:
            passing = self.passing_time
        else:
            passing = float(np.clip(tau, self.passing_time, self.max_passing_time))

        dist = passing_distance(self.global_plan, float(me.s), passing)
        horizon = planning_horizon(v0, self.params, dist, self.horizon_margin)
        stations = stations_ahead(self.track, float(me.s), horizon, self.ds)

        o0 = float(me.d) - float(self.global_plan.d_at(me.s))
        if abs(o0) < self.offset_deadband:
            o0 = 0.0          # close enough to the line: plan from the line

        # at the end of the plan we must be back to the racing-line speed
        v_limit = np.full(len(stations), np.inf)
        v_limit[-1] = race_speed(self.global_plan, stations[-1])
        return Frame(float(obs.t), float(me.s), v0, o0, stations, v_limit)

    def evaluate(self, f, offsets, v_limit=None):
        """ offsets -> Candidates, with the end-of-plan cap (and an extra cap if given). """
        cap = f.v_limit if v_limit is None else np.minimum(f.v_limit, v_limit)
        return offsets_to_candidates(self.track, self.global_plan, f.stations, offsets,
                                     self.params, f.v0, f.t0, cap)

    def relevant_cars(self, obs, predictions, f):
        """
        The cars we need to care about: ahead within the horizon, next to us, or just behind
        (closer than the re-entry spacing - we must not cut in front of them either).
        """
        cars = []
        for car in predicted_cars(self.track, obs, predictions):
            behind_zone = (0.5 * (self.params.length + car.params.length)
                           + self.t_hw * car.v_now + self.d_offset + 5.0)
            if f.s0 - behind_zone <= car.s_now <= f.stations[-1]:
                cars.append(car)
        return cars

    def checked_cars(self, cars, f):
        """
        Cars to check for collisions: everyone, except cars fully behind us in our lane. Those
        are following us and keep their own distance with their ACC.
        """
        my_d = float(self.global_plan.d_at(f.s0)) + f.o0
        out = []
        for car in cars:
            behind = car.s_now < f.s0 - 0.5 * (self.params.length + car.params.length)
            same_lane = abs(car.d_now - my_d) < 0.5 * (self.params.width + car.params.width)
            if not (behind and same_lane):
                out.append(car)
        return out

    def feasible(self, c, cars, f, check_start=True, v_cap=None):
        """
        Which candidates we are allowed to drive: on the track, no collision, and a safe
        re-entry ("within limits and collision free" in slides 6a).

        With check_start we also refuse a plan that would need to brake hard right away: its
        first speed must not be below our current speed (or below the ACC cap if there is one,
        since braking behind a leader is the ACC's job, not the plan's).
        """
        ok = c.within_limits(self.d_min, self.d_max)

        if check_start:
            allowed = np.full(len(c), f.v0)
            if v_cap is not None:
                cap_at_start = np.broadcast_to(v_cap, c.v.shape)[:, 0]
                allowed = np.minimum(allowed, cap_at_start)
            ok &= c.v[:, 0] >= allowed - 0.5

        checked = self.checked_cars(cars, f)
        ok &= clearances(c, self.params, checked) >= self.d_safe
        ok &= reentry_ok(c, self.params, checked, self.t_hw, self.d_offset)
        return ok
