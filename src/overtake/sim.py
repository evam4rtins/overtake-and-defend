"""
Simulation of the cars on the track: collisions checks, simulation loop and 
success criterion
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
import numpy as np
from .track import CornerTrack
from .types import VehicleState, VehicleParams, Trajectory, Observation, CarView

# ------------------------------ Collision Checks ------------------------------

def _circles(x, y, psi, params: VehicleParams):
    """ 
    Three circles that cover the car's bounding box, used for collision checking 
    
    Returns:
        cx, cy: 3xN arrays of circle centers
        radius: radius of the circles
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    psi = np.atleast_1d(psi)
    offsets = np.array([-1.0, 0.0, 1.0]) * params.length / 3.0
    cx = x[:, None] + offsets * np.cos(psi)[:, None]
    cy = y[:, None] + offsets * np.sin(psi)[:, None]
    radius = np.hypot(params.length / 6.0, params.width / 2.0)
    return np.stack((cx, cy), axis=-1), radius

def footprint_distance(xa, ya, psia, params_a: VehicleParams, 
                       xb, yb, psib, params_b: VehicleParams):
    """
    Distance between two vehicles' footprints (negative if overlapping)

    Returns:
        float or np.ndarray: distance between the two vehicles' footprints
    """
    ca, ra = _circles(xa, ya, psia, params_a)
    cb, rb = _circles(xb, yb, psib, params_b)
    diff = ca[:, :, None, :] - cb[:, None, :, :]
    clearance = np.linalg.norm(diff, axis=-1).min(axis=(1, 2)) - ra - rb
    return float(clearance[0]) if np.ndim(xa) == 0 else clearance

def trajectory_clearance(traj_a: Trajectory, params_a: VehicleParams,
                         traj_b: Trajectory, params_b: VehicleParams, dt: float = 0.05) -> float:
    """
    Smallest clearance between two cars following two trajectories, compared at equal times

    Returns:
        float: minimum clearance over the common time window [m] (< 0 = collision);
               inf if the two trajectories do not overlap in time.
    """
    t0, t1 = max(traj_a.t[0], traj_b.t[0]), min(traj_a.t[-1], traj_b.t[-1])
    if t1 < t0:
        return float("inf")
    t = np.append(np.arange(t0, t1, dt), t1)
    xa, ya, pa, _ = traj_a.at_time(t)
    xb, yb, pb, _ = traj_b.at_time(t)
    return float(np.min(footprint_distance(xa, ya, pa, params_a, xb, yb, pb, params_b)))

#------------------------------------- Car -------------------------------------

def start_state(track: CornerTrack, s: float, d: float, v:float) -> VehicleState:
    """
    Create a vehicle state from Frenet coordinates (s, d) and speed v
    """
    x, y = track.from_frenet(s, d)
    _, _, psi, _ = track.at_s(s)
    return VehicleState(x=float(x), y=float(y), psi=float(psi), v=float(v))

@dataclass
class Car:
    """
    One car in the simulation
    
    Attributes:
        name:          Unique name.
        params:        VehicleParams.
        model:         PointMass or KinematicBicycle.
        agent:         object with act(obs) -> action, or None (driven from outside).
        initial_state: VehicleState at t = 0.
        global_plan:   optional racing line of this car.
        state, s, d, finish_time, log: runtime values.
    """

    name: str
    params: VehicleParams
    model: Any
    agent: Any
    initial_state: VehicleState
    global_plan: Any = None
    state: Optional[VehicleState] = None
    s: float = 0.0
    d: float = 0.0
    finish_time: Optional[float] = None
    log: Dict[str, list] = field(default_factory=dict)

    def view(self) -> CarView:
        plan = getattr(self.agent, "plan", None) if self.agent is not None else None
        return CarView(self.name, self.state, self.s, self.d, self.params, plan)


LOG_KEYS = ["t", "x", "y", "psi", "v", "s", "d", "a_x", "a_y"]

@dataclass
class EpisodeResult:
    """
    Result of a simulation episode

    Attributes:
        cars:          dict name -> dict of np.ndarray time series.
        finish_times:  dict name -> time the car crossed the finish line [s].
        collision:     bool. True if two cars touched.
        off_track:     list of names of the cars that left the track.
        min_clearance: float. Smallest clearance between two cars over the run [m].
        t_end:         float. Time at which the run stopped [s].
        params:        dict name -> VehicleParams.
    """
    cars: Dict[str, Dict[str, np.ndarray]]
    finish_times: Dict[str, Optional[float]]
    collision: bool
    off_track: List[str]
    min_clearance: float
    t_end: float
    params: Dict[str, VehicleParams]

# ------------------------------ Simulation Loop ------------------------------

class Simulator:
    """
    Simulation loop for any number of cars on one track.

    Args:
        track:               CornerTrack.
        cars:                list of Car.
        dt:                  float. Time step [s].
        t_max:               float. Maximum duration of a run [s].
        finish_margin:       float. The finish line is this far before the end of the track [m].
        off_track_tolerance: float. How far the side of a car may cross the track edge before the
                             car counts as off track [m].
        stop_on_collision:   bool. End the run at the first collision.
    """

    def __init__(self, track: CornerTrack, cars: List[Car], dt: float = 0.05, 
                 t_max: float = 40.0, finish_margin: float = 5.0, 
                 off_track_tolerance: float = 0.5, stop_on_collision: bool = True):
        names = [car.name for car in cars]
        if len(set(names)) != len(names):
            raise ValueError("Car names must be unique")
        self.track = track
        self.cars = cars
        self.dt = dt
        self.t_max = t_max
        self.s_finish = track.length - finish_margin
        self.off_track_tolerance = off_track_tolerance
        self.stop_on_collision = stop_on_collision
        for car in cars:
            if car.agent is not None and hasattr(car.agent, "setup"):
                car.agent.setup(track, car)
        self.reset()

    def car(self, name: str) -> Car:
        return next(car for car in self.cars if car.name == name)
    
    def reset(self) -> None:
        """ Reset the simulation to its initial state. """
        self.t = 0.0
        self.collision = False
        self.off_track = []
        self.min_clearance = np.inf
        for car in self.cars:
            car.state = car.initial_state
            car.s, car.d = self.track.to_frenet(car.state.x, car.state.y)
            car.finish_time = None
            car.log = {key: [] for key in LOG_KEYS}
            if car.agent is not None and hasattr(car.agent, "reset"):
                car.agent.reset()
        self._check_collisions()
        self._log()
    
    def observation(self, car: Car) -> Observation:
        return Observation(
            t=self.t, 
            track=self.track, 
            me=car.view(),
            others=[c.view() for c in self.cars if c is not car]
            )

    def step(self, external_action: Optional[Dict[str, Any]] = None) -> bool:
        actions = {}
        for car in self.cars:
            if car.agent is None:
                actions[car.name] = external_action[car.name]
            else:
                actions[car.name] = car.agent.act(self.observation(car))

        for car in self.cars:
            car.state = car.model.step(car.state, np.asarray(actions[car.name], dtype=float), self.dt)
            car.s, car.d = self.track.to_frenet(car.state.x, car.state.y, s_hint=car.s)
        self.t += self.dt
        self._check_finish_and_limits()
        self._check_collisions()
        self._log()
        return self.done
    
    def run(self) -> EpisodeResult:
        """ Run the simulation until completion. """
        self.reset()
        while not self.done:
            self.step()
        return self.result()
    
    @property
    def done(self) -> bool:
        if self.collision and self.stop_on_collision:
            return True
        if all(car.finish_time is not None for car in self.cars):
            return True
        return self.t >= self.t_max - 1e-9
    
    def result(self) -> EpisodeResult:
        return EpisodeResult(
            cars={c.name: {k: np.asarray(v) for k, v in c.log.items()} for c in self.cars},
            finish_times={c.name: c.finish_time for c in self.cars},
            collision=self.collision, 
            off_track=list(self.off_track),
            min_clearance=self.min_clearance, 
            t_end=self.t,
            params={c.name: c.params for c in self.cars}
            )
    
    def _check_finish_and_limits(self) -> None:
        for car in self.cars:
            if car.finish_time is None and car.s >= self.s_finish:
                car.finish_time = self.t
            half = self.track.width / 2 - car.params.width / 2
            if abs(car.d) > half + self.off_track_tolerance and car.name not in self.off_track:
                self.off_track.append(car.name)

    def _check_collisions(self) -> None:
        for i, a in enumerate(self.cars):
            for b in self.cars[i + 1:]:
                gap = footprint_distance(a.state.x, a.state.y, a.state.psi, a.params,
                                         b.state.x, b.state.y, b.state.psi, b.params)
                self.min_clearance = min(self.min_clearance, gap)
                if gap < 0:
                    self.collision = True

    def _log(self) -> None:
        for car in self.cars:
            st = car.state
            for key, value in (("t", self.t), ("x", st.x), ("y", st.y), ("psi", st.psi), ("v", st.v),
                               ("s", car.s), ("d", car.d), ("a_x", st.a_x), ("a_y", st.a_y)):
                car.log[key].append(value)

# ------------------------------ Success Criterion ------------------------------

@dataclass
class Outcome:
    """
    Result of an overtaking attempt.
    
    Attributes:
        success:       bool. Ego ahead at the finish by tau_min, no collision, not off track, safe re-entry.
        margin:        float or None. t_opponent - t_ego at the finish line [s] (> 0 = ego ahead).
        overtook:      bool. The ego was ahead of the opponent at some point.
        safe_reentry:  bool. Whenever the ego was ahead and in the opponent's lane, the gap was large enough.
        collision, off_track: bool.
        min_clearance: float. Smallest clearance between the two cars [m].
        t_ego, t_opponent: finish times [s] or None.
    """

    success: bool
    margin: Optional[float]
    overtook: bool
    safe_reentry: bool
    collision: bool
    off_track: bool
    min_clearance: float
    t_ego: Optional[float]
    t_opponent: Optional[float]

def outcome(result: EpisodeResult, ego: str = "ego", opponent: str = "opponent", tau_min: float = 0.1,
            reentry_t_hw: float = 0.3, reentry_offset: float = 2.0) -> Outcome:
    """
    Evaluate one run with the success criterion.

    Safe re-entry: once the ego is ahead and laterally overlaps the opponent, the gap
    between the ego's rear and the opponent's front must be at least
    reentry_t_hw * v_opponent + reentry_offset; otherwise the ego cut in too early.

    Args:
        result:   EpisodeResult.
        ego, opponent: str. Names of the two cars.
        tau_min:  float. Minimum time margin at the finish line [s].
        reentry_t_hw, reentry_offset: float. Re-entry spacing policy [s], [m].

    Returns:
        Outcome.
    """
    e, o = result.cars[ego], result.cars[opponent]
    pe, po = result.params[ego], result.params[opponent]
    t_e, t_o = result.finish_times.get(ego), result.finish_times.get(opponent)
    margin = (t_o - t_e) if (t_e is not None and t_o is not None) else None

    gap = e["s"] - o["s"] - 0.5 * (pe.length + po.length)        # ego rear -> opponent front [m]
    ahead = gap > 0
    same_lane = np.abs(e["d"] - o["d"]) < 0.5 * (pe.width + po.width)
    required = reentry_t_hw * o["v"] + reentry_offset
    safe_reentry = bool(np.all(gap[ahead & same_lane] >= required[ahead & same_lane]))

    off = ego in result.off_track
    finished_first = t_e is not None and (t_o is None or margin >= tau_min)
    success = (not result.collision) and (not off) and finished_first and safe_reentry
    return Outcome(success=success, margin=margin, overtook=bool(np.any(ahead)), safe_reentry=safe_reentry,
                   collision=result.collision, off_track=off, min_clearance=result.min_clearance,
                   t_ego=t_e, t_opponent=t_o)