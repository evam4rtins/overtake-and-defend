""" 

Shared data types, used by every module of the package 

Conventions:
- SI units: meters, seconds, radians, m/s, m/s^2
- Global frame: x, y in meters; heading psi measured counter-clockwise from the x-axis
- Frenet frame: s = arc length along the track centreline, d = lateral offset, 
d > 0 to the left of the centreline.

"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional
import numpy as np

# ------------------------------------ Car ---------------------------------------
@dataclass(frozen=True)
class VehicleParams:
    """ Physical parameters of one car. """
    a_y_max: float = 15.0     # lateral grip [m/s^2]
    a_x_max: float = 15.0     # braking grip [m/s^2]
    a_engine : float = 8.0    # engine acceleration at low speed [m/s^2]
    v_max: float = 75.0       # top speed [m/s]
    length: float = 4.7       # car length [m]
    width: float = 1.9        # car width [m]
    l_f: float = 1.5          # distance from CG to front axle [m]
    l_r: float = 1.5          # distance from CG to rear axle [m]
    delta_max: float = 0.35   # max steering angle [rad]

    @property
    def wheelbase(self) -> float:
        """ Wheelbase of the car [m]. """
        return self.l_f + self.l_r
    
    def scaled(self, **ratios: float) -> VehicleParams:
        """ Return a copy of the vehicle parameters, scaled by the given ratios. """
        unknown = set(ratios) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown VehicleParams fields: {unknown}")
        return replace(self, **{k: getattr(self, k) * v for k, v in ratios.items()})
    

@dataclass(frozen=True)
class VehicleState:
    """ State of one car at one instant in time. """
    x: float                # global x position [m]
    y: float                # global y position [m]
    psi: float              # heading [rad]
    v: float                # speed [m/s]
    omega: float = 0.0      # yaw rate [rad/s]
    a_x: float = 0.0        # longitudinal acceleration [m/s^2]
    a_y: float = 0.0        # lateral acceleration [m/s^2]


# ------------------------------------ Trajectory ---------------------------------------
@dataclass
class Trajectory:
    t : np.ndarray                         # time [s]
    x : np.ndarray                         # global x position [m]
    y : np.ndarray                         # global y position [m]
    psi: np.ndarray                        # heading [rad]
    v: np.ndarray                          # speed [m/s]
    s: Optional[np.ndarray] = None         # arc length along the track centreline [m]

    def __post_init__(self):
        self.t, self.x, self.y, self.psi, self.v = (
            np.asarray(a, dtype=float) for a in (self.t, self.x, self.y, self.psi, self.v))
        n = len(self.t)
        if not all(len(a) == n for a in (self.x, self.y, self.psi, self.v)):
            raise ValueError("All trajectory arrays must have the same length.")
        if n > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("Time array must be strictly increasing.")
        if self.s is None:
            ds = np.hypot(np.diff(self.x), np.diff(self.y))
            self.s = np.concatenate(([0], np.cumsum(ds)))
        else:
            self.s = np.asarray(self.s, dtype=float)
    
    def  __len__(self) -> int:
        return len(self.t)
    
    @property
    def duration(self) -> float:
        """ Duration of the trajectory [s]. """
        return self.t[-1] - self.t[0]
    
    def at_time(self, t) -> float:
        """ Return the state of the trajectory at time t [s], interpolating if necessary."""
        t = np.asarray(t, dtype=float)
        psi = np.unwrap(self.psi)
        return (np.interp(t, self.t, self.x),
                np.interp(t, self.t, self.y),
                np.interp(t, self.t, psi),
                np.interp(t, self.t, self.v))
    
    def shifted(self, dt: float) -> Trajectory:
        """ Return a copy of the trajectory, shifted in time by dt [s]. """
        return replace(self, t=self.t + dt)
    
    # ------------------------------------ Observations ---------------------------------------

@dataclass
class CarView:
    """ What is known about one car at one instant in time. (ideal perception) """ 
    name: str
    state: VehicleState
    s: float
    d: float
    params: VehicleParams
    plan: Optional[Trajectory] = None  # current plan of the car, if known

@dataclass
class Observation:
   """ What an agent receives about the environment at one instant in time."""
   t: float
   track: Any  # track obejct track.CornerTrack
   me: CarView
   others: List[CarView] = field(default_factory=list)

   def car_ahead(self) -> Optional[CarView]:
       """ Return the car ahead of me, if any. """
       ahead = [c for c in self.others if c.s > self.me.s]
       return min(ahead, key=lambda c: c.s, default=None)