from __future__ import annotations
from typing import Optional
import numpy as np
from .control import PurePursuit, acc_speed
from .types import Observation, Trajectory

class RacingLineAgent:
    """ 
    Follows the car's global plan with ACC behind a slower car. 
    
    Args:
        controller: PurePursuit or None.
        use_acc:    bool. Slow down behind a car in the same lane.
        t_hw, d_offset: float. ACC spacing policy [s], [m].
    """

    def __init__(self, controller: Optional[PurePursuit] = None, use_acc: bool = True, 
                 t_hw: float = 0.6, d_offset: float = 5.0):
        self.controller = controller or PurePursuit()
        self.use_acc = use_acc
        self.t_hw = t_hw
        self.d_offset = d_offset
        self.plan: Optional[Trajectory] = None

    def setup(self, track, car) -> None:
        if car.globan_plan is None:
            raise ValueError(f"RacingLineAgent needs a global plan for car '{car.name}'.")
        self.track = track
        self.params = car.params
        self.model = car.model
        self.line = car.global_plan.trajectory

    def reset(self) -> None:
        self.plan = None
    
    def act(self, obs: Observation) -> np.ndarray:
        me = obs.me
        # Current plan in absolute time: the racing line, timed so that it passes here now.
        t_line_here = float(np.interp(me.s, self.line.s, self.line.t))
        self.plan = self.line.shifted(obs.t - t_line_here)
        v_cap = self._acc_cap(obs) if self.use_acc else None
        return self.controller.control(me.state, self.line, self.params, self.model.action_names, v_cap)

    def _acc_cap(self, obs: Observation) -> Optional[float]:
        """ ACC speed for the closest car ahead in the same lane, or None. """
        me, caps = obs.me, []
        for other in obs.others:
            if other.s <= me.s or abs(other.d - me.d) > 0.5 * (me.params.width + other.params.width) + 0.5:
                continue
            gap = other.s - me.s - 0.5 * (me.params.length + other.params.length)
            caps.append(acc_speed(gap, me.state.v, other.state.v, self.t_hw, self.d_offset))
        return min(caps) if caps else None
                                       
