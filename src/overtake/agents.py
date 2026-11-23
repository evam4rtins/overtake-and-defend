from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from .control import PurePursuit, acc_speed
from .planners.base import REENTRY_OFFSET, REENTRY_T_HW, Decision, LocalPlanner, acc_cap
from .prediction import ExactPlan
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
        if car.global_plan is None:
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


class PlannerAgent:
    """
    Drives a car with a local planner: predictor -> planner -> controller.

    Every replan_period seconds the other cars are predicted (prediction.py) and the planner computes a
    new plan; in between, the controller (control.py) tracks the last plan. The reference speed of the
    controller is capped by the ACC behind a car in the same lane, so an aborted attempt degrades
    into following instead of a collision (proposal 5.4). The plan is published as self.plan, so
    that the other cars can predict this one with ExactPlan.

    Minimal version used to run and test the planners; the scenario builder is built around it.

    Args:
        planner: LocalPlanner (FollowPlanner, SigmoidPlanner, ...).
        predictor: object with predict(car, t) -> Trajectory; default ExactPlan over 10 s (the
                   length of the longest plans).
        controller: PurePursuit or None.
        replan_period: float. Time between two planning steps [s] (tasks: 0.1 to 0.2 s).
        use_acc: bool. Cap the reference speed with the ACC.
        t_hw, d_offset: float. ACC spacing [s], [m]; by default the re-entry spacing of sim.outcome.
    """

    def __init__(self, planner: LocalPlanner, predictor=None, controller: Optional[PurePursuit] = None,
                 replan_period: float = 0.2, use_acc: bool = True, t_hw: float = REENTRY_T_HW,
                 d_offset: float = REENTRY_OFFSET):
        self.planner = planner
        self.predictor = predictor or ExactPlan(horizon=10.0)
        self.controller = controller or PurePursuit()
        self.replan_period = replan_period
        self.use_acc = use_acc
        self.t_hw = t_hw
        self.d_offset = d_offset
        self.plan: Optional[Trajectory] = None
        self.decisions: List[Tuple[float, Decision]] = []

    def setup(self, track, car) -> None:
        self.params = car.params
        self.model = car.model
        self.planner.setup(track, car.params, car.global_plan)

    def reset(self) -> None:
        self.plan = None
        self._t_plan = -np.inf
        self.decisions = []
        self.planner.reset()

    def act(self, obs: Observation) -> np.ndarray:
        if self.plan is None or obs.t - self._t_plan >= self.replan_period - 1e-9:
            predictions = {other.name: self.predictor.predict(other, obs.t) for other in obs.others}
            self.plan = self.planner.plan(obs, predictions)
            self._t_plan = obs.t
            if self.planner.decision is not None:
                self.decisions.append((obs.t, self.planner.decision))
        v_cap = acc_cap(obs.me, obs.others, self.t_hw, self.d_offset) if self.use_acc else None
        return self.controller.control(obs.me.state, self.plan, self.params, self.model.action_names, v_cap)
