"""
Follow planner: stay on the racing line and keep a safe distance behind the car ahead (ACC).

It drives the Phase 1 opponent, and it is the "stay behind" option that the other planners
always keep. If the car is off its line (for example after an aborted pass), it steers back
smoothly.
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from ..types import Observation, Trajectory
from .base import (Candidates, Decision, Frame, LocalPlanner, PredictedCar, max_steepness,
                   race_speed, sigmoid_steps)


class FollowPlanner(LocalPlanner):
    """
    Racing line + ACC.

    Args:
        return_delays: sequence of float. Delays [m] tried for the return to the racing line when
                       the car is off its line.
        cap_factors: sequence of float. Factors applied to the speed cap behind the leader.
        **kwargs: options of LocalPlanner.

    Among the combinations (delay, factor), the fastest feasible plan is used; if none is
    feasible, the most cautious one (longest delay, smallest factor).
    """

    name = "follow"

    def __init__(self, return_delays: Sequence[float] = (0.0, 10.0, 20.0, 40.0, 80.0),
                 cap_factors: Sequence[float] = (1.0, 0.95, 0.9, 0.8, 0.65), **kwargs):
        super().__init__(**kwargs)
        self.return_delays = tuple(float(x) for x in return_delays)
        self.cap_factors = tuple(float(x) for x in cap_factors)

    def plan(self, obs: Observation, predictions: Dict[str, Trajectory]) -> Trajectory:
        f = self.frame(obs)
        cars = self.relevant_cars(obs, predictions, f)
        c, m, ok = self.candidate(f, cars)
        self.decision = Decision(t=f.t0, kind="follow", cost=float(c.cost[m]), follow_cost=float(c.cost[m]),
                                 n_candidates=len(c), n_feasible=int(ok.sum()))
        return c.trajectory(m)

    def candidate(self, f: Frame, cars: Sequence[PredictedCar]) -> Tuple[Candidates, int, np.ndarray]:
        """
        The follow plans of this step (one per return delay and cap factor) and the one to use.

        Returns:
            c: Candidates. The follow plans.
            m: int. Index of the plan to use: the fastest feasible one, else the most cautious one.
            ok: np.ndarray (M,) bool. Feasibility of each plan.
        """
        delays = self.return_delays if abs(f.o0) > 0.05 else (0.0,)
        cap = self.leader_cap(f, cars)
        factors = self.cap_factors if cap is not None else (1.0,)
        a = max_steepness(f.o0, f.v0, self.params, self.curvature_share)
        centres = f.s0 + 3.0 / a + np.repeat(np.asarray(delays), len(factors))
        offsets = f.o0 * (1.0 - sigmoid_steps(f.stations, centres, a, f.s0))
        if cap is None:
            c = self.evaluate(f, offsets)
        else:
            c = self.evaluate(f, offsets, np.tile(np.asarray(factors), len(delays))[:, None] * cap[None, :])
        ok = self.feasible(c, cars, f, check_start=False)
        if np.any(ok):
            m = int(np.flatnonzero(ok)[np.argmin(c.cost[ok])])
        else:
            m = len(c) - 1
        return c, m, ok

    def leader(self, f: Frame, cars: Sequence[PredictedCar]) -> Optional[PredictedCar]:
        """
        Closest car ahead that is in the current lane of this car or that comes onto its racing
        line at some point of its prediction. The second case matters when the two racing lines
        differ (different cars) and only merge at the corner.
        """
        best = None
        for car in cars:
            if car.s_now <= f.s0:
                continue
            width = 0.5 * (self.params.width + car.params.width) + 0.5
            in_lane = abs(car.d_now - (float(self.global_plan.d_at(f.s0)) + f.o0)) < width
            on_line = np.any(np.abs(car.d - self.global_plan.d_at(car.s)) < width)
            if (on_line or in_lane) and (best is None or car.s_now < best.s_now):
                best = car
        return best

    def leader_cap(self, f: Frame, cars: Sequence[PredictedCar]) -> Optional[np.ndarray]:
        """
        Speed cap that keeps the car behind its leader, at each station [m/s], or None.

        The car copies the predicted speed the leader has a distance Delta further along the
        track: with the same speed profile shifted by Delta, the distance between the two cars
        stays Delta. Delta is the ACC spacing (centre to centre) or the current distance if it is
        smaller; the ACC of the controller then opens the gap progressively.
        """
        lead = self.leader(f, cars)
        if lead is None:
            return None
        half_lengths = 0.5 * (self.params.length + lead.params.length)
        desired = half_lengths + self.t_hw * race_speed(self.global_plan, f.stations) + self.d_offset
        delta = np.minimum(lead.s_now - f.s0, desired)
        return lead.speed_at_station(f.stations + delta)
