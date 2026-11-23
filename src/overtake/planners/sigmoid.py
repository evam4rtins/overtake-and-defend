"""
Sigmoid planner (Approach A of the proposal, slides 6a).

At each step the car tries a set of overtaking paths: move out to a side with a sigmoid,
pass, then come back to the racing line as soon as it is safely ahead. Paths that leave the
track or come too close to the other car are dropped, and the fastest remaining one is
compared with simply following. A small hysteresis keeps the car from switching back and
forth between options.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
from ..types import Observation, Trajectory
from .base import (Decision, Frame, LocalPlanner, earliest_reentry, max_steepness, race_speed,
                   sigmoid_steps)
from .follow import FollowPlanner


@dataclass
class _Spec:
    """ One overtaking manoeuvre before its return point is known. """
    width: float          # target offset w from the racing line [m]
    c1: float             # centre of the lane change [m]
    a1: float             # steepness of the lane change [1/m]
    in_progress: bool     # the manoeuvre chosen at the previous step


@dataclass
class _Option:
    """ A complete candidate (one row of a Candidates batch) and its evaluation. """
    kind: str
    row: int
    cost: float
    width: float = 0.0
    c1: float = float("nan")
    c2: float = float("nan")
    in_progress: bool = False


class SigmoidPlanner(LocalPlanner):
    """
    Approach A: sigmoid candidates around the racing line, the fastest feasible one wins.

    Args:
        widths: sequence of float. Sizes |w| of the lateral shift [m]; both signs are tried.
        c1_step: float. Smallest spacing of the lane-change positions c1 [m].
        n_c1: int. Number of lane-change positions tried; they cover the whole horizon.
        hysteresis: float. Time advantage needed to change manoeuvre [s].
        refine_step: float. How far c2 is moved when the full path fails its check [m].
        n_refine: int. How many times c2 may be moved.
        finish_margin: float. The finish line is this far before the end of the track [m]
                       (same as the simulator); a pass completed at the finish needs no return.
        **kwargs: options of LocalPlanner.
    """

    name = "sigmoid"

    def __init__(self, widths: Sequence[float] = (3.5, 5.0, 6.5, 8.0), c1_step: float = 15.0,
                 n_c1: int = 12, hysteresis: float = 0.05, refine_step: float = 10.0, n_refine: int = 2,
                 finish_margin: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.widths = tuple(float(w) for w in widths)
        self.c1_step = c1_step
        self.n_c1 = n_c1
        self.hysteresis = hysteresis
        self.refine_step = refine_step
        self.n_refine = n_refine
        self.finish_margin = finish_margin
        self.follow = FollowPlanner(**self.options())

    def setup(self, track, params, global_plan) -> None:
        super().setup(track, params, global_plan)
        self.follow.setup(track, params, global_plan)

    def reset(self) -> None:
        super().reset()
        self._memory: Optional[_Spec] = None      # manoeuvre in progress
        self.last = None                          # candidates of the last step, for plots

    # ------------------------------------ planning step ---------------------------------------
    def plan(self, obs: Observation, predictions: Dict[str, Trajectory]) -> Trajectory:
        f = self.frame(obs)
        cars = self.relevant_cars(obs, predictions, f)
        follow_c, follow_m, follow_ok = self.follow.candidate(f, cars)
        follow = _Option("follow", follow_m, float(follow_c.cost[follow_m]))
        follow_feasible = bool(follow_ok[follow_m])

        specs = self._specs(f) if cars else []
        full, options, n_ok = None, [], 0
        if specs:
            full, options, n_ok = self._evaluate_specs(f, specs, cars)

        choice = self._choose(follow, follow_feasible, options)
        if choice.kind == "follow":
            self._memory = None
            traj = follow_c.trajectory(choice.row)
        else:
            self._memory = _Spec(choice.width, choice.c1, 0.0, True)
            traj = full.trajectory(choice.row)
        self.decision = Decision(t=f.t0, kind=choice.kind, width=choice.width, c1=choice.c1, c2=choice.c2,
                                 cost=choice.cost, follow_cost=follow.cost,
                                 n_candidates=len(follow_c) + (len(full) if full is not None else 0),
                                 n_feasible=int(follow_ok.sum()) + n_ok)
        self.last = dict(frame=f, follow=follow_c, follow_ok=follow_ok, full=full, options=options, choice=choice)
        return traj

    # ------------------------------------ candidates ---------------------------------------
    def _specs(self, f: Frame) -> List[_Spec]:
        """
        Grid of (w, c1) and the manoeuvre in progress, with the steepness of each lane change.

        The size of w is bounded by the track (proposal 5.2): a lane change whose target offset
        is outside the track limits where it is completed (90 % of the shift) is not generated.
        """
        specs, s_end = [], f.stations[-1]
        step = max(self.c1_step, (s_end - f.s0) / self.n_c1)

        def fits(w, s):
            d = float(self.global_plan.d_at(s)) + f.o0 + 0.9 * (w - f.o0)
            return self.d_min - 0.1 <= d <= self.d_max + 0.1

        for sign in (1.0, -1.0):
            for size in self.widths:
                w = sign * size
                shift = w - f.o0
                if abs(shift) < 0.1:                       # already at this offset: S1 = 1
                    specs.append(_Spec(w, -np.inf, 1.0, False))
                    continue
                for k in range(self.n_c1):
                    start = f.s0 + k * step
                    v_ref = max(f.v0, float(race_speed(self.global_plan, start + 30.0)))
                    a1 = max_steepness(shift, v_ref, self.params, self.curvature_share)
                    c1 = start + 3.0 / a1
                    if c1 + 4.0 / a1 > s_end:
                        break
                    if fits(w, c1 + 2.2 / a1):             # sigma(2.2) = 0.9
                        specs.append(_Spec(w, c1, a1, False))
        if self._memory is not None:
            w, c1 = self._memory.width, self._memory.c1
            v_ref = max(f.v0, float(race_speed(self.global_plan, max(c1, f.s0))))
            a1 = max_steepness(w - f.o0, v_ref, self.params, self.curvature_share)
            specs.append(_Spec(w, c1, a1, True))
        return specs

    def _out_offsets(self, f: Frame, specs: Sequence[_Spec]) -> np.ndarray:
        """ Offsets of the "out" paths: move to w around c1 and stay there. """
        w = np.array([sp.width for sp in specs])[:, None]
        steps = sigmoid_steps(f.stations, [sp.c1 for sp in specs], [sp.a1 for sp in specs], f.s0)
        return f.o0 + (w - f.o0) * steps

    def _evaluate_specs(self, f: Frame, specs: Sequence[_Spec], cars):
        """ Return points, full candidates and their feasibility. """
        out = self._out_offsets(f, specs)
        cap = self._behind_cap(f, cars)
        c_out = self.evaluate(f, out, cap(out))
        targets = self.checked_cars(cars, f)
        k_star = earliest_reentry(c_out, self.params, targets, self.t_hw, self.d_offset) if targets else \
            np.zeros(len(specs), dtype=int)
        s_end = f.stations[-1]
        finish_inside = s_end >= self.track.length - self.finish_margin

        rows, meta = [], []                       # meta: (spec index, c2)
        for i, sp in enumerate(specs):
            if k_star[i] < 0:
                if finish_inside:                 # passing at the finish line: no return needed
                    rows.append(out[i])
                    meta.append((i, np.inf))
                continue
            k = int(k_star[i])
            a2 = max_steepness(sp.width, max(float(c_out.v[i, k]), 1.0), self.params, self.curvature_share)
            c2 = max(f.stations[k] + 1.0 / a2, sp.c1 + 3.0 / sp.a1 + 1.0 / a2)
            for j in range(self.n_refine + 1):
                c2j = c2 + j * self.refine_step
                if c2j + 4.0 / a2 > s_end:        # the return would not be finished in the horizon
                    if finish_inside:
                        rows.append(out[i])
                        meta.append((i, np.inf))
                    break
                back = sp.width * sigmoid_steps(f.stations, c2j, a2, f.s0)[0]
                rows.append(out[i] - back)
                meta.append((i, c2j))
        if not rows:
            return None, [], 0

        rows = np.array(rows)
        full_cap = cap(rows)
        full = self.evaluate(f, rows, full_cap)
        ok = self.feasible(full, cars, f, check_start=True, v_cap=full_cap)
        options, seen = [], set()
        for r, (i, c2) in enumerate(meta):        # first feasible return point of each manoeuvre
            if ok[r] and i not in seen:
                seen.add(i)
                sp = specs[i]
                options.append(_Option("overtake", r, float(full.cost[r]), sp.width, sp.c1, c2, sp.in_progress))
        return full, options, int(ok.sum())

    def _behind_cap(self, f: Frame, cars):
        """
        Speed cap of the candidates while they are still behind the leader in its lane.

        An overtake starts by following (slides 6a: ACC behind the preceding car) and the car
        only drives at its own speed once it has moved out of the leader's lane: until the
        lateral distance to the leader's offset allows the footprints to clear by d_safe, the
        candidate keeps the follow cap of the follow planner.

        Returns:
            function offsets (M, N) -> speed cap (M, N) [m/s], or -> None if there is no leader.
        """
        lead = self.follow.leader(f, cars)
        if lead is None:
            return lambda offsets: None
        follow_cap = self.follow.leader_cap(f, cars)
        o_lead = lead.d_now - float(self.global_plan.d_at(lead.s_now))      # leader offset from our line
        radius = lambda p: np.hypot(p.length / 6.0, p.width / 2.0)          # footprint circles (sim.py)
        separation = radius(self.params) + radius(lead.params) + self.d_safe

        def cap(offsets):
            apart = np.abs(offsets - o_lead) >= separation
            first = np.where(apart.any(axis=1), np.argmax(apart, axis=1), offsets.shape[1])
            k = np.arange(offsets.shape[1])[None, :]
            return np.where(k < first[:, None], follow_cap[None, :], np.inf)
        return cap

    # ------------------------------------ decision ---------------------------------------
    def _choose(self, follow: _Option, follow_feasible: bool, options: Sequence[_Option]) -> _Option:
        """ Fastest feasible option, with hysteresis (slides 6a: publish the minimum-cost path). """
        h = self.hysteresis
        best = min((o for o in options if not o.in_progress), key=lambda o: o.cost, default=None)
        current = next((o for o in options if o.in_progress), None)
        if self._memory is not None and current is not None:            # manoeuvre in progress
            choice = current
            if best is not None and best.cost < current.cost - h:
                choice = best
            if follow_feasible and follow.cost < choice.cost - h:
                choice = follow
            return choice
        if self._memory is not None:                                     # cannot be continued
            if best is not None and (not follow_feasible or best.cost < follow.cost):
                return best
            return follow
        if best is not None and (not follow_feasible or best.cost < follow.cost - h):
            return best
        return follow
