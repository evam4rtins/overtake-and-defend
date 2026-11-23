"""
Local planners of Phase 1 (proposal Section 5).

- base.py:    common interface LocalPlanner.plan(obs, predictions) -> Trajectory and shared helpers
- follow.py:  racing line + ACC, the Phase 1 opponent and the "stay behind" option
- sigmoid.py: Approach A, sigmoid candidates with minimum-time selection
- lattice.py: Approach B, graph search (to do)
"""

from .base import Decision, LocalPlanner
from .follow import FollowPlanner
from .sigmoid import SigmoidPlanner

PLANNERS = {"follow": FollowPlanner, "sigmoid": SigmoidPlanner}

__all__ = ["Decision", "LocalPlanner", "FollowPlanner", "SigmoidPlanner", "PLANNERS"]
