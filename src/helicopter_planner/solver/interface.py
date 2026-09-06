from __future__ import annotations
from typing import Protocol
from helicopter_planner.domain import ProblemData, Solution

class Solver(Protocol):
    """Stable plugin boundary for Phase 2+ optimization algorithms."""
    def solve(self, problem: ProblemData) -> Solution:
        ...
