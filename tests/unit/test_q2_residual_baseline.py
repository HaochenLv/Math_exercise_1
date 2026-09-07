from __future__ import annotations

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import (
    Assignment,
    FlightPlan,
    PersonRequest,
    ProblemData,
    SeaStop,
    Solution,
)
from helicopter_planner.solver.q2.residual_baseline import (
    Q2ResidualFlowChainingBaseline,
)


def _problem() -> ProblemData:
    locations = ["A01", "A02", "A03", "F001", "F002", "F003", "F004"]
    distances = {
        a: {b: (0.0 if a == b else 10.0) for b in locations}
        for a in locations
    }
    requests = {
        "P1": PersonRequest("P1", "LAND", "F001"),
        "P2": PersonRequest("P2", "F002", "F003"),
        "P3": PersonRequest("P3", "F003", "F004"),
    }
    return ProblemData(distances=distances, requests=requests)


def test_b0b_chains_two_adjacent_shuttle_ods() -> None:
    problem = _problem()
    q1 = Solution(
        flights={
            "q1": FlightPlan("q1", "A01", "T1", [SeaStop("F001")])
        },
        assignments={
            "P1": Assignment("P1", "q1", 0, 1)
        },
    )
    result = Q2ResidualFlowChainingBaseline(
        packing_time_seconds=2.0
    ).solve(problem, q1)
    assert len(result.chain_decisions) == 1
    assert not result.direct_shuttle_decisions
    assert len(result.solution.assignments) == 3
    assert check_solution(problem, result.solution, require_all_requests=True).ok
