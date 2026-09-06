from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import (
    AIRCRAFT_TYPES,
    Assignment,
    FlightPlan,
    PersonRequest,
    ProblemData,
    SeaStop,
    Solution,
)
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.solver.q1.partial_relocate import (
    Q1PartialPassengerLocalSearchSolver,
)


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    locations = ("A01", "F001")
    distances = {
        "A01": {"A01": 0.0, "F001": 100.0},
        "F001": {"A01": 100.0, "F001": 0.0},
    }
    requests: dict[str, PersonRequest] = {}
    solution = Solution()

    solution.flights["FLT000001"] = FlightPlan(
        flight_uid="FLT000001",
        base_airport="A01",
        aircraft_type="T3",
        sea_stops=[SeaStop("F001")],
    )
    solution.flights["FLT000002"] = FlightPlan(
        flight_uid="FLT000002",
        base_airport="A01",
        aircraft_type="T1",
        sea_stops=[SeaStop("F001")],
    )

    def add_people(uid: str, count: int) -> None:
        for i in range(count):
            pid = f"{uid}_{i:02d}"
            requests[pid] = PersonRequest(pid, "A01", "F001")
            solution.assignments[pid] = Assignment(pid, uid, 0, 1)

    add_people("FLT000001", 17)
    add_people("FLT000002", 10)

    return ProblemData(
        distances=distances,
        requests=requests,
        aircraft_types=dict(AIRCRAFT_TYPES),
    ), solution


def test_partial_relocate_can_improve_when_whole_block_cannot_move():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1PartialPassengerLocalSearchSolver(
        starting_solver=_StubStartingSolver(starting)
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert result.decisions
    assert result.metrics.total_aircraft_usage_minutes < start_metrics.total_aircraft_usage_minutes
    first = result.decisions[0]
    assert 0 < first.moved_passenger_count < 17
    assert first.immediate_aircraft_savings_minutes > 0
    assert first.total_iteration_aircraft_savings_minutes > 0

    counts = sorted(
        sum(1 for a in result.solution.assignments.values() if a.flight_uid == uid)
        for uid in result.solution.flights
    )
    assert sum(counts) == 27
    assert len(counts) == 2

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
