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
from helicopter_planner.solver.q1.count_split_recombination import (
    Q1ExactCountSplitPairSolver,
)


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    facilities = ("F001", "F002", "F003", "F004")
    locations = ("A01", *facilities)
    close_pairs = {
        frozenset(("F001", "F002")),
        frozenset(("F003", "F004")),
    }
    distances = {origin: {} for origin in locations}
    for origin in locations:
        for destination in locations:
            if origin == destination:
                distance = 0.0
            elif "A01" in {origin, destination}:
                distance = 30.0
            elif frozenset((origin, destination)) in close_pairs:
                distance = 2.0
            else:
                distance = 65.0
            distances[origin][destination] = distance

    requests: dict[str, PersonRequest] = {}
    solution = Solution()
    routes = {
        "FLT000001": (("F001", 8), ("F003", 7)),
        "FLT000002": (("F002", 8), ("F004", 7)),
    }
    for uid, blocks in routes.items():
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport="A01",
            aircraft_type="T2",
            sea_stops=[SeaStop(destination) for destination, _ in blocks],
        )
        for delivery_index, (destination, count) in enumerate(blocks, start=1):
            for person_index in range(count):
                pid = f"{uid}_{destination}_{person_index:02d}"
                requests[pid] = PersonRequest(pid, "A01", destination)
                solution.assignments[pid] = Assignment(pid, uid, 0, delivery_index)

    return (
        ProblemData(
            distances=distances,
            requests=requests,
            aircraft_types=AIRCRAFT_TYPES,
        ),
        solution,
    )


def test_exact_count_split_pair_solver_rebuilds_pair_and_stays_feasible():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1ExactCountSplitPairSolver(
        starting_solver=_StubStartingSolver(starting),
        max_iterations=4,
        neighbor_count=1,
        max_union_destinations=4,
        cp_time_limit_seconds=1.0,
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert result.decisions
    assert result.metrics.total_aircraft_usage_minutes < start_metrics.total_aircraft_usage_minutes
    assert len(result.solution.assignments) == len(problem.requests)

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
