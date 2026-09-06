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
from helicopter_planner.solver.q1.cross_airport import Q1LandCrossAirportLocalSearchSolver


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    locations = ("A01", "A02", "F001", "F002")
    raw = {
        ("A01", "A02"): 90.0,
        ("A01", "F001"): 100.0,
        ("A01", "F002"): 100.0,
        ("A02", "F001"): 10.0,
        ("A02", "F002"): 10.0,
        ("F001", "F002"): 5.0,
    }
    distances = {origin: {} for origin in locations}
    for origin in locations:
        for destination in locations:
            if origin == destination:
                distances[origin][destination] = 0.0
            elif (origin, destination) in raw:
                distances[origin][destination] = raw[(origin, destination)]
            elif (destination, origin) in raw:
                distances[origin][destination] = raw[(destination, origin)]
            else:
                raise AssertionError((origin, destination))

    requests: dict[str, PersonRequest] = {}
    solution = Solution()
    specs = {"T1": AIRCRAFT_TYPES["T1"]}

    solution.flights["FLT000001"] = FlightPlan(
        flight_uid="FLT000001",
        base_airport="A01",
        aircraft_type="T1",
        sea_stops=[SeaStop("F001")],
    )
    for i in range(3):
        pid = f"LAND_{i}"
        requests[pid] = PersonRequest(pid, "LAND", "F001")
        solution.assignments[pid] = Assignment(pid, "FLT000001", 0, 1)

    solution.flights["FLT000002"] = FlightPlan(
        flight_uid="FLT000002",
        base_airport="A02",
        aircraft_type="T1",
        sea_stops=[SeaStop("F002")],
    )
    for i in range(3):
        pid = f"FIXED_{i}"
        requests[pid] = PersonRequest(pid, "A02", "F002")
        solution.assignments[pid] = Assignment(pid, "FLT000002", 0, 1)

    return ProblemData(distances=distances, requests=requests, aircraft_types=specs), solution


def test_land_cross_airport_move_can_delete_source_flight():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1LandCrossAirportLocalSearchSolver(
        starting_solver=_StubStartingSolver(starting),
        max_iterations=20,
        polish_max_iterations=20,
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert result.metrics.total_aircraft_usage_minutes < start_metrics.total_aircraft_usage_minutes
    assert len(result.solution.flights) == 1
    assert result.decisions
    decision = result.decisions[0]
    assert decision.source_airport == "A01"
    assert decision.target_airport == "A02"
    assert decision.destination_id == "F001"
    assert decision.moved_passenger_count == 3

    for pid in ("LAND_0", "LAND_1", "LAND_2"):
        assignment = result.solution.assignments[pid]
        assert result.solution.flights[assignment.flight_uid].base_airport == "A02"
    for pid in ("FIXED_0", "FIXED_1", "FIXED_2"):
        assignment = result.solution.assignments[pid]
        assert result.solution.flights[assignment.flight_uid].base_airport == "A02"

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
