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
from helicopter_planner.solver.q1.local_search import Q1FacilityBlockLocalSearchSolver


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    locations = ("A01", "F001", "F002", "F003")
    raw = {
        ("A01", "F001"): 50.0,
        ("A01", "F002"): 50.0,
        ("A01", "F003"): 50.0,
        ("F001", "F002"): 100.0,
        ("F001", "F003"): 100.0,
        ("F002", "F003"): 5.0,
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

    def add_people(uid: str, destination: str, count: int, delivery_index: int) -> None:
        for i in range(count):
            pid = f"{uid}_{destination}_{i:02d}"
            requests[pid] = PersonRequest(pid, "A01", destination)
            solution.assignments[pid] = Assignment(pid, uid, 0, delivery_index)

    solution.flights["FLT000001"] = FlightPlan(
        flight_uid="FLT000001",
        base_airport="A01",
        aircraft_type="T1",
        sea_stops=[SeaStop("F001"), SeaStop("F003")],
    )
    add_people("FLT000001", "F001", 3, 1)
    add_people("FLT000001", "F003", 3, 2)

    solution.flights["FLT000002"] = FlightPlan(
        flight_uid="FLT000002",
        base_airport="A01",
        aircraft_type="T1",
        sea_stops=[SeaStop("F002")],
    )
    add_people("FLT000002", "F002", 3, 1)

    return ProblemData(distances=distances, requests=requests, aircraft_types=specs), solution


def test_facility_block_relocate_improves_pair_and_stays_feasible():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1FacilityBlockLocalSearchSolver(
        starting_solver=_StubStartingSolver(starting)
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert result.metrics.total_aircraft_usage_minutes < start_metrics.total_aircraft_usage_minutes
    assert result.decisions
    assert result.decisions[0].move_type in {"relocate", "relocate_reverse"}

    destinations_by_flight = {}
    for uid in result.solution.flights:
        destinations_by_flight[uid] = {
            problem.requests[a.person_id].destination_id
            for a in result.solution.assignments.values()
            if a.flight_uid == uid
        }
    assert {"F002", "F003"} in destinations_by_flight.values()

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
