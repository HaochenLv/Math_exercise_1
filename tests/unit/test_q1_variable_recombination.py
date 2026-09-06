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
from helicopter_planner.solver.q1.variable_recombination import (
    Q1VariableFlightCountRecombinationSolver,
)


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    facilities = tuple(f"F{i:03d}" for i in range(1, 7))
    locations = ("A01", *facilities)
    cluster_pairs = {
        frozenset(("F001", "F002")),
        frozenset(("F003", "F004")),
        frozenset(("F005", "F006")),
    }
    distances = {origin: {} for origin in locations}
    for origin in locations:
        for destination in locations:
            if origin == destination:
                distance = 0.0
            elif "A01" in {origin, destination}:
                distance = 20.0
            elif frozenset((origin, destination)) in cluster_pairs:
                distance = 2.0
            else:
                distance = 60.0
            distances[origin][destination] = distance

    requests: dict[str, PersonRequest] = {}
    solution = Solution()
    routes = {
        "FLT000001": ("F001", "F003", "F005"),
        "FLT000002": ("F002", "F004", "F006"),
    }
    for uid, destinations in routes.items():
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport="A01",
            aircraft_type="T1",
            sea_stops=[SeaStop(destination) for destination in destinations],
        )
        for delivery_index, destination in enumerate(destinations, start=1):
            for person_index in range(3):
                pid = f"{uid}_{destination}_{person_index:02d}"
                requests[pid] = PersonRequest(pid, "A01", destination)
                solution.assignments[pid] = Assignment(pid, uid, 0, delivery_index)

    return (
        ProblemData(
            distances=distances,
            requests=requests,
            aircraft_types={"T1": AIRCRAFT_TYPES["T1"]},
        ),
        solution,
    )


def test_variable_recombination_can_add_a_flight_to_reduce_total_time():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1VariableFlightCountRecombinationSolver(
        starting_solver=_StubStartingSolver(starting),
        max_iterations=3,
        max_blocks=6,
        neighbor_count=2,
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert result.decisions
    assert result.decisions[0].old_flight_count == 2
    assert result.decisions[0].new_flight_count == 3
    assert result.metrics.number_of_flights == 3
    assert result.metrics.total_aircraft_usage_minutes < start_metrics.total_aircraft_usage_minutes

    destination_sets = {
        frozenset(
            problem.requests[assignment.person_id].destination_id
            for assignment in result.solution.assignments.values()
            if assignment.flight_uid == uid
        )
        for uid in result.solution.flights
    }
    assert frozenset(("F001", "F002")) in destination_sets
    assert frozenset(("F003", "F004")) in destination_sets
    assert frozenset(("F005", "F006")) in destination_sets

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
