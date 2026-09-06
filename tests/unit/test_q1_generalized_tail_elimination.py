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
from helicopter_planner.solver.q1.generalized_tail_elimination import (
    Q1RetypeAwareTailEliminationSolver,
)


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    locations = ("A01", "F001", "F002")
    raw = {
        ("A01", "F001"): 100.0,
        ("A01", "F002"): 100.0,
        ("F001", "F002"): 5.0,
    }
    distances = {origin: {} for origin in locations}
    for origin in locations:
        for destination in locations:
            if origin == destination:
                distances[origin][destination] = 0.0
            elif (origin, destination) in raw:
                distances[origin][destination] = raw[(origin, destination)]
            else:
                distances[origin][destination] = raw[(destination, origin)]

    requests: dict[str, PersonRequest] = {}
    solution = Solution()
    specs = {
        "T1": AIRCRAFT_TYPES["T1"],
        "T3": AIRCRAFT_TYPES["T3"],
    }

    def add_flight(
        uid: str,
        destination: str,
        count: int,
    ) -> None:
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport="A01",
            aircraft_type="T1",
            sea_stops=[SeaStop(destination)],
        )
        for i in range(count):
            pid = f"{uid}_{i:02d}"
            requests[pid] = PersonRequest(pid, "A01", destination)
            solution.assignments[pid] = Assignment(pid, uid, 0, 1)

    # The first T1 is full, so V2's current-type spare-seat rule cannot absorb
    # any donor passengers. V5 may retype the recipient to T3 and combine all
    # 18 passengers in one two-destination route.
    add_flight("FLT000001", "F001", 12)
    add_flight("FLT000002", "F002", 6)

    return (
        ProblemData(
            distances=distances,
            requests=requests,
            aircraft_types=specs,
        ),
        solution,
    )


def test_retype_aware_tail_elimination_can_upgrade_recipient_and_delete_donor():
    problem, starting = _toy_problem_and_solution()
    start_metrics = evaluate_solution(problem, starting)

    result = Q1RetypeAwareTailEliminationSolver(
        starting_solver=_StubStartingSolver(starting),
    ).solve_with_diagnostics(problem)

    assert result.converged
    assert len(result.solution.flights) == 1
    assert result.metrics.number_of_flights == 1
    assert (
        result.metrics.total_aircraft_usage_minutes
        < start_metrics.total_aircraft_usage_minutes
    )
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.donor_passenger_count in {6, 12}
    assert sum(
        count for _, count in decision.recipient_allocations
    ) == decision.donor_passenger_count
    assert decision.recipient_type_changes
    assert any(
        old_type == "T1" and new_type == "T3"
        for _, old_type, new_type in decision.recipient_type_changes
    )

    only_flight = next(iter(result.solution.flights.values()))
    assert only_flight.aircraft_type == "T3"
    served = {
        problem.requests[assignment.person_id].destination_id
        for assignment in result.solution.assignments.values()
    }
    assert served == {"F001", "F002"}

    check = check_solution(
        problem,
        result.solution,
        require_all_requests=True,
    )
    assert check.ok, check.errors
