from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import Assignment, FlightPlan, PersonRequest, ProblemData, SeaStop, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.solver.q2.partial_relocate import Q2PartialPassengerRelocationSolver


def _toy_problem() -> ProblemData:
    locations = ["A01", "A02", "A03", "F001", "F002"]
    distances = {
        a: {b: (0.0 if a == b else 20.0) for b in locations}
        for a in locations
    }
    requests = {}
    for i in range(13):
        pid = f"d{i:02d}"
        requests[pid] = PersonRequest(pid, "F001", "F002")
    for i in range(11):
        pid = f"h{i:02d}"
        requests[pid] = PersonRequest(pid, "F001", "F002")
    return ProblemData(distances=distances, requests=requests)


def test_b2_moves_one_passenger_to_cross_capacity_threshold() -> None:
    problem = _toy_problem()
    solution = Solution(
        flights={
            "q2b0b-0001": FlightPlan(
                flight_uid="q2b0b-0001",
                base_airport="A01",
                aircraft_type="T2",
                sea_stops=[SeaStop("F001"), SeaStop("F002")],
            ),
            "host": FlightPlan(
                flight_uid="host",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F001"), SeaStop("F002")],
            ),
        }
    )
    for i in range(13):
        pid = f"d{i:02d}"
        solution.assignments[pid] = Assignment(pid, "q2b0b-0001", 1, 2)
    for i in range(11):
        pid = f"h{i:02d}"
        solution.assignments[pid] = Assignment(pid, "host", 1, 2)

    assert check_solution(problem, solution, require_all_requests=True).ok
    before = evaluate_solution(problem, solution)

    result = Q2PartialPassengerRelocationSolver(
        max_hosts_per_donor=8,
        tail_max_hosts_per_donor=8,
    ).solve(problem, solution)

    assert result.metrics.total_aircraft_usage_minutes < before.total_aircraft_usage_minutes
    assert len(result.partial_relocations) >= 1
    first = result.partial_relocations[0]
    assert first.moved_passenger_count == 1
    assert first.donor_aircraft_type_after == "T1"
    assert result.metrics.number_of_flights == 2
    assert check_solution(problem, result.solution, require_all_requests=True).ok
