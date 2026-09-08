from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import Assignment, FlightPlan, PersonRequest, ProblemData, SeaStop, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.solver.q2.pair_recombination import Q2ExactPairRecombinationSolver


def _chain_toy_problem() -> ProblemData:
    locations = ["A01", "A02", "A03", "F001", "F002", "F003"]
    distances = {
        a: {b: (0.0 if a == b else 20.0) for b in locations}
        for a in locations
    }
    requests = {}
    for i in range(12):
        pid = f"a{i:02d}"
        requests[pid] = PersonRequest(pid, "F001", "F002")
    for i in range(12):
        pid = f"b{i:02d}"
        requests[pid] = PersonRequest(pid, "F002", "F003")
    return ProblemData(distances=distances, requests=requests)


def test_b3_recombines_two_residual_routes_into_one_chain() -> None:
    problem = _chain_toy_problem()
    solution = Solution(
        flights={
            "q2b0b-0001": FlightPlan(
                flight_uid="q2b0b-0001",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F001"), SeaStop("F002")],
            ),
            "q2b0b-0002": FlightPlan(
                flight_uid="q2b0b-0002",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F002"), SeaStop("F003")],
            ),
        }
    )
    for i in range(12):
        pid = f"a{i:02d}"
        solution.assignments[pid] = Assignment(pid, "q2b0b-0001", 1, 2)
    for i in range(12):
        pid = f"b{i:02d}"
        solution.assignments[pid] = Assignment(pid, "q2b0b-0002", 1, 2)

    assert check_solution(problem, solution, require_all_requests=True).ok
    before = evaluate_solution(problem, solution)

    result = Q2ExactPairRecombinationSolver(
        neighbor_count=8,
        max_union_facilities=6,
        cp_time_limit_seconds=2.0,
        max_iterations=4,
        tail_max_hosts_per_donor=8,
    ).solve(problem, solution)

    assert result.metrics.total_aircraft_usage_minutes < before.total_aircraft_usage_minutes
    assert result.metrics.number_of_flights == 1
    assert len(result.decisions) == 1
    assert result.decisions[0].new_flight_count == 1
    assert result.decisions[0].cp_status == "OPTIMAL"
    assert check_solution(problem, result.solution, require_all_requests=True).ok
