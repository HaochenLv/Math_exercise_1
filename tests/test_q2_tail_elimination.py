from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import Assignment, FlightPlan, PersonRequest, ProblemData, SeaStop, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.solver.q2.tail_elimination import Q2ResidualTailEliminationSolver


def _toy_problem() -> ProblemData:
    locations = ["A01", "A02", "A03", "F001", "F002"]
    distances = {
        a: {b: (0.0 if a == b else 20.0) for b in locations}
        for a in locations
    }
    requests = {
        "p0": PersonRequest("p0", "LAND", "F001"),
        "p1": PersonRequest("p1", "F001", "F002"),
        "d1": PersonRequest("d1", "F001", "LAND"),
        "d2": PersonRequest("d2", "F001", "LAND"),
    }
    return ProblemData(distances=distances, requests=requests)


def test_b1_eliminates_small_return_tail() -> None:
    problem = _toy_problem()
    solution = Solution(
        flights={
            "host": FlightPlan(
                flight_uid="host",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F001"), SeaStop("F002")],
            ),
            "q2b0b-0001": FlightPlan(
                flight_uid="q2b0b-0001",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F001")],
            ),
        },
        assignments={
            "p0": Assignment("p0", "host", 0, 1),
            "p1": Assignment("p1", "host", 1, 2),
            "d1": Assignment("d1", "q2b0b-0001", 1, 2),
            "d2": Assignment("d2", "q2b0b-0001", 1, 2),
        },
    )
    assert check_solution(problem, solution, require_all_requests=True).ok
    before = evaluate_solution(problem, solution)

    result = Q2ResidualTailEliminationSolver(max_hosts_per_donor=8).solve(problem, solution)

    assert result.metrics.total_aircraft_usage_minutes < before.total_aircraft_usage_minutes
    assert result.metrics.number_of_flights == 1
    assert len(result.tail_eliminations) == 1
    assert result.tail_eliminations[0].donor_kind == "return"
    assert check_solution(problem, result.solution, require_all_requests=True).ok
    assert {a.flight_uid for a in result.solution.assignments.values()} == {"host"}
