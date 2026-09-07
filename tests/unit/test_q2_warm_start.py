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
from helicopter_planner.solver.q2.warm_start import (
    Q2WarmStartPacker,
    build_onboard_trace,
)


def _toy_problem_and_warm_start() -> tuple[ProblemData, Solution]:
    locations = ("A01", "F001", "F002")
    distances = {
        origin: {
            destination: (
                0.0
                if origin == destination
                else 20.0
                if "A01" in {origin, destination}
                else 5.0
            )
            for destination in locations
        }
        for origin in locations
    }
    requests = {
        "O1": PersonRequest("O1", "A01", "F001"),
        "O2": PersonRequest("O2", "A01", "F002"),
        "R1": PersonRequest("R1", "F001", "LAND"),
        "S1": PersonRequest("S1", "F001", "F002"),
        "R2": PersonRequest("R2", "F002", "LAND"),
    }
    solution = Solution(
        flights={
            "FLT000001": FlightPlan(
                flight_uid="FLT000001",
                base_airport="A01",
                aircraft_type="T1",
                sea_stops=[SeaStop("F001"), SeaStop("F002")],
            )
        },
        assignments={
            "O1": Assignment("O1", "FLT000001", 0, 1),
            "O2": Assignment("O2", "FLT000001", 0, 2),
        },
    )
    return (
        ProblemData(
            distances=distances,
            requests=requests,
            aircraft_types=AIRCRAFT_TYPES,
        ),
        solution,
    )


def test_q2_b0a_packs_return_and_shuttle_without_changing_aircraft_cost():
    problem, warm_start = _toy_problem_and_warm_start()
    start_metrics = evaluate_solution(problem, warm_start)

    result = Q2WarmStartPacker(max_time_seconds=2.0).pack(
        problem, warm_start
    )

    assert result.packed_extra_count == 3
    assert result.packed_by_kind == {"return": 2, "shuttle": 1}
    assert set(result.solution.assignments) == set(problem.requests)
    assert (
        result.metrics.total_aircraft_usage_minutes
        == start_metrics.total_aircraft_usage_minutes
    )
    assert result.metrics.number_of_flights == start_metrics.number_of_flights
    assert (
        abs(
            result.metrics.total_fuel_consumption_kg
            - start_metrics.total_fuel_consumption_kg
        )
        < 1e-9
    )

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors

    trace = build_onboard_trace(problem, result.solution)
    f001 = next(row for row in trace if row["location"] == "F001")
    assert f001["dropoffs"] == 1
    assert f001["pickups"] == 2
    assert (
        f001["onboard_after"]
        == f001["onboard_before"] - f001["dropoffs"] + f001["pickups"]
    )
