from helicopter_planner.domain import AIRCRAFT_TYPES, PersonRequest, ProblemData
from helicopter_planner.solver.q1 import Q1SingleFacilityBaselineSolver


def test_land_allocation_uses_existing_capacity_before_opening_new_trip():
    distances = {
        "A01": {"F001": 100.0},
        "A02": {"F001": 140.0},
        "A03": {"F001": 160.0},
        "F001": {"A01": 100.0, "A02": 140.0, "A03": 160.0},
    }
    requests = {}
    for i in range(16):
        pid = f"A01_{i:02d}"
        requests[pid] = PersonRequest(pid, "A01", "F001")
    requests["A02_00"] = PersonRequest("A02_00", "A02", "F001")
    requests["LAND_00"] = PersonRequest("LAND_00", "LAND", "F001")
    requests["LAND_01"] = PersonRequest("LAND_01", "LAND", "F001")

    problem = ProblemData(
        distances=distances,
        requests=requests,
        aircraft_types={"T2": AIRCRAFT_TYPES["T2"]},
    )
    result = Q1SingleFacilityBaselineSolver().solve_with_diagnostics(problem)

    assert len(result.solution.flights) == 2
    assert result.decisions[0].land_allocation == (0, 2, 0)
    for pid in ("LAND_00", "LAND_01"):
        assignment = result.solution.assignments[pid]
        assert result.solution.flights[assignment.flight_uid].base_airport == "A02"
