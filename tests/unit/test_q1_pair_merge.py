from helicopter_planner.domain import AIRCRAFT_TYPES, PersonRequest, ProblemData
from helicopter_planner.checking.checker import check_solution
from helicopter_planner.solver.q1.pair_merge import Q1GreedyPairSavingsSolver
from helicopter_planner.solver.q1.route_builder import Q1RouteBuilder


def _toy_problem() -> ProblemData:
    locations = ("A01", "A02", "A03", "F001", "F002")
    D = {a: {b: 0.0 for b in locations} for a in locations}

    def set_sym(a: str, b: str, d: float) -> None:
        D[a][b] = d
        D[b][a] = d

    set_sym("A01", "F001", 100.0)
    set_sym("A01", "F002", 100.0)
    set_sym("F001", "F002", 20.0)
    for airport, base in (("A02", 150.0), ("A03", 160.0)):
        set_sym(airport, "F001", base)
        set_sym(airport, "F002", base)
    set_sym("A01", "A02", 50.0)
    set_sym("A01", "A03", 60.0)
    set_sym("A02", "A03", 40.0)

    requests = {}
    for i in range(4):
        pid = f"P1_{i}"
        requests[pid] = PersonRequest(pid, "A01", "F001")
    for i in range(4):
        pid = f"P2_{i}"
        requests[pid] = PersonRequest(pid, "A01", "F002")

    return ProblemData(
        distances=D,
        requests=requests,
        aircraft_types={"T2": AIRCRAFT_TYPES["T2"]},
    )


def test_route_builder_two_services_direct():
    problem = _toy_problem()
    route = Q1RouteBuilder().build(problem, "A01", ("F001", "F002"), "T2")
    assert route is not None
    assert [s.facility_id for s in route.sea_stops] == ["F001", "F002"]
    assert route.service_stop_indices == (1, 2)
    assert route.service_arrival_minutes == (28, 44)
    assert route.trip_minutes == 82


def test_pair_merge_reduces_two_direct_baseline_flights():
    problem = _toy_problem()
    result = Q1GreedyPairSavingsSolver().solve_with_diagnostics(problem)
    assert result.baseline_metrics.number_of_flights == 2
    assert result.metrics.number_of_flights == 1
    assert result.baseline_metrics.total_aircraft_usage_minutes == 132
    assert result.metrics.total_aircraft_usage_minutes == 82
    assert len(result.decisions) == 1
    assert result.decisions[0].aircraft_savings_minutes == 50
    checked = check_solution(problem, result.solution, require_all_requests=True)
    assert checked.ok, checked.errors
