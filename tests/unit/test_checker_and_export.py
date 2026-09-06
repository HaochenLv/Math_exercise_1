from pathlib import Path
from helicopter_planner.checking import check_solution
from helicopter_planner.domain import Assignment, FlightPlan, SeaStop, Solution
from helicopter_planner.evaluation import evaluate_solution
from helicopter_planner.io import export_q1_solution, load_q1_problem
from reference_validator import validate_submission

ROOT = Path(__file__).resolve().parents[2]

def test_handcrafted_q1_solution_roundtrip(tmp_path):
    problem = load_q1_problem(ROOT / "data/raw/distances.csv", ROOT / "tests/fixtures/peopleQ1_sample.csv")
    flight = FlightPlan("FLT000001", "A02", "T3", [SeaStop("F021", False)])
    solution = Solution(flights={flight.flight_uid: flight}, assignments={"P0008": Assignment("P0008", flight.flight_uid, 0, 1)})
    result = check_solution(problem, solution, require_all_requests=False)
    assert result.ok, result.errors
    metrics = evaluate_solution(problem, solution)
    assert metrics.number_of_flights == 1
    assert metrics.total_aircraft_usage_minutes > 0
    routes = tmp_path / "routes.csv"
    assignments = tmp_path / "assignments.csv"
    export_q1_solution(solution, routes, assignments)
    request_file = tmp_path / "requests.csv"
    request_file.write_text("person_id,origin_id,destination_id\nP0008,A02,F021\n", encoding="utf-8")
    report = validate_submission(ROOT / "data/raw/distances.csv", request_file, routes, assignments)
    assert report.ok, report.errors
    assert report.metrics["number_of_flights"] == 1
