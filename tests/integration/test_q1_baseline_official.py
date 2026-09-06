from pathlib import Path

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.io.export_solution import export_q1_solution
from helicopter_planner.io.load_problem import load_q1_problem
from helicopter_planner.solver.q1 import Q1SingleFacilityBaselineSolver
from reference_validator.validator import validate_submission


ROOT = Path(__file__).resolve().parents[2]


def test_official_q1_baseline_is_complete_and_independently_valid(tmp_path):
    distances = ROOT / "data/raw/distances.csv"
    people = ROOT / "data/raw/peopleQ1.csv"
    problem = load_q1_problem(distances, people)

    result = Q1SingleFacilityBaselineSolver().solve_with_diagnostics(problem)
    assert len(result.solution.assignments) == 1600
    assert len(result.decisions) == 52

    fast = check_solution(problem, result.solution, require_all_requests=True)
    assert fast.ok, fast.errors

    metrics = evaluate_solution(problem, result.solution)
    assert metrics.total_aircraft_usage_minutes == 17429
    assert metrics.number_of_flights == 110

    routes = tmp_path / "q1-routes.csv"
    assignments = tmp_path / "q1-assignments.csv"
    export_q1_solution(result.solution, routes, assignments)
    report = validate_submission(
        distances, people, routes, assignments, require_all_requests=True
    )
    assert report.ok, report.errors
    assert report.metrics["total_aircraft_usage_minutes"] == 17429
    assert report.metrics["number_of_flights"] == 110
