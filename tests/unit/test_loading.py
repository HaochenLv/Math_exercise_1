from pathlib import Path
from helicopter_planner.io import load_q1_problem

ROOT = Path(__file__).resolve().parents[2]

def test_q1_loader_on_real_schema_sample_and_full_distance_matrix():
    problem = load_q1_problem(ROOT / "data/raw/distances.csv", ROOT / "tests/fixtures/peopleQ1_sample.csv")
    assert len(problem.requests) == 6
    assert len(problem.distances) == 55
    assert problem.requests["P0002"].origin_id == "LAND"
    assert problem.requests["P0002"].destination_id == "F022"
    assert problem.distance("A01", "F006") == 235
