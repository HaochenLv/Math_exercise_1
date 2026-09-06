from pathlib import Path
from reference_validator import validate_submission

ROOT = Path(__file__).resolve().parents[2]

def test_official_q1_q2_format_example_is_accepted():
    report = validate_submission(ROOT / "data/raw/distances.csv", ROOT / "tests/golden/official_requests.csv", ROOT / "tests/golden/official_routes.csv", ROOT / "tests/golden/official_assignments.csv")
    assert report.ok, report.errors
    assert report.metrics["number_of_flights"] == 1
    assert report.metrics["total_aircraft_usage_minutes"] == 188
