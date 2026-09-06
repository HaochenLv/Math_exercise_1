from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.io.export_solution import export_q1_solution
from helicopter_planner.io.load_problem import load_q1_problem
from helicopter_planner.solver.q1 import Q1SingleFacilityBaselineSolver
from reference_validator.validator import validate_submission


def _assert_metric_agreement(solver_metrics: dict, validator_metrics: dict) -> None:
    for key, expected in solver_metrics.items():
        actual = validator_metrics[key]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-6:
                raise RuntimeError(
                    f"metric mismatch for {key}: solver={expected}, validator={actual}"
                )
        elif actual != expected:
            raise RuntimeError(
                f"metric mismatch for {key}: solver={expected}, validator={actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q1/baseline"),
    )
    args = parser.parse_args()

    distances_path = ROOT / "data/raw/distances.csv"
    people_path = ROOT / "data/raw/peopleQ1.csv"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = load_q1_problem(distances_path, people_path)
    solver = Q1SingleFacilityBaselineSolver()
    result = solver.solve_with_diagnostics(problem)

    fast_check = check_solution(problem, result.solution, require_all_requests=True)
    if not fast_check.ok:
        raise RuntimeError("fast checker failed:\n" + "\n".join(fast_check.errors))

    metrics = evaluate_solution(problem, result.solution)
    metric_dict = asdict(metrics)

    routes_path = output_dir / "q1-routes.csv"
    assignments_path = output_dir / "q1-assignments.csv"
    export_q1_solution(result.solution, routes_path, assignments_path)

    validation = validate_submission(
        distances_path,
        people_path,
        routes_path,
        assignments_path,
        require_all_requests=True,
    )
    if not validation.ok:
        raise RuntimeError(
            "reference validator failed:\n" + "\n".join(validation.errors)
        )
    _assert_metric_agreement(metric_dict, validation.metrics)

    (output_dir / "metrics.json").write_text(
        json.dumps(metric_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation.json").write_text(
        json.dumps(
            {
                "ok": validation.ok,
                "errors": list(validation.errors),
                "metrics": validation.metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "facility_decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "facility_id",
                "aircraft_type",
                "land_A01",
                "land_A02",
                "land_A03",
                "trips_A01",
                "trips_A02",
                "trips_A03",
                "aircraft_usage_minutes",
                "passenger_travel_minutes",
                "fuel_kg",
                "number_of_flights",
            ]
        )
        for d in result.decisions:
            writer.writerow(
                [
                    d.facility_id,
                    d.aircraft_type,
                    *d.land_allocation,
                    *d.trip_counts,
                    d.total_aircraft_usage_minutes,
                    d.total_passenger_travel_minutes,
                    f"{d.total_fuel_consumption_kg:.6f}",
                    d.number_of_flights,
                ]
            )

    print(json.dumps(metric_dict, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
