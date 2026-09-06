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
from helicopter_planner.solver.q1 import Q1GreedyPairSavingsSolver
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
        default=Path("outputs/q1/pair_merge_v1"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    people_path = ROOT / "data/raw/peopleQ1.csv"
    problem = load_q1_problem(distances_path, people_path)

    result = Q1GreedyPairSavingsSolver().solve_with_diagnostics(problem)
    fast_check = check_solution(problem, result.solution, require_all_requests=True)
    if not fast_check.ok:
        raise RuntimeError("fast checker failed:\n" + "\n".join(fast_check.errors))

    metrics = evaluate_solution(problem, result.solution)
    metric_dict = asdict(metrics)
    baseline_dict = asdict(result.baseline_metrics)

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

    comparison = {
        "baseline": baseline_dict,
        "pair_merge_v1": metric_dict,
        "aircraft_usage_savings_minutes": (
            result.baseline_metrics.total_aircraft_usage_minutes
            - metrics.total_aircraft_usage_minutes
        ),
        "aircraft_usage_improvement_ratio": (
            (
                result.baseline_metrics.total_aircraft_usage_minutes
                - metrics.total_aircraft_usage_minutes
            )
            / result.baseline_metrics.total_aircraft_usage_minutes
        ),
        "number_of_merges": len(result.decisions),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metric_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
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

    with (output_dir / "merge_decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "left_flight_uid",
                "right_flight_uid",
                "base_airport",
                "left_destination",
                "right_destination",
                "aircraft_type",
                "service_order",
                "old_aircraft_usage_minutes",
                "new_aircraft_usage_minutes",
                "aircraft_savings_minutes",
                "old_passenger_travel_minutes",
                "new_passenger_travel_minutes",
                "old_fuel_kg",
                "new_fuel_kg",
            ]
        )
        for d in result.decisions:
            writer.writerow(
                [
                    d.left_flight_uid,
                    d.right_flight_uid,
                    d.base_airport,
                    d.left_destination,
                    d.right_destination,
                    d.aircraft_type,
                    "->".join(d.service_order),
                    d.old_aircraft_usage_minutes,
                    d.new_aircraft_usage_minutes,
                    d.aircraft_savings_minutes,
                    d.old_passenger_travel_minutes,
                    d.new_passenger_travel_minutes,
                    f"{d.old_fuel_kg:.6f}",
                    f"{d.new_fuel_kg:.6f}",
                ]
            )

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
