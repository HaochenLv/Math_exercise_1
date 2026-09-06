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
from helicopter_planner.solver.q1.partial_relocate import (
    Q1PartialPassengerLocalSearchSolver,
)
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
        default=Path("outputs/q1/partial_relocate_v4"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    people_path = ROOT / "data/raw/peopleQ1.csv"
    problem = load_q1_problem(distances_path, people_path)

    result = Q1PartialPassengerLocalSearchSolver().solve_with_diagnostics(problem)
    if not result.converged:
        raise RuntimeError("partial relocate search hit max_iterations before convergence")

    fast_check = check_solution(problem, result.solution, require_all_requests=True)
    if not fast_check.ok:
        raise RuntimeError("fast checker failed:\n" + "\n".join(fast_check.errors))

    metrics = evaluate_solution(problem, result.solution)
    metric_dict = asdict(metrics)
    starting_dict = asdict(result.starting_metrics)

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

    savings = (
        result.starting_metrics.total_aircraft_usage_minutes
        - metrics.total_aircraft_usage_minutes
    )
    comparison = {
        "facility_block_local_search_v3_start": starting_dict,
        "partial_passenger_local_search_v4": metric_dict,
        "additional_aircraft_usage_savings_minutes": savings,
        "additional_aircraft_usage_improvement_ratio": (
            savings / result.starting_metrics.total_aircraft_usage_minutes
        ),
        "number_of_partial_relocate_moves": len(result.decisions),
        "number_of_v3_polish_moves": len(result.polish_decisions),
        "converged": result.converged,
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

    with (output_dir / "partial_relocate_moves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "source_flight_uid",
                "target_flight_uid",
                "destination_id",
                "moved_passenger_count",
                "immediate_aircraft_savings_minutes",
                "total_iteration_aircraft_savings_minutes",
                "immediate_passenger_savings_minutes",
                "immediate_fuel_savings_kg",
                "source_old_aircraft_type",
                "source_after_partial_aircraft_type",
                "target_old_aircraft_type",
                "target_after_partial_aircraft_type",
                "polish_move_count",
            ]
        )
        for d in result.decisions:
            writer.writerow(
                [
                    d.iteration,
                    d.source_flight_uid,
                    d.target_flight_uid,
                    d.destination_id,
                    d.moved_passenger_count,
                    d.immediate_aircraft_savings_minutes,
                    d.total_iteration_aircraft_savings_minutes,
                    d.immediate_passenger_savings_minutes,
                    f"{d.immediate_fuel_savings_kg:.6f}",
                    d.source_old_aircraft_type,
                    d.source_after_partial_aircraft_type,
                    d.target_old_aircraft_type,
                    d.target_after_partial_aircraft_type,
                    d.polish_move_count,
                ]
            )

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
