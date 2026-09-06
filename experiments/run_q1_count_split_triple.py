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
from helicopter_planner.solver.q1.count_split_triple import Q1ExactCountSplitTripleSolver
from reference_validator.validator import validate_submission


def _assert_metric_agreement(solver_metrics: dict, validator_metrics: dict) -> None:
    for key, expected in solver_metrics.items():
        actual = validator_metrics[key]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-6:
                raise RuntimeError(f"metric mismatch for {key}: solver={expected}, validator={actual}")
        elif actual != expected:
            raise RuntimeError(f"metric mismatch for {key}: solver={expected}, validator={actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q1/count_split_triple_v11"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    people_path = ROOT / "data/raw/peopleQ1.csv"
    problem = load_q1_problem(distances_path, people_path)

    result = Q1ExactCountSplitTripleSolver().solve_with_diagnostics(problem)
    if not result.converged:
        raise RuntimeError("V11 hit max_iterations before convergence")

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
        raise RuntimeError("reference validator failed:\n" + "\n".join(validation.errors))
    _assert_metric_agreement(metric_dict, validation.metrics)

    savings = result.starting_metrics.total_aircraft_usage_minutes - metrics.total_aircraft_usage_minutes
    comparison = {
        "exact_count_split_pair_v10_start": starting_dict,
        "exact_count_split_triple_v11": metric_dict,
        "additional_aircraft_usage_savings_minutes": savings,
        "additional_aircraft_usage_improvement_ratio": (
            savings / result.starting_metrics.total_aircraft_usage_minutes
        ),
        "flight_count_change": metrics.number_of_flights - result.starting_metrics.number_of_flights,
        "number_of_count_split_triple_moves": len(result.decisions),
        "number_of_pair_polish_moves": len(result.pair_polish_decisions),
        "converged": result.converged,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metric_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation.json").write_text(
        json.dumps({"ok": validation.ok, "errors": list(validation.errors), "metrics": validation.metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "count_split_triple_moves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "base_airport", "source_flight_uids", "destinations",
            "old_flight_count", "new_flight_count", "old_aircraft_usage_minutes",
            "new_aircraft_usage_minutes", "immediate_aircraft_savings_minutes",
            "total_iteration_aircraft_savings_minutes", "immediate_passenger_savings_minutes",
            "immediate_fuel_savings_kg", "new_aircraft_types", "new_service_orders",
            "pair_polish_moves",
        ])
        for d in result.decisions:
            writer.writerow([
                d.iteration, d.base_airport, ";".join(d.source_flight_uids), ";".join(d.destinations),
                d.old_flight_count, d.new_flight_count, d.old_aircraft_usage_minutes,
                d.new_aircraft_usage_minutes, d.immediate_aircraft_savings_minutes,
                d.total_iteration_aircraft_savings_minutes, d.immediate_passenger_savings_minutes,
                f"{d.immediate_fuel_savings_kg:.6f}", ";".join(d.new_aircraft_types),
                " | ".join(";".join(order) for order in d.new_service_orders), d.pair_polish_moves,
            ])

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
