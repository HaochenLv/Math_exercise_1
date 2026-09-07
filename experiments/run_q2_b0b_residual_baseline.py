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
from helicopter_planner.io.load_problem import load_q1_problem, load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.residual_baseline import Q2ResidualFlowChainingBaseline
from helicopter_planner.solver.q2.warm_start import build_onboard_trace
from reference_validator import validate_submission


def _metrics_dict(metrics) -> dict[str, float | int]:
    return {
        "total_aircraft_usage_minutes": metrics.total_aircraft_usage_minutes,
        "total_passenger_travel_minutes": metrics.total_passenger_travel_minutes,
        "number_of_flights": metrics.number_of_flights,
        "total_fuel_consumption_kg": metrics.total_fuel_consumption_kg,
        "passenger_km": metrics.passenger_km,
        "available_seat_km": metrics.available_seat_km,
        "seat_utilization": metrics.seat_utilization,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-routes", type=Path, required=True)
    parser.add_argument("--q1-assignments", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q2/b0b_residual_baseline"),
    )
    parser.add_argument("--packing-time-limit", type=float, default=30.0)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    q1_routes = resolve(args.q1_routes)
    q1_assignments = resolve(args.q1_assignments)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    q1_people_path = ROOT / "data/raw/peopleQ1.csv"
    q2_people_path = ROOT / "data/raw/peopleQ2.csv"

    q1_problem = load_q1_problem(distances_path, q1_people_path)
    q2_problem = load_q2_problem(distances_path, q2_people_path)
    q1_anchor = load_exported_solution(
        q1_routes, q1_assignments, uid_prefix="q1-v10-anchor"
    )
    anchor_check = check_solution(q1_problem, q1_anchor, require_all_requests=True)
    if not anchor_check.ok:
        raise RuntimeError(
            "frozen Q1 anchor invalid:\n" + "\n".join(anchor_check.errors)
        )
    anchor_metrics = evaluate_solution(q1_problem, q1_anchor)
    if (
        anchor_metrics.total_aircraft_usage_minutes != 14935
        or anchor_metrics.number_of_flights != 88
    ):
        raise RuntimeError(
            "unexpected Q1 anchor metrics: "
            f"{anchor_metrics.total_aircraft_usage_minutes} min, "
            f"{anchor_metrics.number_of_flights} flights"
        )

    result = Q2ResidualFlowChainingBaseline(
        packing_time_seconds=args.packing_time_limit
    ).solve(q2_problem, q1_anchor)

    routes_path = output_dir / "q2-routes.csv"
    assignments_path = output_dir / "q2-assignments.csv"
    export_q1_solution(result.solution, routes_path, assignments_path)

    reference = validate_submission(
        distances_path,
        q2_people_path,
        routes_path,
        assignments_path,
        require_all_requests=True,
    )
    if not reference.ok:
        raise RuntimeError(
            "Reference Validator failed:\n" + "\n".join(reference.errors)
        )

    metrics = _metrics_dict(result.metrics)
    for key, value in reference.metrics.items():
        expected = metrics[key]
        if isinstance(expected, float):
            if abs(float(expected) - float(value)) > 1e-6:
                raise RuntimeError(f"metric mismatch {key}: {expected} != {value}")
        elif expected != value:
            raise RuntimeError(f"metric mismatch {key}: {expected} != {value}")

    comparison = {
        "stage": "Q2-B0B complete residual flow-chaining baseline",
        "q1_anchor": _metrics_dict(anchor_metrics),
        "b0a_partial": _metrics_dict(result.b0a_metrics),
        "b0a_assigned_count": result.b0a_assigned_count,
        "b0a_remaining_count": result.b0a_remaining_count,
        "b0b_complete": metrics,
        "additional_aircraft_usage_minutes_after_b0a": (
            result.metrics.total_aircraft_usage_minutes
            - result.b0a_metrics.total_aircraft_usage_minutes
        ),
        "additional_flights_after_b0a": (
            result.metrics.number_of_flights - result.b0a_metrics.number_of_flights
        ),
        "number_of_two_edge_chain_routes": len(result.chain_decisions),
        "number_of_direct_shuttle_routes": len(result.direct_shuttle_decisions),
        "number_of_direct_return_routes": len(result.direct_return_decisions),
        "returns_repacked_zero_cost_after_new_shuttle_routes": (
            result.returns_repacked_zero_cost
        ),
        "chain_aircraft_savings_vs_separate_direct_minutes": sum(
            decision.aircraft_savings_minutes or 0
            for decision in result.chain_decisions
        ),
        "reference_validator": "PASS",
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(
            {
                "ok": reference.ok,
                "errors": list(reference.errors),
                "metrics": reference.metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    decisions = [
        *result.chain_decisions,
        *result.direct_shuttle_decisions,
        *result.direct_return_decisions,
    ]
    with (output_dir / "residual_route_decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "kind",
            "flight_uid",
            "base_airport",
            "aircraft_type",
            "service_order",
            "trip_minutes",
            "passenger_counts",
            "separate_direct_minutes",
            "aircraft_savings_minutes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision in decisions:
            row = asdict(decision)
            row["service_order"] = "->".join(decision.service_order)
            row["passenger_counts"] = "+".join(
                str(x) for x in decision.passenger_counts
            )
            writer.writerow(row)

    trace = build_onboard_trace(q2_problem, result.solution)
    with (output_dir / "onboard_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
