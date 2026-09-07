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
from helicopter_planner.solver.q2.tail_elimination import Q2ResidualTailEliminationSolver
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2/b1_tail_elimination"))
    parser.add_argument("--packing-time-limit", type=float, default=30.0)
    parser.add_argument("--max-hosts-per-donor", type=int, default=24)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distances_path = ROOT / "data/raw/distances.csv"
    q1_people_path = ROOT / "data/raw/peopleQ1.csv"
    q2_people_path = ROOT / "data/raw/peopleQ2.csv"

    q1_problem = load_q1_problem(distances_path, q1_people_path)
    q2_problem = load_q2_problem(distances_path, q2_people_path)
    q1_anchor = load_exported_solution(
        resolve(args.q1_routes), resolve(args.q1_assignments), uid_prefix="q1-v10-anchor"
    )
    q1_check = check_solution(q1_problem, q1_anchor, require_all_requests=True)
    if not q1_check.ok:
        raise RuntimeError("frozen Q1 anchor invalid:\n" + "\n".join(q1_check.errors))
    q1_metrics = evaluate_solution(q1_problem, q1_anchor)
    if q1_metrics.total_aircraft_usage_minutes != 14935 or q1_metrics.number_of_flights != 88:
        raise RuntimeError(
            f"unexpected Q1 anchor: {q1_metrics.total_aircraft_usage_minutes} min / {q1_metrics.number_of_flights} flights"
        )

    b0b = Q2ResidualFlowChainingBaseline(
        packing_time_seconds=args.packing_time_limit
    ).solve(q2_problem, q1_anchor)
    if b0b.metrics.total_aircraft_usage_minutes != 23068 or b0b.metrics.number_of_flights != 131:
        raise RuntimeError(
            f"unexpected deterministic B0B anchor: {b0b.metrics.total_aircraft_usage_minutes} min / {b0b.metrics.number_of_flights} flights"
        )

    result = Q2ResidualTailEliminationSolver(
        max_hosts_per_donor=args.max_hosts_per_donor
    ).solve(q2_problem, b0b.solution)

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
        raise RuntimeError("Reference Validator failed:\n" + "\n".join(reference.errors))

    metrics = _metrics_dict(result.metrics)
    for key, value in reference.metrics.items():
        expected = metrics[key]
        if isinstance(expected, float):
            if abs(float(expected) - float(value)) > 1e-6:
                raise RuntimeError(f"metric mismatch {key}: {expected} != {value}")
        elif expected != value:
            raise RuntimeError(f"metric mismatch {key}: {expected} != {value}")

    comparison = {
        "stage": "Q2-B1 residual route exact polish + tail elimination",
        "q1_anchor": _metrics_dict(q1_metrics),
        "b0b_complete": _metrics_dict(b0b.metrics),
        "after_single_route_polish": _metrics_dict(result.metrics_after_single_route_polish),
        "b1_complete": metrics,
        "single_route_polish_count": len(result.single_route_polish),
        "single_route_polish_aircraft_savings_minutes": (
            result.starting_metrics.total_aircraft_usage_minutes
            - result.metrics_after_single_route_polish.total_aircraft_usage_minutes
        ),
        "tail_elimination_count": len(result.tail_eliminations),
        "tail_elimination_aircraft_savings_minutes": sum(
            decision.aircraft_savings_minutes for decision in result.tail_eliminations
        ),
        "total_aircraft_savings_vs_b0b_minutes": (
            b0b.metrics.total_aircraft_usage_minutes - result.metrics.total_aircraft_usage_minutes
        ),
        "flights_removed_vs_b0b": b0b.metrics.number_of_flights - result.metrics.number_of_flights,
        "eliminated_return_tails": sum(
            decision.donor_kind == "return" for decision in result.tail_eliminations
        ),
        "eliminated_shuttle_tails": sum(
            decision.donor_kind == "shuttle" for decision in result.tail_eliminations
        ),
        "converged": result.converged,
        "reference_validator": "PASS",
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps({"ok": reference.ok, "errors": list(reference.errors), "metrics": reference.metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "single_route_polish.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [asdict(decision) for decision in result.single_route_polish]
        fieldnames = list(rows[0]) if rows else [
            "flight_uid", "before_minutes", "after_minutes", "aircraft_savings_minutes",
            "before_aircraft_type", "after_aircraft_type", "before_base_airport",
            "after_base_airport", "after_service_order"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "after_service_order" in row:
                row["after_service_order"] = "->".join(row["after_service_order"])
            writer.writerow(row)

    elimination_rows: list[dict[str, object]] = []
    for decision in result.tail_eliminations:
        elimination_rows.append(
            {
                "donor_flight_uid": decision.donor_flight_uid,
                "donor_kind": decision.donor_kind,
                "donor_passenger_count": decision.donor_passenger_count,
                "donor_minutes": decision.donor_minutes,
                "added_recipient_minutes": decision.added_recipient_minutes,
                "aircraft_savings_minutes": decision.aircraft_savings_minutes,
                "passenger_minutes_delta": decision.passenger_minutes_delta,
                "fuel_kg_delta": decision.fuel_kg_delta,
                "recipients": ";".join(
                    f"{recipient.flight_uid}:{recipient.passenger_count}:{recipient.added_aircraft_minutes}:"
                    f"{recipient.after_aircraft_type}:{recipient.after_base_airport}:"
                    f"{'->'.join(recipient.after_service_order)}"
                    for recipient in decision.recipients
                ),
            }
        )
    with (output_dir / "tail_eliminations.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "donor_flight_uid", "donor_kind", "donor_passenger_count", "donor_minutes",
            "added_recipient_minutes", "aircraft_savings_minutes", "passenger_minutes_delta",
            "fuel_kg_delta", "recipients"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(elimination_rows)

    trace = build_onboard_trace(q2_problem, result.solution)
    with (output_dir / "onboard_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
