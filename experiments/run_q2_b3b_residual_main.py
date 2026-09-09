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
from helicopter_planner.solver.q2.pair_recombination import Q2ExactPairRecombinationSolver
from helicopter_planner.solver.q2.residual_baseline import Q2ResidualFlowChainingBaseline
from helicopter_planner.solver.q2.residual_main_recombination import Q2ResidualMainPairRecombinationSolver
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


def _write_decisions(path: Path, decisions) -> None:
    rows = [asdict(decision) for decision in decisions]
    fieldnames = list(rows[0]) if rows else [
        "iteration", "source_flight_uids", "pooled_group_count", "union_sea_facilities",
        "cp_status", "cp_objective_aircraft_minutes", "old_aircraft_usage_minutes",
        "new_aircraft_usage_minutes", "immediate_aircraft_savings_minutes",
        "total_iteration_aircraft_savings_minutes", "passenger_minutes_delta", "fuel_kg_delta",
        "old_flight_count", "new_flight_count", "new_aircraft_types", "new_base_airports",
        "new_service_orders", "tail_polish_moves"
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["source_flight_uids"] = ";".join(row["source_flight_uids"])
            row["union_sea_facilities"] = ";".join(row["union_sea_facilities"])
            row["new_aircraft_types"] = ";".join(row["new_aircraft_types"])
            row["new_base_airports"] = ";".join(row["new_base_airports"])
            row["new_service_orders"] = ";".join("->".join(order) for order in row["new_service_orders"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-routes", type=Path, required=True)
    parser.add_argument("--q1-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2/b3b_residual_main"))
    parser.add_argument("--packing-time-limit", type=float, default=30.0)
    parser.add_argument("--b3a-max-iterations", type=int, default=40)
    parser.add_argument("--main-neighbor-count", type=int, default=10)
    parser.add_argument("--max-union-facilities", type=int, default=7)
    parser.add_argument("--cp-time-limit", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--tail-max-hosts-per-donor", type=int, default=32)
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
    b1 = Q2ResidualTailEliminationSolver(max_hosts_per_donor=24).solve(q2_problem, b0b.solution)

    b3a = Q2ExactPairRecombinationSolver(
        neighbor_count=8,
        max_union_facilities=6,
        cp_time_limit_seconds=0.75,
        max_iterations=args.b3a_max_iterations,
        tail_max_hosts_per_donor=args.tail_max_hosts_per_donor,
    ).solve(q2_problem, b1.solution)
    if not b3a.converged:
        raise RuntimeError("B3A warm start did not converge before B3B")
    if b3a.metrics.total_aircraft_usage_minutes != 19879 or b3a.metrics.number_of_flights != 110:
        raise RuntimeError(
            f"unexpected B3A anchor: {b3a.metrics.total_aircraft_usage_minutes} min / {b3a.metrics.number_of_flights} flights"
        )

    result = Q2ResidualMainPairRecombinationSolver(
        neighbor_count=args.main_neighbor_count,
        max_union_facilities=args.max_union_facilities,
        cp_time_limit_seconds=args.cp_time_limit,
        max_iterations=args.max_iterations,
        tail_max_hosts_per_donor=args.tail_max_hosts_per_donor,
    ).solve(q2_problem, b3a.solution)

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
        "stage": "Q2-B3B exact residual x main pair OD-count recombination",
        "q1_anchor": _metrics_dict(q1_metrics),
        "b0b_complete": _metrics_dict(b0b.metrics),
        "b1_complete": _metrics_dict(b1.metrics),
        "b3a_complete": _metrics_dict(b3a.metrics),
        "b3b_complete": metrics,
        "pair_recombination_count": len(result.decisions),
        "two_to_one_recombination_count": sum(d.new_flight_count == 1 for d in result.decisions),
        "total_aircraft_savings_vs_b3a_minutes": (
            b3a.metrics.total_aircraft_usage_minutes - result.metrics.total_aircraft_usage_minutes
        ),
        "flights_removed_vs_b3a": b3a.metrics.number_of_flights - result.metrics.number_of_flights,
        "pair_iterations_scanned": len(result.iteration_diagnostics),
        "cp_pair_solves": sum(row.cp_pair_count for row in result.iteration_diagnostics),
        "cp_optimal_solves": sum(row.cp_optimal_count for row in result.iteration_diagnostics),
        "cp_feasible_not_proven_solves": sum(
            row.cp_feasible_not_proven_count for row in result.iteration_diagnostics
        ),
        "improving_pairs_seen": sum(row.improving_pair_count for row in result.iteration_diagnostics),
        "route_templates_considered": sum(row.route_template_count for row in result.iteration_diagnostics),
        "main_neighbor_count": args.main_neighbor_count,
        "max_union_facilities": args.max_union_facilities,
        "cp_time_limit_seconds": args.cp_time_limit,
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
        json.dumps(
            {"ok": reference.ok, "errors": list(reference.errors), "metrics": reference.metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_decisions(output_dir / "pair_recombinations.csv", result.decisions)

    rows = [asdict(row) for row in result.iteration_diagnostics]
    fieldnames = list(rows[0]) if rows else [
        "iteration", "residual_flight_count", "candidate_pair_count", "eligible_pair_count",
        "cp_pair_count", "cp_optimal_count", "cp_feasible_not_proven_count",
        "cp_no_solution_count", "improving_pair_count", "route_template_count"
    ]
    with (output_dir / "iteration_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    trace = build_onboard_trace(q2_problem, result.solution)
    with (output_dir / "onboard_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
