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
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.pair_recombination import (
    Q2ExactPairRecombinationSolver,
    _FlightSummary,
)
from reference_validator import validate_submission


class Q2AllFlightPairRecombinationSolver(Q2ExactPairRecombinationSolver):
    """C3 diagnostic: remove the historical residual/main identity restriction.

    B3A only summarizes q2b0b residual sorties, and B3B explicitly evaluates
    residual x main pairs.  This subclass keeps the exact same route family,
    OD-count CP model, capacity logic, airport/type/order/refuel optimization,
    and candidate-neighborhood machinery, but makes *every* current sortie a
    peer in the pair neighborhood.  This directly opens main x main and all
    other identity combinations without changing any physical constraint.
    """

    def _summaries(self, problem, solution):
        summaries = {}
        for uid in sorted(solution.flights):
            person_ids = tuple(self._person_ids(solution, uid))
            if not person_ids:
                continue
            profile = self.route_optimizer.profile(problem, person_ids)
            sea = frozenset(
                location
                for origin, destination, _ in profile
                for location in (origin, destination)
                if location.startswith("F")
            )
            if not sea:
                continue
            summaries[uid] = _FlightSummary(
                flight_uid=uid,
                person_ids=person_ids,
                profile=profile,
                sea_facilities=sea,
                cost=self._flight_cost(problem, solution, uid),
            )
        return summaries


def metrics_dict(metrics):
    return {
        "total_aircraft_usage_minutes": metrics.total_aircraft_usage_minutes,
        "total_passenger_travel_minutes": metrics.total_passenger_travel_minutes,
        "number_of_flights": metrics.number_of_flights,
        "total_fuel_consumption_kg": metrics.total_fuel_consumption_kg,
        "passenger_km": metrics.passenger_km,
        "available_seat_km": metrics.available_seat_km,
        "seat_utilization": metrics.seat_utilization,
    }


def write_decisions(path: Path, decisions) -> None:
    rows = [asdict(d) for d in decisions]
    fields = list(rows[0]) if rows else [
        "iteration", "source_flight_uids", "pooled_group_count", "union_sea_facilities",
        "cp_status", "cp_objective_aircraft_minutes", "old_aircraft_usage_minutes",
        "new_aircraft_usage_minutes", "immediate_aircraft_savings_minutes",
        "total_iteration_aircraft_savings_minutes", "passenger_minutes_delta", "fuel_kg_delta",
        "old_flight_count", "new_flight_count", "new_aircraft_types", "new_base_airports",
        "new_service_orders", "tail_polish_moves",
    ]
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for row in rows:
            row["source_flight_uids"] = ";".join(row["source_flight_uids"])
            row["union_sea_facilities"] = ";".join(row["union_sea_facilities"])
            row["new_aircraft_types"] = ";".join(row["new_aircraft_types"])
            row["new_base_airports"] = ";".join(row["new_base_airports"])
            row["new_service_orders"] = ";".join("->".join(x) for x in row["new_service_orders"])
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-routes", type=Path, required=True)
    ap.add_argument("--anchor-assignments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/q2/c3_allflight_pair"))
    ap.add_argument("--neighbor-count", type=int, default=6)
    ap.add_argument("--max-union-facilities", type=int, default=7)
    ap.add_argument("--cp-time-limit", type=float, default=1.0)
    ap.add_argument("--max-iterations", type=int, default=8)
    args = ap.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p

    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    distances = ROOT / "data/raw/distances.csv"
    people = ROOT / "data/raw/peopleQ2.csv"
    problem = load_q2_problem(distances, people)
    anchor = load_exported_solution(
        resolve(args.anchor_routes), resolve(args.anchor_assignments), uid_prefix="q2-c3-anchor"
    )
    ck = check_solution(problem, anchor, require_all_requests=True)
    if not ck.ok:
        raise RuntimeError("anchor invalid:\n" + "\n".join(ck.errors))
    am = evaluate_solution(problem, anchor)
    if am.total_aircraft_usage_minutes != 19126 or am.number_of_flights != 108:
        raise RuntimeError(
            f"unexpected anchor {am.total_aircraft_usage_minutes} min / {am.number_of_flights} flights"
        )

    solver = Q2AllFlightPairRecombinationSolver(
        neighbor_count=args.neighbor_count,
        max_union_facilities=args.max_union_facilities,
        cp_time_limit_seconds=args.cp_time_limit,
        max_iterations=args.max_iterations,
        tail_max_hosts_per_donor=32,
    )
    result = solver.solve(problem, anchor)

    routes = out / "q2-routes.csv"
    assignments = out / "q2-assignments.csv"
    export_q1_solution(result.solution, routes, assignments)
    ref = validate_submission(distances, people, routes, assignments, require_all_requests=True)
    if not ref.ok:
        raise RuntimeError("reference validator failed:\n" + "\n".join(ref.errors))

    rm = result.metrics
    metrics = metrics_dict(rm)
    for key, actual in ref.metrics.items():
        expected = metrics[key]
        if isinstance(expected, float):
            if abs(float(expected) - float(actual)) > 1e-6:
                raise RuntimeError(f"metric mismatch {key}: {expected} != {actual}")
        elif expected != actual:
            raise RuntimeError(f"metric mismatch {key}: {expected} != {actual}")

    diag_rows = [asdict(x) for x in result.iteration_diagnostics]
    if diag_rows:
        with (out / "iteration_diagnostics.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(diag_rows[0]))
            w.writeheader(); w.writerows(diag_rows)
    write_decisions(out / "pair_recombinations.csv", result.decisions)

    summary = {
        "stage": "Q2-C3 identity-free all-flight exact pair OD-count recombination",
        "anchor": metrics_dict(am),
        "result": metrics,
        "aircraft_savings_minutes": am.total_aircraft_usage_minutes - rm.total_aircraft_usage_minutes,
        "improvement_ratio": (
            (am.total_aircraft_usage_minutes - rm.total_aircraft_usage_minutes)
            / am.total_aircraft_usage_minutes
        ),
        "flight_count_change": rm.number_of_flights - am.number_of_flights,
        "accepted_pair_moves": len(result.decisions),
        "iterations_scanned": len(result.iteration_diagnostics),
        "candidate_pairs_scanned": sum(x.candidate_pair_count for x in result.iteration_diagnostics),
        "eligible_pairs_scanned": sum(x.eligible_pair_count for x in result.iteration_diagnostics),
        "cp_pair_solves": sum(x.cp_pair_count for x in result.iteration_diagnostics),
        "cp_optimal": sum(x.cp_optimal_count for x in result.iteration_diagnostics),
        "cp_feasible_not_proven": sum(x.cp_feasible_not_proven_count for x in result.iteration_diagnostics),
        "cp_no_solution": sum(x.cp_no_solution_count for x in result.iteration_diagnostics),
        "improving_pairs_seen": sum(x.improving_pair_count for x in result.iteration_diagnostics),
        "route_templates_considered": sum(x.route_template_count for x in result.iteration_diagnostics),
        "neighbor_count": args.neighbor_count,
        "max_union_facilities": args.max_union_facilities,
        "cp_time_limit_seconds": args.cp_time_limit,
        "max_iterations": args.max_iterations,
        "converged": result.converged,
        "reference_validator": "PASS",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(
        json.dumps({"ok": ref.ok, "errors": list(ref.errors), "metrics": ref.metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
