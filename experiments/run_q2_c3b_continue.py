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
from reference_validator import validate_submission

from experiments.run_q2_c3_allflight_pair import (
    Q2AllFlightPairRecombinationSolver,
    metrics_dict,
    write_decisions,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-routes", type=Path, required=True)
    ap.add_argument("--anchor-assignments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/q2/c3b_continue"))
    ap.add_argument("--neighbor-count", type=int, default=6)
    ap.add_argument("--max-union-facilities", type=int, default=7)
    ap.add_argument("--cp-time-limit", type=float, default=1.0)
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--expected-anchor-minutes", type=int, default=18959)
    ap.add_argument("--expected-anchor-flights", type=int, default=107)
    args = ap.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p

    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    distances = ROOT / "data/raw/distances.csv"
    people = ROOT / "data/raw/peopleQ2.csv"
    problem = load_q2_problem(distances, people)

    anchor = load_exported_solution(
        resolve(args.anchor_routes),
        resolve(args.anchor_assignments),
        uid_prefix="q2-c3b-anchor",
    )
    ck = check_solution(problem, anchor, require_all_requests=True)
    if not ck.ok:
        raise RuntimeError("anchor invalid:\n" + "\n".join(ck.errors))
    am = evaluate_solution(problem, anchor)
    if (
        am.total_aircraft_usage_minutes != args.expected_anchor_minutes
        or am.number_of_flights != args.expected_anchor_flights
    ):
        raise RuntimeError(
            f"unexpected anchor {am.total_aircraft_usage_minutes} min / "
            f"{am.number_of_flights} flights"
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
            w.writeheader()
            w.writerows(diag_rows)
    write_decisions(out / "pair_recombinations.csv", result.decisions)

    summary = {
        "stage": "Q2-C3b continuation of identity-free all-flight exact pair recombination",
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
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "validation.json").write_text(
        json.dumps(
            {"ok": ref.ok, "errors": list(ref.errors), "metrics": ref.metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
