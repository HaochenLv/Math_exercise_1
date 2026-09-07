from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.io.export_solution import export_q1_solution
from helicopter_planner.io.load_problem import load_q1_problem, load_q2_problem
from helicopter_planner.solver.q1.count_split_recombination import (
    Q1ExactCountSplitPairSolver,
)
from helicopter_planner.solver.q2.warm_start import (
    Q2WarmStartPacker,
    build_onboard_trace,
    classify_q2_request,
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
        default=Path("outputs/q2/b0a_warm_start"),
    )
    parser.add_argument(
        "--packing-time-limit",
        type=float,
        default=30.0,
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    q1_people_path = ROOT / "data/raw/peopleQ1.csv"
    q2_people_path = ROOT / "data/raw/peopleQ2.csv"

    q1_problem = load_q1_problem(distances_path, q1_people_path)
    q2_problem = load_q2_problem(distances_path, q2_people_path)

    q1_result = Q1ExactCountSplitPairSolver().solve_with_diagnostics(q1_problem)
    if not q1_result.converged:
        raise RuntimeError("Q1 V10 warm-start solver did not converge")
    q1_solution = q1_result.solution
    q1_check = check_solution(q1_problem, q1_solution, require_all_requests=True)
    if not q1_check.ok:
        raise RuntimeError("Q1 warm-start checker failed:\n" + "\n".join(q1_check.errors))

    if not set(q1_solution.assignments).issubset(q2_problem.requests):
        raise RuntimeError("Q1 request IDs are not a subset of Q2 request IDs")

    q1_metrics = evaluate_solution(q1_problem, q1_solution)
    packer = Q2WarmStartPacker(max_time_seconds=args.packing_time_limit)
    result = packer.pack(q2_problem, q1_solution)

    fast_check = check_solution(
        q2_problem, result.solution, require_all_requests=False
    )
    if not fast_check.ok:
        raise RuntimeError("Q2 B0A fast checker failed:\n" + "\n".join(fast_check.errors))

    routes_path = output_dir / "q2-routes.partial.csv"
    assignments_path = output_dir / "q2-assignments.partial.csv"
    export_q1_solution(result.solution, routes_path, assignments_path)

    validation = validate_submission(
        distances_path,
        q2_people_path,
        routes_path,
        assignments_path,
        require_all_requests=False,
    )
    if not validation.ok:
        raise RuntimeError(
            "Q2 B0A reference validator failed:\n" + "\n".join(validation.errors)
        )

    metric_dict = asdict(result.metrics)
    _assert_metric_agreement(metric_dict, validation.metrics)

    q2_kind_totals = Counter(
        classify_q2_request(request) for request in q2_problem.requests.values()
    )
    packed_ids = set(result.solution.assignments) - set(q1_solution.assignments)
    remaining_ids = sorted(set(q2_problem.requests) - set(result.solution.assignments))

    summary = {
        "stage": "Q2-B0A fixed-Q1-route zero-cost packing",
        "complete_q2_solution": False,
        "q2_request_counts": dict(sorted(q2_kind_totals.items())),
        "q1_warm_start_assigned_count": len(q1_solution.assignments),
        "q1_warm_start_metrics": asdict(q1_metrics),
        "extra_request_count": result.extra_request_count,
        "candidate_request_count": result.candidate_request_count,
        "candidate_option_count": result.candidate_option_count,
        "packed_extra_count": result.packed_extra_count,
        "packed_by_kind": result.packed_by_kind,
        "remaining_request_count": len(remaining_ids),
        "remaining_by_kind": result.remaining_by_kind,
        "total_assigned_after_b0a": len(result.solution.assignments),
        "fraction_of_all_q2_requests_assigned": (
            len(result.solution.assignments) / len(q2_problem.requests)
        ),
        "fraction_of_extra_requests_packed": (
            result.packed_extra_count / result.extra_request_count
            if result.extra_request_count
            else 0.0
        ),
        "cp_status": result.cp_status,
        "zero_incremental_aircraft_cost": {
            "aircraft_usage_minutes_delta": (
                result.metrics.total_aircraft_usage_minutes
                - result.starting_metrics.total_aircraft_usage_minutes
            ),
            "flight_count_delta": (
                result.metrics.number_of_flights
                - result.starting_metrics.number_of_flights
            ),
            "fuel_kg_delta": (
                result.metrics.total_fuel_consumption_kg
                - result.starting_metrics.total_fuel_consumption_kg
            ),
            "available_seat_km_delta": (
                result.metrics.available_seat_km
                - result.starting_metrics.available_seat_km
            ),
        },
        "b0a_partial_metrics": metric_dict,
        "reference_validator": "PASS",
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metric_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "packing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
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

    with (output_dir / "remaining_requests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "person_id",
                "origin_id",
                "destination_id",
                "request_kind",
                "candidate_option_count",
            ]
        )
        for person_id in remaining_ids:
            request = q2_problem.requests[person_id]
            writer.writerow(
                [
                    person_id,
                    request.origin_id,
                    request.destination_id,
                    classify_q2_request(request),
                    result.candidate_options_by_person.get(person_id, 0),
                ]
            )

    with (output_dir / "packed_extra.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "person_id",
                "origin_id",
                "destination_id",
                "request_kind",
                "flight_uid",
                "pickup_stop_order",
                "delivery_stop_order",
            ]
        )
        for person_id in sorted(packed_ids):
            request = q2_problem.requests[person_id]
            assignment = result.solution.assignments[person_id]
            writer.writerow(
                [
                    person_id,
                    request.origin_id,
                    request.destination_id,
                    classify_q2_request(request),
                    assignment.flight_uid,
                    assignment.pickup_index,
                    assignment.delivery_index,
                ]
            )

    trace_rows = build_onboard_trace(q2_problem, result.solution)
    with (output_dir / "onboard_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "flight_uid",
            "aircraft_type",
            "base_airport",
            "stop_index",
            "location",
            "onboard_before",
            "dropoffs",
            "pickups",
            "onboard_after",
            "capacity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"reference_validator=PASS; output_dir={output_dir}")


if __name__ == "__main__":
    main()
