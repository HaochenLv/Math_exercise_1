from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.io.load_problem import load_q1_problem, load_q2_problem
from helicopter_planner.solver.q1.count_split_recombination import Q1ExactCountSplitPairSolver
from helicopter_planner.solver.q2.warm_start import Q2WarmStartPacker, classify_q2_request


def _top(counter: Counter, limit: int = 15) -> list[dict[str, object]]:
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:limit]:
        if isinstance(key, tuple):
            row = {"key": list(key), "count": count}
        else:
            row = {"key": key, "count": count}
        rows.append(row)
    return rows


def _path_gap_category(request, solution, candidate_count: int) -> str:
    if candidate_count > 0:
        return "zero_cost_path_exists_but_capacity_blocked"

    kind = classify_q2_request(request)
    if kind == "shuttle":
        origin_seen = False
        destination_seen = False
        same_flight_wrong_order = False
        same_flight_correct_order = False
        for flight in solution.flights.values():
            route = flight.full_route()
            origin_positions = [i for i, x in enumerate(route) if x == request.origin_id]
            destination_positions = [i for i, x in enumerate(route) if x == request.destination_id]
            origin_seen = origin_seen or bool(origin_positions)
            destination_seen = destination_seen or bool(destination_positions)
            if origin_positions and destination_positions:
                if any(i < j for i in origin_positions for j in destination_positions):
                    same_flight_correct_order = True
                else:
                    same_flight_wrong_order = True
        if same_flight_correct_order:
            return "same_flight_correct_order_but_no_option"
        if same_flight_wrong_order:
            return "same_flight_wrong_order"
        if origin_seen and destination_seen:
            return "endpoints_exist_only_on_different_flights"
        if origin_seen:
            return "origin_seen_destination_missing"
        if destination_seen:
            return "destination_seen_origin_missing"
        return "neither_endpoint_seen"

    if kind == "return":
        origin_seen_any = False
        origin_seen_compatible_base = False
        for flight in solution.flights.values():
            route = flight.full_route()
            if request.origin_id not in route:
                continue
            origin_seen_any = True
            if request.destination_id == "LAND" or request.destination_id == flight.base_airport:
                origin_seen_compatible_base = True
        if origin_seen_compatible_base:
            return "origin_seen_compatible_base_but_no_option"
        if origin_seen_any:
            return "origin_seen_only_on_wrong_base"
        return "return_origin_not_seen"

    return "other"


def _components(edges: Counter[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for (a, b), count in edges.items():
        if count <= 0:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in sorted(adjacency[node]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(component))
    return sorted(components, key=lambda c: (-len(c), c))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2/residual_analysis"))
    parser.add_argument("--packing-time-limit", type=float, default=30.0)
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

    result = Q2WarmStartPacker(max_time_seconds=args.packing_time_limit).pack(
        q2_problem, q1_result.solution
    )
    solution = result.solution
    remaining_ids = sorted(set(q2_problem.requests) - set(solution.assignments))

    remaining_by_kind = Counter()
    candidate_blocked_by_kind = Counter()
    no_path_by_kind = Counter()
    path_gap_categories = Counter()

    shuttle_od: Counter[tuple[str, str]] = Counter()
    shuttle_candidate_blocked_od: Counter[tuple[str, str]] = Counter()
    shuttle_no_path_od: Counter[tuple[str, str]] = Counter()
    shuttle_out = Counter()
    shuttle_in = Counter()
    return_od: Counter[tuple[str, str]] = Counter()
    return_by_origin = Counter()

    for pid in remaining_ids:
        request = q2_problem.requests[pid]
        kind = classify_q2_request(request)
        candidate_count = result.candidate_options_by_person.get(pid, 0)
        remaining_by_kind[kind] += 1
        gap = _path_gap_category(request, solution, candidate_count)
        path_gap_categories[(kind, gap)] += 1
        if candidate_count:
            candidate_blocked_by_kind[kind] += 1
        else:
            no_path_by_kind[kind] += 1

        if kind == "shuttle":
            edge = (request.origin_id, request.destination_id)
            shuttle_od[edge] += 1
            shuttle_out[request.origin_id] += 1
            shuttle_in[request.destination_id] += 1
            if candidate_count:
                shuttle_candidate_blocked_od[edge] += 1
            else:
                shuttle_no_path_od[edge] += 1
        elif kind == "return":
            edge = (request.origin_id, request.destination_id)
            return_od[edge] += 1
            return_by_origin[request.origin_id] += 1

    facilities = sorted(set(shuttle_out) | set(shuttle_in) | set(return_by_origin))
    facility_rows: list[dict[str, object]] = []
    for facility in facilities:
        out_count = shuttle_out[facility]
        in_count = shuttle_in[facility]
        return_count = return_by_origin[facility]
        facility_rows.append(
            {
                "facility": facility,
                "shuttle_out": out_count,
                "shuttle_in": in_count,
                "residual_return": return_count,
                "net_shuttle_out_minus_in": out_count - in_count,
                "chainable_through_hub": min(out_count, in_count),
                "total_residual_activity": out_count + in_count + return_count,
            }
        )
    facility_rows.sort(
        key=lambda row: (-int(row["total_residual_activity"]), str(row["facility"]))
    )

    chain_candidates: list[dict[str, object]] = []
    outgoing_edges: dict[str, list[tuple[str, int]]] = defaultdict(list)
    incoming_edges: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (a, b), count in shuttle_od.items():
        outgoing_edges[a].append((b, count))
        incoming_edges[b].append((a, count))
    for hub in sorted(set(outgoing_edges) & set(incoming_edges)):
        for a, c1 in incoming_edges[hub]:
            for c, c2 in outgoing_edges[hub]:
                if a == c:
                    continue
                chain_candidates.append(
                    {
                        "origin": a,
                        "hub": hub,
                        "destination": c,
                        "first_leg_count": c1,
                        "second_leg_count": c2,
                        "pairable_count": min(c1, c2),
                        "combined_edge_volume": c1 + c2,
                    }
                )
    chain_candidates.sort(
        key=lambda row: (
            -int(row["pairable_count"]),
            -int(row["combined_edge_volume"]),
            str(row["origin"]),
            str(row["hub"]),
            str(row["destination"]),
        )
    )

    shuttle_od_rows: list[dict[str, object]] = []
    for (origin, destination), count in sorted(
        shuttle_od.items(), key=lambda item: (-item[1], item[0])
    ):
        shuttle_od_rows.append(
            {
                "origin": origin,
                "destination": destination,
                "count": count,
                "zero_cost_capacity_blocked": shuttle_candidate_blocked_od[(origin, destination)],
                "no_zero_cost_path": shuttle_no_path_od[(origin, destination)],
                "return_at_origin": return_by_origin[origin],
                "return_at_destination": return_by_origin[destination],
            }
        )

    components = _components(shuttle_od)
    hubs_with_both = [
        row for row in facility_rows if int(row["shuttle_in"]) and int(row["shuttle_out"])
    ]

    summary = {
        "stage": "Q2 B0A residual-demand network analysis",
        "q1_skeleton_aircraft_minutes": result.starting_metrics.total_aircraft_usage_minutes,
        "q1_skeleton_flights": result.starting_metrics.number_of_flights,
        "assigned_after_b0a": len(solution.assignments),
        "remaining_count": len(remaining_ids),
        "remaining_by_kind": dict(sorted(remaining_by_kind.items())),
        "remaining_with_zero_cost_path_but_capacity_blocked": {
            "total": sum(candidate_blocked_by_kind.values()),
            "by_kind": dict(sorted(candidate_blocked_by_kind.items())),
        },
        "remaining_without_zero_cost_path": {
            "total": sum(no_path_by_kind.values()),
            "by_kind": dict(sorted(no_path_by_kind.items())),
        },
        "path_gap_categories": [
            {"kind": kind, "category": category, "count": count}
            for (kind, category), count in sorted(
                path_gap_categories.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
            )
        ],
        "shuttle": {
            "remaining_people": sum(shuttle_od.values()),
            "unique_od_pairs": len(shuttle_od),
            "active_origins": len(shuttle_out),
            "active_destinations": len(shuttle_in),
            "undirected_connected_components": len(components),
            "component_sizes": [len(c) for c in components],
            "hubs_with_both_in_and_out": len(hubs_with_both),
            "sum_hub_chainable_counts": sum(
                int(row["chainable_through_hub"]) for row in hubs_with_both
            ),
            "two_leg_chain_candidates": len(chain_candidates),
            "top_od_pairs": shuttle_od_rows[:20],
            "top_origins": _top(shuttle_out, 15),
            "top_destinations": _top(shuttle_in, 15),
            "top_chain_hubs": hubs_with_both[:15],
            "top_two_leg_chains": chain_candidates[:20],
        },
        "return": {
            "remaining_people": sum(return_od.values()),
            "unique_od_pairs": len(return_od),
            "active_origins": len(return_by_origin),
            "top_origins": _top(return_by_origin, 15),
            "top_od_pairs": [
                {"origin": a, "destination": b, "count": count}
                for (a, b), count in sorted(return_od.items(), key=lambda item: (-item[1], item[0]))[:20]
            ],
        },
    }

    (output_dir / "residual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "residual_shuttle_od.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(shuttle_od_rows[0]) if shuttle_od_rows else ["origin", "destination", "count"])
        writer.writeheader()
        writer.writerows(shuttle_od_rows)

    with (output_dir / "residual_facility_flow.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "facility",
            "shuttle_out",
            "shuttle_in",
            "residual_return",
            "net_shuttle_out_minus_in",
            "chainable_through_hub",
            "total_residual_activity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(facility_rows)

    with (output_dir / "two_leg_chain_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "origin",
            "hub",
            "destination",
            "first_leg_count",
            "second_leg_count",
            "pairable_count",
            "combined_edge_volume",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chain_candidates)

    with (output_dir / "residual_returns.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["origin", "destination", "count"])
        for (origin, destination), count in sorted(return_od.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([origin, destination, count])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
