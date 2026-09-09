from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, REFUEL_FACILITIES, ProblemData, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution, leg_minutes
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.pair_recombination import Q2ExactPairRecombinationSolver


RequestKey = tuple[str, str]


@dataclass(frozen=True)
class RepeatedRoute:
    base_airport: str
    aircraft_type: str
    business_sequence: tuple[str, ...]
    sea_stops: tuple[str, ...]
    refuel_flags: tuple[bool, ...]
    business_stop_indices: tuple[int, ...]
    trip_minutes: int
    total_distance_km: float


@dataclass(frozen=True)
class BaselineSolve:
    status: str
    objective_minutes: int | None
    routes: tuple[dict, ...]


def _build_repeated_route(
    problem: ProblemData,
    base_airport: str,
    business_sequence: tuple[str, ...],
    aircraft_type: str,
    *,
    max_sea_landings: int = 5,
) -> RepeatedRoute | None:
    spec = problem.aircraft_types[aircraft_type]
    max_distance_since_refuel = (
        spec.tank_capacity_kg - spec.min_safe_fuel_kg
    ) / spec.fuel_rate_kg_per_km

    business_set = set(business_sequence)
    technical_refuel = tuple(
        sorted(
            facility
            for facility in REFUEL_FACILITIES
            if facility not in business_set and facility in problem.distances
        )
    )

    # elapsed, total_distance, stop_count, next_business_idx, current,
    # distance_since_refuel, path, refuel_flags, business_stop_indices
    heap: list[tuple] = [
        (0, 0.0, 0, 0, base_airport, 0.0, (), (), ())
    ]
    best_state: dict[tuple, tuple] = {}
    completed: list[tuple] = []

    while heap:
        (
            elapsed,
            total_distance,
            stop_count,
            next_business_idx,
            current,
            used_distance,
            path,
            flags,
            business_indices,
        ) = heapq.heappop(heap)

        state = (
            current,
            next_business_idx,
            round(used_distance, 9),
            stop_count,
        )
        rank = (elapsed, total_distance, path, flags)
        previous = best_state.get(state)
        if previous is not None and previous <= rank:
            continue
        best_state[state] = rank

        if next_business_idx == len(business_sequence):
            return_distance = problem.distance(current, base_airport)
            if used_distance + return_distance <= max_distance_since_refuel + 1e-9:
                completed.append(
                    (
                        elapsed + leg_minutes(return_distance, spec.speed_kmh),
                        total_distance + return_distance,
                        stop_count,
                        path,
                        flags,
                        business_indices,
                    )
                )

        if stop_count >= max_sea_landings:
            continue

        if next_business_idx < len(business_sequence):
            next_business = business_sequence[next_business_idx]
            distance = problem.distance(current, next_business)
            if used_distance + distance <= max_distance_since_refuel + 1e-9:
                arrival_elapsed = elapsed + leg_minutes(distance, spec.speed_kmh)
                refuel_options = (
                    (False, True)
                    if next_business in REFUEL_FACILITIES
                    else (False,)
                )
                for refuel in refuel_options:
                    next_used = 0.0 if refuel else used_distance + distance
                    heapq.heappush(
                        heap,
                        (
                            arrival_elapsed + (20 if refuel else 10),
                            total_distance + distance,
                            stop_count + 1,
                            next_business_idx + 1,
                            next_business,
                            next_used,
                            path + (next_business,),
                            flags + (refuel,),
                            business_indices + (stop_count + 1,),
                        ),
                    )

        # Conservative technical-stop rule: a business facility is never used
        # as a purely technical stop. This preserves the intended first-arrival
        # pickup/delivery semantics of the repeated-service pattern.
        for refuel_facility in technical_refuel:
            if refuel_facility == current:
                continue
            distance = problem.distance(current, refuel_facility)
            if used_distance + distance > max_distance_since_refuel + 1e-9:
                continue
            arrival_elapsed = elapsed + leg_minutes(distance, spec.speed_kmh)
            heapq.heappush(
                heap,
                (
                    arrival_elapsed + 20,
                    total_distance + distance,
                    stop_count + 1,
                    next_business_idx,
                    refuel_facility,
                    0.0,
                    path + (refuel_facility,),
                    flags + (True,),
                    business_indices,
                ),
            )

    if not completed:
        return None

    (
        trip_minutes,
        total_distance,
        _,
        path,
        flags,
        business_indices,
    ) = min(completed)

    return RepeatedRoute(
        base_airport=base_airport,
        aircraft_type=aircraft_type,
        business_sequence=business_sequence,
        sea_stops=tuple(path),
        refuel_flags=tuple(flags),
        business_stop_indices=tuple(business_indices),
        trip_minutes=int(trip_minutes),
        total_distance_km=float(total_distance),
    )


def _selected_group_counts(
    all_counts: Counter[RequestKey],
    base_airport: str,
    business_sequence: tuple[str, ...],
    seats: int,
) -> dict[RequestKey, int]:
    anchor = business_sequence[0]
    result: dict[RequestKey, int] = {}

    # Use LAND-flexible demand first, then fill remaining capacity with demand
    # explicitly tied to this base. This makes the comparison conservative with
    # respect to airport flexibility.
    inbound_land = min(seats, all_counts[("LAND", anchor)])
    if inbound_land:
        result[("LAND", anchor)] = inbound_land
    remaining = seats - inbound_land
    inbound_explicit = min(remaining, all_counts[(base_airport, anchor)])
    if inbound_explicit:
        result[(base_airport, anchor)] = inbound_explicit

    for left, right in zip(business_sequence, business_sequence[1:]):
        count = min(seats, all_counts[(left, right)])
        if count:
            result[(left, right)] = count

    outbound_land = min(seats, all_counts[(anchor, "LAND")])
    if outbound_land:
        result[(anchor, "LAND")] = outbound_land
    remaining = seats - outbound_land
    outbound_explicit = min(remaining, all_counts[(anchor, base_airport)])
    if outbound_explicit:
        result[(anchor, base_airport)] = outbound_explicit

    return result


def _distinct_route_baseline(
    problem: ProblemData,
    group_counts: dict[RequestKey, int],
    route_slots: int,
    route_solver: Q2ExactPairRecombinationSolver,
    *,
    time_limit_seconds: float,
) -> BaselineSolve:
    group_keys = tuple(sorted(group_counts))
    demand = tuple(group_counts[key] for key in group_keys)
    templates = route_solver._route_templates(problem, group_keys)
    if not templates:
        return BaselineSolve("INFEASIBLE", None, ())

    serviceable_by_group = [
        tuple(
            index
            for index, template in enumerate(templates)
            if group_index in template.serviceable_groups
        )
        for group_index in range(len(group_keys))
    ]
    if any(not indices for indices in serviceable_by_group):
        return BaselineSolve("INFEASIBLE", None, ())

    model = cp_model.CpModel()
    z = {}
    x = {}
    used = []
    choices = []

    for slot in range(route_slots):
        empty = model.new_bool_var(f"empty_{slot}")
        template_vars = []
        for template_index in range(len(templates)):
            var = model.new_bool_var(f"z_{slot}_{template_index}")
            z[slot, template_index] = var
            template_vars.append(var)
        model.add_exactly_one([empty, *template_vars])

        u = model.new_bool_var(f"used_{slot}")
        model.add(u + empty == 1)
        used.append(u)

        choice = model.new_int_var(0, len(templates), f"choice_{slot}")
        model.add(
            choice
            == sum(
                (template_index + 1) * z[slot, template_index]
                for template_index in range(len(templates))
            )
        )
        choices.append(choice)

        for group_index, count in enumerate(demand):
            var = model.new_int_var(0, count, f"x_{slot}_{group_index}")
            x[slot, group_index] = var
            model.add(
                var
                <= count
                * sum(
                    z[slot, template_index]
                    for template_index in serviceable_by_group[group_index]
                )
            )

        model.add(sum(x[slot, g] for g in range(len(group_keys))) >= u)

        for template_index, template in enumerate(templates):
            spec = problem.aircraft_types[template.route.aircraft_type]
            selected = z[slot, template_index]
            for leg_groups in template.leg_groups:
                if not leg_groups:
                    continue
                model.add(
                    sum(x[slot, group_index] for group_index in leg_groups)
                    <= spec.seats
                ).only_enforce_if(selected)

    for slot in range(route_slots - 1):
        model.add(used[slot] >= used[slot + 1])
        model.add(choices[slot] <= choices[slot + 1]).only_enforce_if(
            used[slot + 1]
        )

    for group_index, count in enumerate(demand):
        model.add(
            sum(x[slot, group_index] for slot in range(route_slots)) == count
        )

    objective = sum(
        template.route.trip_minutes * z[slot, template_index]
        for slot in range(route_slots)
        for template_index, template in enumerate(templates)
    )
    model.minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)

    if status == cp_model.OPTIMAL:
        status_name = "OPTIMAL"
    elif status == cp_model.FEASIBLE:
        status_name = "FEASIBLE"
    elif status == cp_model.INFEASIBLE:
        status_name = "INFEASIBLE"
    elif status == cp_model.MODEL_INVALID:
        status_name = "MODEL_INVALID"
    else:
        status_name = "UNKNOWN"

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return BaselineSolve(status_name, None, ())

    chosen_routes: list[dict] = []
    for slot in range(route_slots):
        if solver.value(used[slot]) == 0:
            continue
        template_index = next(
            index
            for index in range(len(templates))
            if solver.value(z[slot, index])
        )
        template = templates[template_index]
        allocation = {
            f"{group_keys[g][0]}->{group_keys[g][1]}": int(
                solver.value(x[slot, g])
            )
            for g in range(len(group_keys))
            if solver.value(x[slot, g]) > 0
        }
        chosen_routes.append(
            {
                "base_airport": template.route.base_airport,
                "aircraft_type": template.route.aircraft_type,
                "service_order": list(template.route.service_order),
                "trip_minutes": template.route.trip_minutes,
                "allocation": allocation,
            }
        )

    return BaselineSolve(
        status=status_name,
        objective_minutes=int(round(solver.objective_value)),
        routes=tuple(chosen_routes),
    )


def _flight_costs(
    problem: ProblemData,
    solution: Solution,
) -> dict[str, tuple[int, int, float]]:
    result: dict[str, tuple[int, int, float]] = {}
    pids_by_flight: dict[str, list[str]] = defaultdict(list)
    for pid, assignment in solution.assignments.items():
        pids_by_flight[assignment.flight_uid].append(pid)

    for uid, pids in pids_by_flight.items():
        assignments = {pid: solution.assignments[pid] for pid in pids}
        metrics = evaluate_solution(
            problem,
            Solution(
                flights={uid: solution.flights[uid]},
                assignments=assignments,
            ),
        )
        result[uid] = (
            metrics.total_aircraft_usage_minutes,
            metrics.total_passenger_travel_minutes,
            metrics.total_fuel_consumption_kg,
        )
    return result


def _select_person_ids(
    problem: ProblemData,
    group_counts: dict[RequestKey, int],
    solution: Solution,
    flight_costs: dict[str, tuple[int, int, float]],
) -> tuple[str, ...]:
    by_key: dict[RequestKey, list[str]] = defaultdict(list)
    for pid, request in problem.requests.items():
        by_key[(request.origin_id, request.destination_id)].append(pid)

    uid_by_pid = {
        pid: assignment.flight_uid for pid, assignment in solution.assignments.items()
    }

    selected: list[str] = []
    for key, count in sorted(group_counts.items()):
        grouped: dict[str, list[str]] = defaultdict(list)
        for pid in by_key[key]:
            grouped[uid_by_pid[pid]].append(pid)

        # Prefer concentrated demand on expensive flights so the reported
        # affected region is compact and useful for the next exact-rebuild test.
        ranked_groups = sorted(
            grouped.items(),
            key=lambda item: (
                -len(item[1]),
                -flight_costs[item[0]][0],
                item[0],
            ),
        )
        remaining = count
        for _, pids in ranked_groups:
            take = min(remaining, len(pids))
            selected.extend(sorted(pids)[:take])
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise AssertionError(f"not enough people for {key}: missing {remaining}")

    return tuple(sorted(selected))


def _affected_region(
    selected_pids: tuple[str, ...],
    solution: Solution,
    flight_costs: dict[str, tuple[int, int, float]],
) -> dict:
    selected = set(selected_pids)
    pids_by_flight: dict[str, set[str]] = defaultdict(set)
    for pid, assignment in solution.assignments.items():
        pids_by_flight[assignment.flight_uid].add(pid)

    affected = sorted(
        {solution.assignments[pid].flight_uid for pid in selected_pids}
    )
    fully_covered = sorted(
        uid for uid in affected if pids_by_flight[uid] <= selected
    )
    return {
        "selected_person_count": len(selected_pids),
        "affected_flights": affected,
        "affected_flight_count": len(affected),
        "affected_aircraft_minutes": sum(flight_costs[uid][0] for uid in affected),
        "fully_covered_flights": fully_covered,
        "fully_covered_flight_count": len(fully_covered),
        "fully_covered_aircraft_minutes": sum(
            flight_costs[uid][0] for uid in fully_covered
        ),
    }


def _request_counts(problem: ProblemData) -> Counter[RequestKey]:
    return Counter(
        (request.origin_id, request.destination_id)
        for request in problem.requests.values()
    )


def _pattern_candidates(
    all_counts: Counter[RequestKey],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    sea_edges = sorted(
        (origin, destination)
        for (origin, destination), count in all_counts.items()
        if count > 0
        and origin.startswith("F")
        and destination.startswith("F")
        and origin != destination
    )

    patterns: list[tuple[str, tuple[str, ...]]] = []
    for origin, destination in sea_edges:
        patterns.append(("two_node_return", (origin, destination, origin)))

    adjacency: dict[str, set[str]] = defaultdict(set)
    for origin, destination in sea_edges:
        adjacency[origin].add(destination)

    cycles: set[tuple[str, str, str]] = set()
    for u, v in sea_edges:
        for w in adjacency.get(v, ()):
            if len({u, v, w}) != 3:
                continue
            if u in adjacency.get(w, ()):
                cycles.add((u, v, w))

    for u, v, w in sorted(cycles):
        patterns.append(("three_node_cycle", (u, v, w, u)))

    return tuple(patterns)


def _json_group_counts(group_counts: dict[RequestKey, int]) -> dict[str, int]:
    return {
        f"{origin}->{destination}": count
        for (origin, destination), count in sorted(group_counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q2-routes", type=Path, required=True)
    parser.add_argument("--q2-assignments", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q2/c1_repeated_service_diagnostic"),
    )
    parser.add_argument("--baseline-time-limit", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances_path = ROOT / "data/raw/distances.csv"
    q2_people_path = ROOT / "data/raw/peopleQ2.csv"
    problem = load_q2_problem(distances_path, q2_people_path)
    anchor = load_exported_solution(
        resolve(args.q2_routes),
        resolve(args.q2_assignments),
        uid_prefix="q2-b3b-anchor",
    )
    anchor_check = check_solution(problem, anchor, require_all_requests=True)
    if not anchor_check.ok:
        raise RuntimeError(
            "frozen Q2 anchor invalid:\n" + "\n".join(anchor_check.errors)
        )
    anchor_metrics = evaluate_solution(problem, anchor)
    if (
        anchor_metrics.total_aircraft_usage_minutes != 19126
        or anchor_metrics.number_of_flights != 108
    ):
        raise RuntimeError(
            "unexpected Q2 anchor: "
            f"{anchor_metrics.total_aircraft_usage_minutes} min / "
            f"{anchor_metrics.number_of_flights} flights"
        )

    all_counts = _request_counts(problem)
    patterns = _pattern_candidates(all_counts)
    flight_costs = _flight_costs(problem, anchor)
    route_solver = Q2ExactPairRecombinationSolver(
        cp_time_limit_seconds=args.baseline_time_limit
    )

    rows: list[dict] = []
    baseline_cache: dict[
        tuple[tuple[tuple[str, str, int], ...], int], BaselineSolve
    ] = {}

    for pattern_type, business_sequence in patterns:
        route_slots = 2 if pattern_type == "two_node_return" else 3
        for base_airport in sorted(AIRPORTS):
            for aircraft_type in sorted(problem.aircraft_types):
                spec = problem.aircraft_types[aircraft_type]
                group_counts = _selected_group_counts(
                    all_counts,
                    base_airport,
                    business_sequence,
                    spec.seats,
                )
                internal_count = sum(
                    group_counts.get((left, right), 0)
                    for left, right in zip(
                        business_sequence, business_sequence[1:]
                    )
                )
                if internal_count == 0:
                    continue

                repeated = _build_repeated_route(
                    problem,
                    base_airport,
                    business_sequence,
                    aircraft_type,
                )
                if repeated is None:
                    continue

                cache_key = (
                    tuple(
                        sorted(
                            (origin, destination, count)
                            for (origin, destination), count in group_counts.items()
                        )
                    ),
                    route_slots,
                )
                baseline = baseline_cache.get(cache_key)
                if baseline is None:
                    baseline = _distinct_route_baseline(
                        problem,
                        group_counts,
                        route_slots,
                        route_solver,
                        time_limit_seconds=args.baseline_time_limit,
                    )
                    baseline_cache[cache_key] = baseline

                structural_gap = (
                    None
                    if baseline.objective_minutes is None
                    else baseline.objective_minutes - repeated.trip_minutes
                )

                selected_pids = _select_person_ids(
                    problem, group_counts, anchor, flight_costs
                )
                region = _affected_region(selected_pids, anchor, flight_costs)
                direct_whole_flight_savings = (
                    region["fully_covered_aircraft_minutes"] - repeated.trip_minutes
                )

                rows.append(
                    {
                        "pattern_type": pattern_type,
                        "business_sequence": "->".join(business_sequence),
                        "base_airport": base_airport,
                        "aircraft_type": aircraft_type,
                        "seats": spec.seats,
                        "trip_minutes": repeated.trip_minutes,
                        "sea_landings": len(repeated.sea_stops),
                        "full_sea_path": "->".join(repeated.sea_stops),
                        "refuel_flags": ";".join(
                            "1" if flag else "0" for flag in repeated.refuel_flags
                        ),
                        "total_distance_km": repeated.total_distance_km,
                        "fuel_kg": (
                            repeated.total_distance_km * spec.fuel_rate_kg_per_km
                        ),
                        "transported_people": sum(group_counts.values()),
                        "group_counts": json.dumps(
                            _json_group_counts(group_counts),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "baseline_status": baseline.status,
                        "distinct_route_baseline_minutes": baseline.objective_minutes,
                        "structural_gap_minutes": structural_gap,
                        "baseline_routes": json.dumps(
                            list(baseline.routes),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "selected_person_count": region["selected_person_count"],
                        "affected_flight_count": region["affected_flight_count"],
                        "affected_aircraft_minutes": region[
                            "affected_aircraft_minutes"
                        ],
                        "affected_flights": ";".join(region["affected_flights"]),
                        "fully_covered_flight_count": region[
                            "fully_covered_flight_count"
                        ],
                        "fully_covered_aircraft_minutes": region[
                            "fully_covered_aircraft_minutes"
                        ],
                        "fully_covered_flights": ";".join(
                            region["fully_covered_flights"]
                        ),
                        "direct_whole_flight_savings_minutes": (
                            direct_whole_flight_savings
                        ),
                    }
                )

    rows.sort(
        key=lambda row: (
            -(
                row["structural_gap_minutes"]
                if row["structural_gap_minutes"] is not None
                else -10**9
            ),
            row["trip_minutes"],
            row["pattern_type"],
            row["business_sequence"],
            row["base_airport"],
            row["aircraft_type"],
        )
    )

    top_rows = rows[: args.top_k]
    with (output_dir / "top_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(top_rows[0]) if top_rows else []
        )
        writer.writeheader()
        writer.writerows(top_rows)

    positive = [
        row
        for row in rows
        if row["baseline_status"] == "OPTIMAL"
        and row["structural_gap_minutes"] is not None
        and row["structural_gap_minutes"] > 0
    ]

    # Greedy OD-count-disjoint diagnostic. This is not a guaranteed global
    # improvement over the anchor; it measures omitted repeated-route advantage
    # across demand-disjoint local subproblems.
    remaining_counts = Counter(all_counts)
    greedy_structural = []
    greedy_gap = 0
    for row in positive:
        parsed = json.loads(row["group_counts"])
        needed = {}
        for text_key, count in parsed.items():
            origin, destination = text_key.split("->", 1)
            needed[(origin, destination)] = int(count)
        if all(remaining_counts[key] >= count for key, count in needed.items()):
            greedy_structural.append(
                {
                    "pattern_type": row["pattern_type"],
                    "business_sequence": row["business_sequence"],
                    "base_airport": row["base_airport"],
                    "aircraft_type": row["aircraft_type"],
                    "structural_gap_minutes": row["structural_gap_minutes"],
                    "group_counts": parsed,
                }
            )
            greedy_gap += int(row["structural_gap_minutes"])
            for key, count in needed.items():
                remaining_counts[key] -= count

    direct_positive = [
        row for row in rows if row["direct_whole_flight_savings_minutes"] > 0
    ]
    used_flights: set[str] = set()
    greedy_direct = []
    greedy_direct_savings = 0
    for row in sorted(
        direct_positive,
        key=lambda item: (
            -item["direct_whole_flight_savings_minutes"],
            item["trip_minutes"],
        ),
    ):
        covered = {uid for uid in row["fully_covered_flights"].split(";") if uid}
        if not covered or covered & used_flights:
            continue
        used_flights |= covered
        greedy_direct.append(
            {
                "business_sequence": row["business_sequence"],
                "base_airport": row["base_airport"],
                "aircraft_type": row["aircraft_type"],
                "covered_flights": sorted(covered),
                "direct_savings_minutes": row[
                    "direct_whole_flight_savings_minutes"
                ],
            }
        )
        greedy_direct_savings += int(
            row["direct_whole_flight_savings_minutes"]
        )

    f017_f014 = [
        row
        for row in rows
        if row["business_sequence"] == "F017->F014->F017"
    ]
    f017_f014.sort(
        key=lambda row: (
            -(
                row["structural_gap_minutes"]
                if row["structural_gap_minutes"] is not None
                else -10**9
            ),
            row["trip_minutes"],
        )
    )

    summary = {
        "stage": "Q2-C1 repeated-service loop diagnostic",
        "anchor": {
            "total_aircraft_usage_minutes": (
                anchor_metrics.total_aircraft_usage_minutes
            ),
            "number_of_flights": anchor_metrics.number_of_flights,
            "seat_utilization": anchor_metrics.seat_utilization,
        },
        "patterns_scanned": len(patterns),
        "route_configurations_evaluated": len(rows),
        "positive_structural_gap_candidates": len(positive),
        "best_structural_gap_minutes": (
            positive[0]["structural_gap_minutes"] if positive else 0
        ),
        "best_candidate": positive[0] if positive else None,
        "f017_f014_best_candidate": f017_f014[0] if f017_f014 else None,
        "greedy_demand_disjoint_structural_gap_minutes": greedy_gap,
        "greedy_demand_disjoint_candidate_count": len(greedy_structural),
        "greedy_demand_disjoint_candidates": greedy_structural,
        "direct_whole_flight_positive_candidates": len(direct_positive),
        "greedy_direct_whole_flight_savings_minutes": greedy_direct_savings,
        "greedy_direct_whole_flight_replacements": greedy_direct,
        "interpretation": {
            "structural_gap": (
                "Difference between the exact best distinct-service local route "
                "family and one repeated-service loop for the same selected OD "
                "counts. It measures an omitted route-family advantage, not "
                "guaranteed global savings versus the anchor."
            ),
            "direct_whole_flight_savings": (
                "Strictly conservative immediate replacement test: only credits "
                "current flights whose complete passenger sets are contained in "
                "the diagnostic loop demand."
            ),
        },
        "threshold_flags": {
            "structural_gap_gt_500": greedy_gap > 500,
            "structural_gap_gt_1000": greedy_gap > 1000,
            "direct_savings_gt_0": greedy_direct_savings > 0,
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "all_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else []
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
