from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_q2_c5_regional_lns import Q2RegionalLNSSolver
from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution
from helicopter_planner.io.export_solution import export_q1_solution
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.pair_recombination import Q2ExactPairRecombinationSolver
from helicopter_planner.solver.q2.tail_elimination import Q2PassengerSetRouteOptimizer
from reference_validator import validate_submission


@dataclass(frozen=True)
class PoolTemplate:
    route: object
    serviceable_groups: tuple[int, ...]
    leg_groups: tuple[tuple[int, ...], ...]


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


def route_key(route):
    return (route.base_airport, route.aircraft_type, route.service_order)


def build_global_template(group_keys, route):
    leg_count = len(route.sea_stops) + 1
    serviceable = []
    intervals = {}
    for gi, (origin, destination) in enumerate(group_keys):
        if origin == "LAND":
            pickup = 0
        elif origin in AIRPORTS:
            if origin != route.base_airport:
                continue
            pickup = 0
        elif origin.startswith("F"):
            if origin not in route.service_order:
                continue
            pickup = route.service_stop_index(origin)
        else:
            continue

        if destination == "LAND":
            delivery = leg_count
        elif destination in AIRPORTS:
            if destination != route.base_airport:
                continue
            delivery = leg_count
        elif destination.startswith("F"):
            if destination not in route.service_order:
                continue
            delivery = route.service_stop_index(destination)
        else:
            continue

        if pickup >= delivery:
            continue
        serviceable.append(gi)
        intervals[gi] = (pickup, delivery)

    if not serviceable:
        return None
    # Every business service facility in the route should be useful to at least
    # one globally serviceable OD. Technical refuel stops are not in service_order.
    useful_facilities = {
        loc
        for gi in serviceable
        for loc in group_keys[gi]
        if loc.startswith("F")
    }
    if not set(route.service_order).issubset(useful_facilities):
        return None

    leg_groups = []
    for leg in range(leg_count):
        leg_groups.append(tuple(gi for gi in serviceable if intervals[gi][0] <= leg < intervals[gi][1]))
    return PoolTemplate(route=route, serviceable_groups=tuple(serviceable), leg_groups=tuple(leg_groups))


def collect_pool(problem, anchor, *, cluster_budget: int, max_routes: int, seed: int):
    route_optimizer = Q2PassengerSetRouteOptimizer()
    generator = Q2ExactPairRecombinationSolver(route_optimizer=route_optimizer)
    # identity-free summaries via C5 helper
    helper = Q2RegionalLNSSolver(cluster_size=5, neighbor_count=9, max_union_facilities=8,
                                 max_group_keys=14, max_clusters_per_iteration=cluster_budget,
                                 cp_time_limit_seconds=1.0, max_iterations=1,
                                 selection_mode="mixed", seed=seed)
    summaries = helper._summaries(problem, anchor)

    routes = {}
    anchor_keys = set()
    # Guaranteed feasible backbone: exact optimized route for every current flight profile.
    for uid, summary in summaries.items():
        opt = route_optimizer.optimize(problem, summary.profile)
        if opt is None:
            raise RuntimeError(f"cannot optimize anchor flight {uid}")
        key = route_key(opt.route)
        routes[key] = opt.route
        anchor_keys.add(key)

    selectors = [
        (5, 9, 7, 12, "shared", seed + 11),
        (5, 8, 7, 12, "close", seed + 23),
        (6, 10, 8, 14, "cost", seed + 37),
        (6, 9, 8, 14, "mixed", seed + 53),
    ]
    cluster_count = 0
    generated_templates = 0
    tic = time.monotonic()
    for size, neigh, union_cap, group_cap, mode, local_seed in selectors:
        selector = Q2RegionalLNSSolver(
            cluster_size=size,
            neighbor_count=neigh,
            max_union_facilities=union_cap,
            max_group_keys=group_cap,
            max_clusters_per_iteration=cluster_budget,
            cp_time_limit_seconds=1.0,
            max_iterations=1,
            selection_mode=mode,
            seed=local_seed,
        )
        clusters = selector._candidate_clusters(problem, summaries)
        for cluster in clusters:
            cluster_count += 1
            keys = tuple(sorted(selector._cluster_group_keys(summaries, cluster)))
            for template in generator._route_templates(problem, keys):
                generated_templates += 1
                routes.setdefault(route_key(template.route), template.route)
        print(f"[C6] pool selector={mode}{size} clusters={len(clusters)} unique_routes={len(routes)} elapsed={time.monotonic()-tic:.1f}s", flush=True)

    # Convert to global serviceability templates before pruning.
    all_group_keys = tuple(sorted({
        (req.origin_id, req.destination_id) for req in problem.requests.values()
    }))
    candidates = []
    for key, route in routes.items():
        gt = build_global_template(all_group_keys, route)
        if gt is None:
            continue
        seats = problem.aircraft_types[route.aircraft_type].seats
        efficiency = route.trip_minutes / max(1, seats * len(gt.serviceable_groups))
        rank = (0 if key in anchor_keys else 1, efficiency, route.trip_minutes, -len(gt.serviceable_groups), key)
        candidates.append((rank, key, gt))
    candidates.sort(key=lambda x: x[0])

    # Keep all anchor backbone columns, then the best diverse generated columns.
    kept = candidates[:max_routes]
    templates = [x[2] for x in kept]
    kept_keys = {x[1] for x in kept}
    missing_anchor = anchor_keys - kept_keys
    if missing_anchor:
        by_key = {key: gt for _, key, gt in candidates}
        for key in sorted(missing_anchor):
            if key in by_key:
                templates.append(by_key[key])
    print(f"[C6] pool done raw_unique_routes={len(routes)} global_templates={len(candidates)} kept={len(templates)} generated_local_templates={generated_templates}", flush=True)
    return all_group_keys, templates, route_optimizer


def solve_master(problem, anchor, group_keys, templates, route_optimizer, *, time_limit: float, max_copies: int):
    people_by_key = defaultdict(list)
    for pid, req in problem.requests.items():
        people_by_key[(req.origin_id, req.destination_id)].append(pid)
    for ids in people_by_key.values():
        ids.sort()
    demand = tuple(len(people_by_key[key]) for key in group_keys)

    serviceable_by_group = [[] for _ in group_keys]
    for ti, template in enumerate(templates):
        for gi in template.serviceable_groups:
            serviceable_by_group[gi].append(ti)
    missing = [group_keys[i] for i, items in enumerate(serviceable_by_group) if not items]
    if missing:
        raise RuntimeError(f"route pool cannot serve {len(missing)} OD groups: {missing[:8]}")

    model = cp_model.CpModel()
    y = {}
    x = {}
    for ti, template in enumerate(templates):
        route = template.route
        y[ti] = model.new_int_var(0, max_copies, f"y_{ti}")
        spec = problem.aircraft_types[route.aircraft_type]
        for gi in template.serviceable_groups:
            x[ti, gi] = model.new_int_var(0, demand[gi], f"x_{ti}_{gi}")
            model.add(x[ti, gi] <= demand[gi] * y[ti])
        # Aggregate interval capacity across identical copies. For interval
        # demands this is a valid copy-capacity relaxation that is later packed
        # exactly into individual sorties.
        for leg_groups in template.leg_groups:
            if leg_groups:
                model.add(sum(x[ti, gi] for gi in leg_groups) <= spec.seats * y[ti])
        model.add(sum(x[ti, gi] for gi in template.serviceable_groups) >= y[ti])

    for gi, count in enumerate(demand):
        model.add(sum(x[ti, gi] for ti in serviceable_by_group[gi]) == count)

    aircraft = sum(template.route.trip_minutes * y[ti] for ti, template in enumerate(templates))
    model.minimize(aircraft)

    # Warm-start hints from the anchor, mapped onto exact optimized route keys.
    key_to_ti = {route_key(t.route): i for i, t in enumerate(templates)}
    hint_y = Counter()
    hint_x = Counter()
    for uid in sorted(anchor.flights):
        ids = sorted(pid for pid, a in anchor.assignments.items() if a.flight_uid == uid)
        if not ids:
            continue
        profile = route_optimizer.profile(problem, ids)
        opt = route_optimizer.optimize(problem, profile)
        if opt is None:
            continue
        ti = key_to_ti.get(route_key(opt.route))
        if ti is None:
            continue
        hint_y[ti] += 1
        counts = Counter((problem.requests[pid].origin_id, problem.requests[pid].destination_id) for pid in ids)
        gi_by_key = {key: gi for gi, key in enumerate(group_keys)}
        for key, n in counts.items():
            hint_x[ti, gi_by_key[key]] += n
    for ti, value in hint_y.items():
        model.add_hint(y[ti], value)
    for key, value in hint_x.items():
        if key in x:
            model.add_hint(x[key], value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 0
    solver.parameters.log_search_progress = False
    print(f"[C6] master templates={len(templates)} x_vars={len(x)} groups={len(group_keys)} time_limit={time_limit}s", flush=True)
    tic = time.monotonic()
    status = solver.solve(model)
    elapsed = time.monotonic() - tic
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"global master failed: {status_name}")
    objective = int(round(solver.objective_value))
    bound = int(math.floor(solver.best_objective_bound + 1e-9))
    print(f"[C6] master status={status_name} objective={objective} bound={bound} elapsed={elapsed:.1f}s", flush=True)

    selected = []
    for ti, template in enumerate(templates):
        copies = solver.value(y[ti])
        if copies <= 0:
            continue
        counts = {gi: solver.value(x[ti, gi]) for gi in template.serviceable_groups if solver.value(x[ti, gi]) > 0}
        selected.append((ti, template, copies, counts))
    return selected, objective, bound, status_name, elapsed, people_by_key


def exact_pack_template(problem, template, copies, counts, *, time_limit=5.0):
    if copies == 1:
        return [dict(counts)]
    groups = sorted(counts)
    model = cp_model.CpModel()
    z = {}
    for c in range(copies):
        for gi in groups:
            z[c, gi] = model.new_int_var(0, counts[gi], f"z_{c}_{gi}")
    for gi in groups:
        model.add(sum(z[c, gi] for c in range(copies)) == counts[gi])
    spec = problem.aircraft_types[template.route.aircraft_type]
    for c in range(copies):
        for leg_groups in template.leg_groups:
            active = [gi for gi in leg_groups if gi in counts]
            if active:
                model.add(sum(z[c, gi] for gi in active) <= spec.seats)
        model.add(sum(z[c, gi] for gi in groups) >= 1)
    # symmetry: total passenger count nonincreasing across copies
    loads = []
    for c in range(copies):
        load = model.new_int_var(1, sum(counts.values()), f"load_{c}")
        model.add(load == sum(z[c, gi] for gi in groups))
        loads.append(load)
    for c in range(copies - 1):
        model.add(loads[c] >= loads[c + 1])
    cp = cp_model.CpSolver()
    cp.parameters.max_time_in_seconds = time_limit
    cp.parameters.num_search_workers = 1
    status = cp.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [{gi: cp.value(z[c, gi]) for gi in groups if cp.value(z[c, gi]) > 0} for c in range(copies)]


def reconstruct(problem, group_keys, selected, people_by_key, route_optimizer):
    cursors = {key: 0 for key in group_keys}
    solution = Solution(flights={}, assignments={})
    flight_index = 0
    for _, template, copies, counts in selected:
        packed = exact_pack_template(problem, template, copies, counts)
        if packed is None:
            raise RuntimeError(f"cannot exactly pack selected template {route_key(template.route)} x{copies}")
        for copy_counts in packed:
            ids = []
            for gi, n in copy_counts.items():
                key = group_keys[gi]
                start = cursors[key]
                ids.extend(people_by_key[key][start:start+n])
                cursors[key] += n
            profile = route_optimizer.profile(problem, ids)
            optimized = route_optimizer.optimize(problem, profile)
            if optimized is None:
                raise RuntimeError("reconstructed passenger set became infeasible")
            uid = f"q2-c6-{flight_index:04d}"
            flight_index += 1
            route_optimizer.install(problem, solution, uid, ids, optimized)
    for key in group_keys:
        if cursors[key] != len(people_by_key[key]):
            raise RuntimeError(f"reconstruction did not cover {key}: {cursors[key]}/{len(people_by_key[key])}")
    return solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-routes", type=Path, required=True)
    ap.add_argument("--anchor-assignments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cluster-budget", type=int, default=24)
    ap.add_argument("--max-routes", type=int, default=3000)
    ap.add_argument("--master-time-limit", type=float, default=90.0)
    ap.add_argument("--max-copies", type=int, default=25)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--expected-anchor-minutes", type=int, default=18606)
    ap.add_argument("--expected-anchor-flights", type=int, default=104)
    args = ap.parse_args()

    def resolve(path):
        return path if path.is_absolute() else ROOT / path

    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    distances = ROOT / "data/raw/distances.csv"
    people = ROOT / "data/raw/peopleQ2.csv"
    problem = load_q2_problem(distances, people)
    anchor = load_exported_solution(resolve(args.anchor_routes), resolve(args.anchor_assignments), uid_prefix="q2-c6-anchor")
    ck = check_solution(problem, anchor, require_all_requests=True)
    if not ck.ok:
        raise RuntimeError("anchor invalid:\n" + "\n".join(ck.errors))
    am = evaluate_solution(problem, anchor)
    if am.total_aircraft_usage_minutes != args.expected_anchor_minutes or am.number_of_flights != args.expected_anchor_flights:
        raise RuntimeError(f"unexpected anchor {am.total_aircraft_usage_minutes}/{am.number_of_flights}")

    group_keys, templates, route_optimizer = collect_pool(
        problem, anchor, cluster_budget=args.cluster_budget, max_routes=args.max_routes, seed=args.seed
    )
    selected, master_obj, bound, status_name, elapsed, people_by_key = solve_master(
        problem, anchor, group_keys, templates, route_optimizer,
        time_limit=args.master_time_limit, max_copies=args.max_copies,
    )
    solution = reconstruct(problem, group_keys, selected, people_by_key, route_optimizer)
    ck = check_solution(problem, solution, require_all_requests=True)
    if not ck.ok:
        raise RuntimeError("C6 reconstructed solution invalid:\n" + "\n".join(ck.errors))
    metrics = evaluate_solution(problem, solution)

    routes = out / "q2-routes.csv"
    assignments = out / "q2-assignments.csv"
    export_q1_solution(solution, routes, assignments)
    ref = validate_submission(distances, people, routes, assignments, require_all_requests=True)
    if not ref.ok:
        raise RuntimeError("reference validator failed:\n" + "\n".join(ref.errors))

    summary = {
        "stage": "Q2-C6 global route-pool set-partitioning master",
        "anchor": metrics_dict(am),
        "result": metrics_dict(metrics),
        "aircraft_savings_minutes": am.total_aircraft_usage_minutes - metrics.total_aircraft_usage_minutes,
        "flight_count_change": metrics.number_of_flights - am.number_of_flights,
        "route_pool_size": len(templates),
        "selected_route_types": len(selected),
        "master_objective_minutes": master_obj,
        "master_best_bound_minutes": bound,
        "master_status": status_name,
        "master_elapsed_seconds": elapsed,
        "cluster_budget": args.cluster_budget,
        "max_routes": args.max_routes,
        "master_time_limit_seconds": args.master_time_limit,
        "reference_validator": "PASS",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps({"ok": ref.ok, "errors": list(ref.errors), "metrics": ref.metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
