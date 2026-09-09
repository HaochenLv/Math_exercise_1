from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Assignment, FlightPlan, SeaStop, Solution
from helicopter_planner.evaluation.metrics import evaluate_solution, leg_minutes, stop_minutes
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.tail_elimination import Q2PassengerSetRouteOptimizer
from reference_validator import validate_submission

TARGET_KEYS = (("T2", 25), ("T2", 26), ("T3", 41), ("T3", 57))
U = "F044"
V = "F037"


@dataclass(frozen=True)
class RepeatRoute:
    base: str
    aircraft_type: str
    sea_path: tuple[str, ...]
    refuel_flags: tuple[bool, ...]
    trip_minutes: int
    distance_km: float


def _person_ids(solution: Solution, uid: str) -> list[str]:
    return sorted(pid for pid, a in solution.assignments.items() if a.flight_uid == uid)


def _find_uid(solution: Solution, aircraft_type: str, flight_no: int) -> str:
    suffix = f"-{aircraft_type}-{flight_no:04d}"
    matches = [uid for uid in solution.flights if uid.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one UID for {aircraft_type}/{flight_no}, got {matches}")
    return matches[0]


def _route_cost(problem, solution: Solution, uid: str) -> int:
    assignments = {pid: a for pid, a in solution.assignments.items() if a.flight_uid == uid}
    return evaluate_solution(problem, Solution(flights={uid: solution.flights[uid]}, assignments=assignments)).total_aircraft_usage_minutes


def _build_repeat_route(problem, base: str, aircraft_type: str) -> RepeatRoute | None:
    # Purposefully narrow C2 route family: A -> F044 -> F037 -> F044 -> A.
    # Enumerate refuelling decisions at each F044 visit. F037 is not a refuel site.
    spec = problem.aircraft_types[aircraft_type]
    path = (U, V, U)
    best = None
    for flags in ((False, False, False), (True, False, False), (False, False, True), (True, False, True)):
        fuel = spec.tank_capacity_kg
        elapsed = 0
        dist_total = 0.0
        current = base
        feasible = True
        for i, nxt in enumerate(path):
            d = problem.distance(current, nxt)
            fuel -= d * spec.fuel_rate_kg_per_km
            if fuel + 1e-9 < spec.min_safe_fuel_kg:
                feasible = False
                break
            elapsed += leg_minutes(d, spec.speed_kmh)
            dist_total += d
            if flags[i]:
                if nxt != U:
                    feasible = False
                    break
                elapsed += 20
                fuel = spec.tank_capacity_kg
            else:
                elapsed += 10
            current = nxt
        if not feasible:
            continue
        d = problem.distance(current, base)
        fuel -= d * spec.fuel_rate_kg_per_km
        if fuel + 1e-9 < spec.min_safe_fuel_kg:
            continue
        elapsed += leg_minutes(d, spec.speed_kmh)
        dist_total += d
        key = (elapsed, dist_total, flags)
        if best is None or key < best[0]:
            best = (key, RepeatRoute(base, aircraft_type, path, flags, elapsed, dist_total))
    return None if best is None else best[1]


def _repeat_group_interval(origin: str, destination: str, base: str) -> tuple[int, int] | None:
    # Stops: 0=base, 1=U(first), 2=V, 3=U(second), 4=base.
    # We choose the most useful legal occurrence for repeated U.
    if origin == "LAND":
        o = 0
    elif origin in AIRPORTS:
        if origin != base:
            return None
        o = 0
    elif origin == U:
        # If going to V, use first U; otherwise use second U when returning landward.
        o = 1 if destination == V else 3
    elif origin == V:
        o = 2
    else:
        return None

    if destination == "LAND":
        d = 4
    elif destination in AIRPORTS:
        if destination != base:
            return None
        d = 4
    elif destination == U:
        # From V use second U; from base use first U.
        d = 3 if origin == V else 1
    elif destination == V:
        d = 2
    else:
        return None
    return (o, d) if o < d else None


def _install_repeat(solution: Solution, uid: str, route: RepeatRoute, people, problem) -> None:
    solution.flights[uid] = FlightPlan(
        flight_uid=uid,
        base_airport=route.base,
        aircraft_type=route.aircraft_type,
        sea_stops=[SeaStop(f, r) for f, r in zip(route.sea_path, route.refuel_flags, strict=True)],
    )
    for pid in people:
        req = problem.requests[pid]
        interval = _repeat_group_interval(req.origin_id, req.destination_id, route.base)
        if interval is None:
            raise AssertionError(f"repeat route cannot serve {pid}: {req.origin_id}->{req.destination_id}")
        solution.assignments[pid] = Assignment(pid, uid, interval[0], interval[1])


def _export(solution: Solution, routes_path: Path, assignments_path: Path) -> None:
    # Renumber per aircraft type exactly as official Q2 format requires.
    by_type = defaultdict(list)
    for uid, flight in solution.flights.items():
        by_type[flight.aircraft_type].append(uid)
    number = {}
    for t in sorted(by_type):
        for i, uid in enumerate(sorted(by_type[t]), start=1):
            number[uid] = i

    with routes_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["aircraft_type", "flight_no", "stop_order", "facility_id", "refuel"])
        for uid in sorted(solution.flights, key=lambda x: (solution.flights[x].aircraft_type, number[x])):
            f = solution.flights[uid]
            no = number[uid]
            w.writerow([f.aircraft_type, no, 0, f.base_airport, 0])
            for i, stop in enumerate(f.sea_stops, start=1):
                w.writerow([f.aircraft_type, no, i, stop.facility_id, int(stop.refuel)])
            w.writerow([f.aircraft_type, no, len(f.sea_stops) + 1, f.base_airport, 0])

    with assignments_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["person_id", "aircraft_type", "flight_no", "pickup_stop_order", "delivery_stop_order"])
        for pid in sorted(solution.assignments):
            a = solution.assignments[pid]
            f = solution.flights[a.flight_uid]
            w.writerow([pid, f.aircraft_type, number[a.flight_uid], a.pickup_index, a.delivery_index])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-routes", type=Path, required=True)
    ap.add_argument("--anchor-assignments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/q2/c2_f044_f037"))
    args = ap.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    problem = load_q2_problem(ROOT / "data/raw/distances.csv", ROOT / "data/raw/peopleQ2.csv")
    anchor = load_exported_solution(args.anchor_routes, args.anchor_assignments, uid_prefix="q2-c2-anchor")
    chk = check_solution(problem, anchor, require_all_requests=True)
    if not chk.ok:
        raise RuntimeError("anchor invalid: " + "\n".join(chk.errors))
    anchor_metrics = evaluate_solution(problem, anchor)
    if anchor_metrics.total_aircraft_usage_minutes != 19126:
        raise RuntimeError(f"unexpected anchor minutes {anchor_metrics.total_aircraft_usage_minutes}")

    target_uids = [_find_uid(anchor, *key) for key in TARGET_KEYS]
    target_people = {uid: _person_ids(anchor, uid) for uid in target_uids}
    old_region_minutes = sum(_route_cost(problem, anchor, uid) for uid in target_uids)

    # Candidate repeat routes over all legal bases/types. We search passenger extraction counts
    # from the four frozen flights, then exactly reoptimize each residual passenger set.
    repeat_routes = [
        r for base in AIRPORTS for t in ("T1", "T2", "T3")
        if (r := _build_repeat_route(problem, base, t)) is not None
    ]
    optimizer = Q2PassengerSetRouteOptimizer()

    # Eligible passengers by source flight for the repeated loop.
    elig = {}
    for uid in target_uids:
        ids = []
        for pid in target_people[uid]:
            req = problem.requests[pid]
            # base-specific compatibility checked later; include only U/V/land endpoints here.
            if req.origin_id in {"LAND", U, V, "A01", "A02", "A03"} and req.destination_id in {"LAND", U, V, "A01", "A02", "A03"}:
                ids.append(pid)
        elig[uid] = ids

    best = None
    tested = 0
    feasible = 0

    # To keep C2 small but exact enough for the diagnostic, enumerate extraction counts by
    # (source flight, OD key) only for groups serviceable by the repeat route. Counts can range
    # freely; CP-SAT chooses them with 4 leg capacity constraints. Residual flights are then
    # reoptimized exactly. We iterate a bounded set of CP solutions by repeated no-good cuts.
    for rr in repeat_routes:
        spec = problem.aircraft_types[rr.aircraft_type]
        group_rows = []
        for uid in target_uids:
            grouped = defaultdict(list)
            for pid in elig[uid]:
                req = problem.requests[pid]
                interval = _repeat_group_interval(req.origin_id, req.destination_id, rr.base)
                if interval is not None:
                    grouped[(req.origin_id, req.destination_id, interval)].append(pid)
            for (o, d, interval), ids in sorted(grouped.items()):
                group_rows.append((uid, o, d, interval, tuple(sorted(ids))))
        if not group_rows:
            continue

        model = cp_model.CpModel()
        xs = []
        for gi, (_, o, d, interval, ids) in enumerate(group_rows):
            xs.append(model.new_int_var(0, len(ids), f"x{gi}"))
        for leg in range(4):
            model.add(sum(xs[i] for i, row in enumerate(group_rows) if row[3][0] <= leg < row[3][1]) <= spec.seats)
        model.add(sum(xs) >= 1)
        # Bias toward high repeat occupancy, but enumerate multiple alternatives below.
        model.maximize(sum(xs))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue

        # Enumerate top-volume solutions and nearby alternatives by repeatedly capping total moved.
        max_moved = int(round(solver.objective_value))
        for moved_target in range(max_moved, max(0, max_moved - 8), -1):
            model2 = cp_model.CpModel()
            ys = []
            for gi, (_, o, d, interval, ids) in enumerate(group_rows):
                ys.append(model2.new_int_var(0, len(ids), f"y{gi}"))
            for leg in range(4):
                model2.add(sum(ys[i] for i, row in enumerate(group_rows) if row[3][0] <= leg < row[3][1]) <= spec.seats)
            model2.add(sum(ys) == moved_target)
            # Encourage pulling from expensive/multiple flights by rewarding number of nonzero source groups.
            used = []
            for i, row in enumerate(group_rows):
                b = model2.new_bool_var(f"u{i}")
                model2.add(ys[i] >= 1).only_enforce_if(b)
                model2.add(ys[i] == 0).only_enforce_if(b.Not())
                used.append(b)
            model2.maximize(sum(used))
            s2 = cp_model.CpSolver()
            s2.parameters.max_time_in_seconds = 2.0
            s2.parameters.num_search_workers = 1
            st = s2.solve(model2)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                continue
            tested += 1

            moved_people = []
            residual = {uid: list(target_people[uid]) for uid in target_uids}
            for i, row in enumerate(group_rows):
                take = s2.value(ys[i])
                if take <= 0:
                    continue
                uid, _, _, _, ids = row
                chosen = list(ids[:take])
                moved_people.extend(chosen)
                chosen_set = set(chosen)
                residual[uid] = [pid for pid in residual[uid] if pid not in chosen_set]

            rebuilt = []
            ok = True
            residual_minutes = 0
            for uid in target_uids:
                ids = residual[uid]
                if not ids:
                    rebuilt.append((uid, (), None))
                    continue
                profile = optimizer.profile(problem, ids)
                opt = optimizer.optimize(problem, profile)
                if opt is None:
                    ok = False
                    break
                residual_minutes += opt.aircraft_minutes
                rebuilt.append((uid, tuple(ids), opt))
            if not ok:
                continue
            feasible += 1
            total_region = rr.trip_minutes + residual_minutes
            key = (total_region, -len(moved_people), rr.trip_minutes, rr.aircraft_type, rr.base)
            if best is None or key < best[0]:
                best = (key, rr, tuple(sorted(moved_people)), tuple(rebuilt))

    if best is None:
        raise RuntimeError("C2 found no feasible regional rebuild")

    _, rr, moved_people, rebuilt = best
    result = Solution(flights=dict(anchor.flights), assignments=dict(anchor.assignments))
    for uid, ids, opt in rebuilt:
        if not ids:
            if uid in result.flights:
                del result.flights[uid]
            continue
        optimizer.install(problem, result, uid, ids, opt)
    repeat_uid = "q2-c2-repeat-F044-F037-F044"
    _install_repeat(result, repeat_uid, rr, moved_people, problem)

    result_check = check_solution(problem, result, require_all_requests=True)
    if not result_check.ok:
        raise RuntimeError("C2 internal checker failed:\n" + "\n".join(result_check.errors))
    result_metrics = evaluate_solution(problem, result)

    routes_path = out / "q2-routes.csv"
    assignments_path = out / "q2-assignments.csv"
    _export(result, routes_path, assignments_path)
    ref = validate_submission(
        ROOT / "data/raw/distances.csv", ROOT / "data/raw/peopleQ2.csv",
        routes_path, assignments_path, require_all_requests=True,
    )
    if not ref.ok:
        raise RuntimeError("reference validator failed:\n" + "\n".join(ref.errors))

    summary = {
        "stage": "Q2-C2 F044/F037 four-flight regional rebuild with repeated service",
        "anchor_minutes": anchor_metrics.total_aircraft_usage_minutes,
        "old_region_minutes": old_region_minutes,
        "new_region_minutes": old_region_minutes - (anchor_metrics.total_aircraft_usage_minutes - result_metrics.total_aircraft_usage_minutes),
        "regional_savings_minutes": anchor_metrics.total_aircraft_usage_minutes - result_metrics.total_aircraft_usage_minutes,
        "new_global_minutes": result_metrics.total_aircraft_usage_minutes,
        "new_global_flights": result_metrics.number_of_flights,
        "seat_utilization": result_metrics.seat_utilization,
        "repeat_base": rr.base,
        "repeat_aircraft_type": rr.aircraft_type,
        "repeat_path": list(rr.sea_path),
        "repeat_refuel_flags": list(rr.refuel_flags),
        "repeat_trip_minutes": rr.trip_minutes,
        "moved_people": len(moved_people),
        "search_cases_tested": tested,
        "feasible_cases": feasible,
        "reference_validator": "PASS",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "moved_people.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h); w.writerow(["person_id", "origin_id", "destination_id"])
        for pid in moved_people:
            req = problem.requests[pid]
            w.writerow([pid, req.origin_id, req.destination_id])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
