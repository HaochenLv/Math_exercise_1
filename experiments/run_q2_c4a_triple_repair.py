from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.io.export_solution import export_q1_solution
from helicopter_planner.io.load_problem import load_q2_problem
from helicopter_planner.io.load_solution_csv import load_exported_solution
from helicopter_planner.solver.q2.pair_recombination import (
    Q2ExactPairRecombinationSolver,
    _FlightSummary,
)
from helicopter_planner.solver.q2.tail_elimination import OptimizedPassengerRoute
from reference_validator import validate_submission


@dataclass(frozen=True)
class TripleDecision:
    iteration: int
    source_flight_uids: tuple[str, str, str]
    pooled_group_count: int
    union_sea_facilities: tuple[str, ...]
    cp_status: str
    cp_objective_aircraft_minutes: int
    old_aircraft_usage_minutes: int
    new_aircraft_usage_minutes: int
    immediate_aircraft_savings_minutes: int
    passenger_minutes_delta: int
    fuel_kg_delta: float
    old_flight_count: int
    new_flight_count: int
    new_aircraft_types: tuple[str, ...]
    new_base_airports: tuple[str, ...]
    new_service_orders: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TripleIterationDiagnostics:
    iteration: int
    candidate_triple_count: int
    solved_triple_count: int
    cp_optimal_count: int
    cp_feasible_not_proven_count: int
    cp_no_solution_count: int
    improving_triple_count: int
    route_template_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class TripleResult:
    solution: object
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[TripleDecision, ...]
    diagnostics: tuple[TripleIterationDiagnostics, ...]
    converged: bool


@dataclass(frozen=True)
class _TripleCandidate:
    source_uids: tuple[str, str, str]
    passenger_groups: tuple[tuple[str, ...], ...]
    optimized_routes: tuple[OptimizedPassengerRoute, ...]
    old_cost: tuple[int, int, float]
    cp_status: str
    cp_objective_aircraft_minutes: int
    template_count: int
    pooled_group_count: int
    union_sea_facilities: tuple[str, ...]

    @property
    def new_aircraft_minutes(self) -> int:
        return sum(route.aircraft_minutes for route in self.optimized_routes)

    @property
    def new_passenger_minutes(self) -> int:
        return sum(route.passenger_minutes for route in self.optimized_routes)

    @property
    def new_fuel_kg(self) -> float:
        return sum(route.fuel_kg for route in self.optimized_routes)

    @property
    def savings(self) -> int:
        return self.old_cost[0] - self.new_aircraft_minutes

    @property
    def passenger_delta(self) -> int:
        return self.new_passenger_minutes - self.old_cost[1]

    @property
    def fuel_delta(self) -> float:
        return self.new_fuel_kg - self.old_cost[2]


class Q2AllFlightTripleRecombinationSolver(Q2ExactPairRecombinationSolver):
    """C4a: identity-free exact three-flight OD-count destroy-and-repair.

    The 2-flight identity-free neighborhood has converged at the C3b anchor.
    C4a destroys three geographically related current sorties at a time, pools
    all exact OD counts, and repartitions them over up to three rebuilt sorties.

    Each route slot uses the same exact distinct-service route family as C3:
    legal base airport, T1/T2/T3, service order and refuelling are enumerated,
    while CP-SAT assigns integer OD counts and enforces capacity on every leg.

    This first triple experiment intentionally limits pooled OD groups and the
    sea-facility union to keep the diagnostic bounded. It is a neighborhood
    restriction, not a claim of global optimality.
    """

    def __init__(
        self,
        *,
        neighbor_count: int = 4,
        max_union_facilities: int = 6,
        max_group_keys: int = 10,
        max_triples_per_iteration: int = 140,
        cp_time_limit_seconds: float = 1.5,
        max_iterations: int = 4,
    ) -> None:
        super().__init__(
            neighbor_count=neighbor_count,
            max_union_facilities=max_union_facilities,
            cp_time_limit_seconds=cp_time_limit_seconds,
            max_iterations=max_iterations,
            tail_max_hosts_per_donor=32,
        )
        self.max_group_keys = max_group_keys
        self.max_triples_per_iteration = max_triples_per_iteration

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

    @staticmethod
    def _pair_distance(problem, left: _FlightSummary, right: _FlightSummary) -> float:
        return min(
            problem.distance(a, b)
            for a in left.sea_facilities
            for b in right.sea_facilities
        )

    def _candidate_triples(self, problem, summaries):
        uids = sorted(summaries)
        triples = set()

        for uid in uids:
            left = summaries[uid]
            ranked = sorted(
                (
                    self._pair_distance(problem, left, summaries[other_uid]),
                    other_uid,
                )
                for other_uid in uids
                if other_uid != uid
            )
            neighbors = [other_uid for _, other_uid in ranked[: self.neighbor_count]]
            for a, b in itertools.combinations(neighbors, 2):
                triple = tuple(sorted((uid, a, b)))
                union = set().union(*(summaries[x].sea_facilities for x in triple))
                if len(union) > self.max_union_facilities:
                    continue
                group_keys = {
                    (origin, destination)
                    for x in triple
                    for origin, destination, count in summaries[x].profile
                    if count > 0
                }
                if len(group_keys) <= self.max_group_keys:
                    triples.add(triple)

        def score(triple):
            ss = [summaries[x] for x in triple]
            union = set().union(*(x.sea_facilities for x in ss))
            pair_distance_sum = sum(
                self._pair_distance(problem, ss[i], ss[j])
                for i in range(3)
                for j in range(i + 1, 3)
            )
            shared = sum(
                len(ss[i].sea_facilities & ss[j].sea_facilities)
                for i in range(3)
                for j in range(i + 1, 3)
            )
            old_minutes = sum(x.cost[0] for x in ss)
            return (len(union), pair_distance_sum, -shared, -old_minutes, triple)

        ranked = sorted(triples, key=score)
        return tuple(ranked[: self.max_triples_per_iteration])

    def _solve_triple(self, problem, summaries, triple):
        ss = [summaries[uid] for uid in triple]
        union = tuple(sorted(set().union(*(x.sea_facilities for x in ss))))
        if len(union) > self.max_union_facilities:
            return None, ("NOT_RUN", 0)

        pooled_ids = tuple(sorted(pid for s in ss for pid in s.person_ids))
        people_by_key = {}
        for pid in pooled_ids:
            req = problem.requests[pid]
            people_by_key.setdefault((req.origin_id, req.destination_id), []).append(pid)
        group_keys = tuple(sorted(people_by_key))
        if len(group_keys) > self.max_group_keys:
            return None, ("NOT_RUN", 0)
        demand = tuple(len(people_by_key[key]) for key in group_keys)

        old_aircraft = sum(s.cost[0] for s in ss)
        templates = tuple(
            template
            for template in self._route_templates(problem, group_keys)
            if template.route.trip_minutes < old_aircraft
        )
        if not templates:
            return None, ("INFEASIBLE", 0)

        serviceable_by_group = [
            tuple(
                index
                for index, template in enumerate(templates)
                if group_index in template.serviceable_groups
            )
            for group_index in range(len(group_keys))
        ]
        if any(not indices for indices in serviceable_by_group):
            return None, ("INFEASIBLE", len(templates))

        model = cp_model.CpModel()
        route_slots = 3
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

        model.add(used[0] >= used[1])
        model.add(used[1] >= used[2])
        model.add(choices[0] <= choices[1]).only_enforce_if(used[1])
        model.add(choices[1] <= choices[2]).only_enforce_if(used[2])

        for group_index, count in enumerate(demand):
            model.add(
                sum(x[slot, group_index] for slot in range(route_slots)) == count
            )

        aircraft_expr = sum(
            template.route.trip_minutes * z[slot, template_index]
            for slot in range(route_slots)
            for template_index, template in enumerate(templates)
        )
        model.minimize(aircraft_expr)

        cp = cp_model.CpSolver()
        cp.parameters.max_time_in_seconds = self.cp_time_limit_seconds
        cp.parameters.num_search_workers = 1
        cp.parameters.random_seed = 0
        status = cp.solve(model)
        status_name = self._status_name(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, (status_name, len(templates))

        cp_objective = int(round(cp.objective_value))
        if cp_objective >= old_aircraft:
            return None, (status_name, len(templates))

        cursors = {key: 0 for key in group_keys}
        passenger_groups = []
        for slot in range(route_slots):
            if cp.value(used[slot]) == 0:
                continue
            ids = []
            for group_index, key in enumerate(group_keys):
                count = cp.value(x[slot, group_index])
                start = cursors[key]
                ids.extend(people_by_key[key][start : start + count])
                cursors[key] += count
            if not ids:
                raise AssertionError("C4a selected an empty used route slot")
            passenger_groups.append(tuple(sorted(ids)))

        if any(cursors[key] != len(people_by_key[key]) for key in group_keys):
            raise AssertionError("C4a CP allocation did not cover pooled demand")

        optimized_routes = []
        for ids in passenger_groups:
            profile = self.route_optimizer.profile(problem, ids)
            optimized = self.route_optimizer.optimize(problem, profile)
            if optimized is None:
                return None, (status_name, len(templates))
            optimized_routes.append(optimized)

        old_cost = (
            old_aircraft,
            sum(s.cost[1] for s in ss),
            sum(s.cost[2] for s in ss),
        )
        candidate = _TripleCandidate(
            source_uids=triple,
            passenger_groups=tuple(passenger_groups),
            optimized_routes=tuple(optimized_routes),
            old_cost=old_cost,
            cp_status=status_name,
            cp_objective_aircraft_minutes=cp_objective,
            template_count=len(templates),
            pooled_group_count=len(group_keys),
            union_sea_facilities=union,
        )
        return (candidate if candidate.savings > 0 else None), (
            status_name,
            len(templates),
        )

    @staticmethod
    def _candidate_key(candidate):
        return (
            -candidate.savings,
            candidate.passenger_delta,
            candidate.fuel_delta,
            len(candidate.optimized_routes),
            candidate.source_uids,
        )

    def _apply_triple(self, problem, solution, candidate):
        groups = list(
            zip(candidate.passenger_groups, candidate.optimized_routes, strict=True)
        )
        groups.sort(
            key=lambda item: (
                item[1].aircraft_minutes,
                item[1].route.aircraft_type,
                item[1].route.base_airport,
                item[1].route.service_order,
                item[0],
            )
        )
        for index, uid in enumerate(candidate.source_uids):
            if index < len(groups):
                ids, optimized = groups[index]
                self.route_optimizer.install(problem, solution, uid, ids, optimized)
            elif uid in solution.flights:
                del solution.flights[uid]

    def solve_triples(self, problem, starting_solution):
        solution = copy.deepcopy(starting_solution)
        ck = check_solution(problem, solution, require_all_requests=True)
        if not ck.ok:
            raise ValueError("C4a starting solution invalid:\n" + "\n".join(ck.errors))
        starting_metrics = evaluate_solution(problem, solution)
        decisions = []
        diagnostics = []
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            tic = time.monotonic()
            summaries = self._summaries(problem, solution)
            triples = self._candidate_triples(problem, summaries)
            print(
                f"[C4a] iteration={iteration} flights={len(summaries)} "
                f"candidate_triples={len(triples)}",
                flush=True,
            )

            best = None
            cp_optimal = 0
            cp_feasible = 0
            cp_none = 0
            solved = 0
            improving = 0
            template_total = 0

            for idx, triple in enumerate(triples, 1):
                candidate, (status_name, template_count) = self._solve_triple(
                    problem, summaries, triple
                )
                if status_name == "NOT_RUN":
                    continue
                solved += 1
                template_total += template_count
                if status_name == "OPTIMAL":
                    cp_optimal += 1
                elif status_name == "FEASIBLE":
                    cp_feasible += 1
                else:
                    cp_none += 1

                if candidate is not None:
                    improving += 1
                    if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                        best = candidate

                if idx % 20 == 0 or idx == len(triples):
                    current_best = 0 if best is None else best.savings
                    print(
                        f"[C4a] iteration={iteration} progress={idx}/{len(triples)} "
                        f"solved={solved} improving={improving} best_savings={current_best} "
                        f"elapsed={time.monotonic()-tic:.1f}s",
                        flush=True,
                    )

            elapsed = time.monotonic() - tic
            diagnostics.append(
                TripleIterationDiagnostics(
                    iteration=iteration,
                    candidate_triple_count=len(triples),
                    solved_triple_count=solved,
                    cp_optimal_count=cp_optimal,
                    cp_feasible_not_proven_count=cp_feasible,
                    cp_no_solution_count=cp_none,
                    improving_triple_count=improving,
                    route_template_count=template_total,
                    elapsed_seconds=elapsed,
                )
            )

            if best is None or best.savings <= 0:
                converged = True
                print(
                    f"[C4a] iteration={iteration} no improving triple; converged",
                    flush=True,
                )
                break

            before = evaluate_solution(problem, solution)
            self._apply_triple(problem, solution, best)
            ck = check_solution(problem, solution, require_all_requests=True)
            if not ck.ok:
                raise AssertionError(
                    "C4a invalid after triple repair:\n" + "\n".join(ck.errors)
                )
            after = evaluate_solution(problem, solution)
            actual_savings = (
                before.total_aircraft_usage_minutes
                - after.total_aircraft_usage_minutes
            )
            if actual_savings <= 0:
                raise AssertionError("accepted C4a triple did not improve aircraft time")

            decisions.append(
                TripleDecision(
                    iteration=iteration,
                    source_flight_uids=best.source_uids,
                    pooled_group_count=best.pooled_group_count,
                    union_sea_facilities=best.union_sea_facilities,
                    cp_status=best.cp_status,
                    cp_objective_aircraft_minutes=best.cp_objective_aircraft_minutes,
                    old_aircraft_usage_minutes=best.old_cost[0],
                    new_aircraft_usage_minutes=best.new_aircraft_minutes,
                    immediate_aircraft_savings_minutes=actual_savings,
                    passenger_minutes_delta=best.passenger_delta,
                    fuel_kg_delta=best.fuel_delta,
                    old_flight_count=3,
                    new_flight_count=len(best.optimized_routes),
                    new_aircraft_types=tuple(
                        route.route.aircraft_type for route in best.optimized_routes
                    ),
                    new_base_airports=tuple(
                        route.route.base_airport for route in best.optimized_routes
                    ),
                    new_service_orders=tuple(
                        route.route.service_order for route in best.optimized_routes
                    ),
                )
            )
            print(
                f"[C4a] ACCEPT iteration={iteration} savings={actual_savings} "
                f"global={after.total_aircraft_usage_minutes} "
                f"flights={after.number_of_flights} sources={best.source_uids}",
                flush=True,
            )

        final_ck = check_solution(problem, solution, require_all_requests=True)
        if not final_ck.ok:
            raise AssertionError("C4a final solution invalid:\n" + "\n".join(final_ck.errors))
        return TripleResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            diagnostics=tuple(diagnostics),
            converged=converged,
        )


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-routes", type=Path, required=True)
    ap.add_argument("--anchor-assignments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/q2/c4a_triple"))
    ap.add_argument("--neighbor-count", type=int, default=4)
    ap.add_argument("--max-union-facilities", type=int, default=6)
    ap.add_argument("--max-group-keys", type=int, default=10)
    ap.add_argument("--max-triples-per-iteration", type=int, default=140)
    ap.add_argument("--cp-time-limit", type=float, default=1.5)
    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--expected-anchor-minutes", type=int, default=18928)
    ap.add_argument("--expected-anchor-flights", type=int, default=107)
    args = ap.parse_args()

    def resolve(path):
        return path if path.is_absolute() else ROOT / path

    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    distances = ROOT / "data/raw/distances.csv"
    people = ROOT / "data/raw/peopleQ2.csv"
    problem = load_q2_problem(distances, people)
    anchor = load_exported_solution(
        resolve(args.anchor_routes),
        resolve(args.anchor_assignments),
        uid_prefix="q2-c4a-anchor",
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

    solver = Q2AllFlightTripleRecombinationSolver(
        neighbor_count=args.neighbor_count,
        max_union_facilities=args.max_union_facilities,
        max_group_keys=args.max_group_keys,
        max_triples_per_iteration=args.max_triples_per_iteration,
        cp_time_limit_seconds=args.cp_time_limit,
        max_iterations=args.max_iterations,
    )
    result = solver.solve_triples(problem, anchor)

    routes = out / "q2-routes.csv"
    assignments = out / "q2-assignments.csv"
    export_q1_solution(result.solution, routes, assignments)
    ref = validate_submission(
        distances, people, routes, assignments, require_all_requests=True
    )
    if not ref.ok:
        raise RuntimeError("reference validator failed:\n" + "\n".join(ref.errors))

    metrics = metrics_dict(result.metrics)
    for key, actual in ref.metrics.items():
        expected = metrics[key]
        if isinstance(expected, float):
            if abs(float(expected) - float(actual)) > 1e-6:
                raise RuntimeError(f"metric mismatch {key}: {expected} != {actual}")
        elif expected != actual:
            raise RuntimeError(f"metric mismatch {key}: {expected} != {actual}")

    decision_rows = [asdict(x) for x in result.decisions]
    if decision_rows:
        with (out / "triple_recombinations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as h:
            w = csv.DictWriter(h, fieldnames=list(decision_rows[0]))
            w.writeheader()
            for row in decision_rows:
                row["source_flight_uids"] = ";".join(row["source_flight_uids"])
                row["union_sea_facilities"] = ";".join(row["union_sea_facilities"])
                row["new_aircraft_types"] = ";".join(row["new_aircraft_types"])
                row["new_base_airports"] = ";".join(row["new_base_airports"])
                row["new_service_orders"] = ";".join(
                    "->".join(order) for order in row["new_service_orders"]
                )
                w.writerow(row)

    diag_rows = [asdict(x) for x in result.diagnostics]
    if diag_rows:
        with (out / "iteration_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as h:
            w = csv.DictWriter(h, fieldnames=list(diag_rows[0]))
            w.writeheader()
            w.writerows(diag_rows)

    summary = {
        "stage": "Q2-C4a identity-free exact three-flight OD-count destroy-and-repair",
        "anchor": metrics_dict(am),
        "result": metrics,
        "aircraft_savings_minutes": (
            am.total_aircraft_usage_minutes
            - result.metrics.total_aircraft_usage_minutes
        ),
        "flight_count_change": (
            result.metrics.number_of_flights - am.number_of_flights
        ),
        "accepted_triple_moves": len(result.decisions),
        "iterations_scanned": len(result.diagnostics),
        "candidate_triples_scanned": sum(
            x.candidate_triple_count for x in result.diagnostics
        ),
        "solved_triples": sum(x.solved_triple_count for x in result.diagnostics),
        "improving_triples_seen": sum(
            x.improving_triple_count for x in result.diagnostics
        ),
        "route_templates_considered": sum(
            x.route_template_count for x in result.diagnostics
        ),
        "neighbor_count": args.neighbor_count,
        "max_union_facilities": args.max_union_facilities,
        "max_group_keys": args.max_group_keys,
        "max_triples_per_iteration": args.max_triples_per_iteration,
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
            {
                "ok": ref.ok,
                "errors": list(ref.errors),
                "metrics": ref.metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
