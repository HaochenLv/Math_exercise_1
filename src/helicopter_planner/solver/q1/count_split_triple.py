from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from ortools.sat.python import cp_model

from helicopter_planner.domain import Assignment, FlightPlan, ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.count_split_recombination import (
    CountSplitDecision,
    Q1ExactCountSplitPairSolver,
    _RouteConfig,
)
from helicopter_planner.solver.q1.local_search import (
    Q1FacilityBlockLocalSearchSolver,
    _OptimizedPattern,
)
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder


class _StartingSolver(Protocol):
    def solve(self, problem: ProblemData) -> Solution: ...


class _StaticStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


@dataclass(frozen=True)
class CountSplitTripleDecision:
    iteration: int
    base_airport: str
    source_flight_uids: tuple[str, str, str]
    destinations: tuple[str, ...]
    old_flight_count: int
    new_flight_count: int
    old_aircraft_usage_minutes: int
    new_aircraft_usage_minutes: int
    immediate_aircraft_savings_minutes: int
    total_iteration_aircraft_savings_minutes: int
    immediate_passenger_savings_minutes: int
    immediate_fuel_savings_kg: float
    new_aircraft_types: tuple[str, ...]
    new_service_orders: tuple[tuple[str, ...], ...]
    pair_polish_moves: int


@dataclass(frozen=True)
class CountSplitTripleResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[CountSplitTripleDecision, ...]
    pair_polish_decisions: tuple[CountSplitDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _TripleCountCandidate:
    source_uids: tuple[str, str, str]
    base_airport: str
    destinations: tuple[str, ...]
    passenger_groups: tuple[tuple[str, ...], ...]
    patterns: tuple[_OptimizedPattern, ...]
    old_aircraft_usage_minutes: int
    old_passenger_travel_minutes: int
    old_fuel_kg: float

    @property
    def new_aircraft_usage_minutes(self) -> int:
        return sum(pattern.route.trip_minutes for pattern in self.patterns)

    @property
    def new_passenger_travel_minutes(self) -> int:
        return sum(pattern.passenger_travel_minutes for pattern in self.patterns)

    @property
    def new_fuel_kg(self) -> float:
        return sum(pattern.fuel_kg for pattern in self.patterns)

    @property
    def aircraft_savings_minutes(self) -> int:
        return self.old_aircraft_usage_minutes - self.new_aircraft_usage_minutes

    @property
    def passenger_savings_minutes(self) -> int:
        return self.old_passenger_travel_minutes - self.new_passenger_travel_minutes

    @property
    def fuel_savings_kg(self) -> float:
        return self.old_fuel_kg - self.new_fuel_kg

    @property
    def new_flight_count(self) -> int:
        return len(self.patterns)


class Q1ExactCountSplitTripleSolver(Q1ExactCountSplitPairSolver):
    """Q1 V11: exact destination-count splitting over three local routes.

    V10 proved that current destination blocks are not perfectly aligned with
    the primary objective, but a two-route repair can only coordinate one pair.
    V11 destroys three nearby same-base routes and pools passenger *counts* by
    destination. A CP-SAT repair may split every destination count arbitrarily
    across up to four rebuilt routes.

    Route-slot costs are true minimum aircraft times for the chosen support and
    load: all feasible aircraft types, service orders and legal refueling
    patterns are evaluated by the common route builder. The CP-SAT objective is
    therefore exact for aircraft time inside each searched three-route repair.

    After an accepted V11 move, the exact V10 pair neighborhood is polished to
    convergence. V10 in turn invokes the earlier generalized tail search.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 12,
        polish_max_iterations: int = 200,
        neighbor_count: int = 5,
        max_union_destinations: int = 6,
        cp_time_limit_seconds: float = 0.30,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1ExactCountSplitPairSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=20,
                polish_max_iterations=polish_max_iterations,
                neighbor_count=8,
                max_union_destinations=6,
                cp_time_limit_seconds=0.20,
            )
        super().__init__(
            starting_solver=starting_solver,
            route_builder=builder,
            max_service_destinations=max_service_destinations,
            max_iterations=max_iterations,
            polish_max_iterations=polish_max_iterations,
            neighbor_count=neighbor_count,
            max_union_destinations=max_union_destinations,
            cp_time_limit_seconds=cp_time_limit_seconds,
        )

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> CountSplitTripleResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[CountSplitTripleDecision] = []
        pair_polish_decisions: list[CountSplitDecision] = []
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}
        pattern_cache: dict[
            tuple[str, tuple[tuple[str, int], ...]], _OptimizedPattern | None
        ] = {}
        support_cost_cache: dict[tuple[str, tuple[str, ...], int], int | None] = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _TripleCountCandidate | None = None

            for triple in self._candidate_triples(problem, solution, summaries):
                candidate = self._solve_triple(
                    problem,
                    solution,
                    summaries,
                    triple,
                    route_cache,
                    pattern_cache,
                    support_cost_cache,
                )
                if candidate is None:
                    continue
                if best is None or self._triple_key(candidate) < self._triple_key(best):
                    best = candidate

            if best is None:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            self._apply_triple_candidate(problem, solution, best, iteration)

            pair_polish = Q1ExactCountSplitPairSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=20,
                polish_max_iterations=self.polish_max_iterations,
                neighbor_count=8,
                max_union_destinations=6,
                cp_time_limit_seconds=0.20,
            ).solve_with_diagnostics(problem)
            if not pair_polish.converged:
                raise RuntimeError("V11 V10 polish hit max_iterations")
            solution = pair_polish.solution
            pair_polish_decisions.extend(pair_polish.decisions)

            after = evaluate_solution(problem, solution)
            total_savings = before.total_aircraft_usage_minutes - after.total_aircraft_usage_minutes
            if total_savings <= 0:
                raise AssertionError("accepted V11 iteration did not improve aircraft time")

            decisions.append(
                CountSplitTripleDecision(
                    iteration=iteration,
                    base_airport=best.base_airport,
                    source_flight_uids=best.source_uids,
                    destinations=best.destinations,
                    old_flight_count=3,
                    new_flight_count=best.new_flight_count,
                    old_aircraft_usage_minutes=best.old_aircraft_usage_minutes,
                    new_aircraft_usage_minutes=best.new_aircraft_usage_minutes,
                    immediate_aircraft_savings_minutes=best.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    immediate_passenger_savings_minutes=best.passenger_savings_minutes,
                    immediate_fuel_savings_kg=best.fuel_savings_kg,
                    new_aircraft_types=tuple(pattern.route.aircraft_type for pattern in best.patterns),
                    new_service_orders=tuple(pattern.route.service_order for pattern in best.patterns),
                    pair_polish_moves=len(pair_polish.decisions),
                )
            )

        return CountSplitTripleResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            pair_polish_decisions=tuple(pair_polish_decisions),
            converged=converged,
        )

    def _candidate_triples(self, problem, solution, summaries) -> tuple[tuple[str, str, str], ...]:
        by_base: dict[str, list[str]] = {}
        for uid, flight in solution.flights.items():
            by_base.setdefault(flight.base_airport, []).append(uid)

        triples: set[tuple[str, str, str]] = set()
        for raw_uids in by_base.values():
            uids = sorted(raw_uids)
            if len(uids) < 3:
                continue
            destination_sets = {
                uid: tuple(destination for destination, _ in summaries[uid].blocks)
                for uid in uids
            }
            for uid in uids:
                ranked = sorted(
                    (
                        self._route_distance(problem, destination_sets[uid], destination_sets[other]),
                        other,
                    )
                    for other in uids
                    if other != uid
                )
                neighbors = [other for _, other in ranked[: self.neighbor_count]]
                for left, right in itertools.combinations(neighbors, 2):
                    triple = tuple(sorted((uid, left, right)))
                    union = set().union(*(set(destination_sets[item]) for item in triple))
                    if len(union) <= self.max_union_destinations:
                        triples.add(triple)

            # Shared-destination triples are the highest-value count-split cases.
            for triple in itertools.combinations(uids, 3):
                union = set().union(*(set(destination_sets[item]) for item in triple))
                if len(union) > self.max_union_destinations:
                    continue
                sets = [set(destination_sets[item]) for item in triple]
                if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
                    triples.add(tuple(sorted(triple)))

        return tuple(sorted(triples))

    def _solve_triple(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries,
        triple: tuple[str, str, str],
        route_cache,
        pattern_cache,
        support_cost_cache,
    ) -> _TripleCountCandidate | None:
        flights = [solution.flights[uid] for uid in triple]
        if len({flight.base_airport for flight in flights}) != 1:
            return None
        base = flights[0].base_airport

        pooled_ids = tuple(
            sorted(pid for uid in triple for pid in summaries[uid].passenger_ids)
        )
        people_by_destination: dict[str, list[str]] = {}
        for pid in pooled_ids:
            destination = problem.requests[pid].destination_id
            people_by_destination.setdefault(destination, []).append(pid)
        destinations = tuple(sorted(people_by_destination))
        if len(destinations) > self.max_union_destinations:
            return None

        demand = {destination: len(people_by_destination[destination]) for destination in destinations}
        total_people = len(pooled_ids)
        max_seats = max(spec.seats for spec in problem.aircraft_types.values())
        min_routes = (total_people + max_seats - 1) // max_seats
        route_slots = 4
        if min_routes > route_slots:
            return None

        configs: list[_RouteConfig] = []
        for support_size in range(1, min(self.max_service_destinations, len(destinations)) + 1):
            for support in itertools.combinations(destinations, support_size):
                max_load = min(max_seats, sum(demand[d] for d in support))
                for load in range(support_size, max_load + 1):
                    cost = self._support_load_cost(
                        problem,
                        base,
                        support,
                        load,
                        route_cache,
                        support_cost_cache,
                    )
                    if cost is not None:
                        configs.append(_RouteConfig(support=support, load=load, trip_minutes=cost))
        if not configs:
            return None

        model = cp_model.CpModel()
        z = {}
        x = {}
        used = []
        for r in range(route_slots):
            empty = model.new_bool_var(f"empty_{r}")
            vars_r = [empty]
            for j in range(len(configs)):
                var = model.new_bool_var(f"z_{r}_{j}")
                z[r, j] = var
                vars_r.append(var)
            model.add_exactly_one(vars_r)
            u = model.new_bool_var(f"used_{r}")
            model.add(u + empty == 1)
            used.append(u)

            for d_index, destination in enumerate(destinations):
                x[r, d_index] = model.new_int_var(0, demand[destination], f"x_{r}_{d_index}")
                support_expr = sum(
                    z[r, j]
                    for j, config in enumerate(configs)
                    if destination in config.support
                )
                model.add(x[r, d_index] <= demand[destination] * support_expr)
                model.add(x[r, d_index] >= support_expr)

            model.add(
                sum(x[r, d_index] for d_index in range(len(destinations)))
                == sum(config.load * z[r, j] for j, config in enumerate(configs))
            )

        for r in range(route_slots - 1):
            model.add(used[r] >= used[r + 1])
        model.add(sum(used) >= min_routes)
        for d_index, destination in enumerate(destinations):
            model.add(sum(x[r, d_index] for r in range(route_slots)) == demand[destination])

        model.minimize(
            sum(
                config.trip_minutes * z[r, j]
                for r in range(route_slots)
                for j, config in enumerate(configs)
            )
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.cp_time_limit_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        old_aircraft = sum(summaries[uid].aircraft_usage_minutes for uid in triple)
        if solver.objective_value >= old_aircraft - 1e-9:
            return None

        cursors = {destination: 0 for destination in destinations}
        passenger_groups: list[tuple[str, ...]] = []
        for r in range(route_slots):
            if solver.value(used[r]) == 0:
                continue
            ids: list[str] = []
            for d_index, destination in enumerate(destinations):
                count = solver.value(x[r, d_index])
                start = cursors[destination]
                ids.extend(people_by_destination[destination][start : start + count])
                cursors[destination] += count
            passenger_groups.append(tuple(sorted(ids)))

        if any(cursors[d] != demand[d] for d in destinations):
            raise AssertionError("V11 CP-SAT allocation did not cover local demand")

        patterns: list[_OptimizedPattern] = []
        for group in passenger_groups:
            pattern = self._optimize_pattern(problem, base, group, route_cache, pattern_cache)
            if pattern is None:
                return None
            patterns.append(pattern)

        candidate = _TripleCountCandidate(
            source_uids=triple,
            base_airport=base,
            destinations=destinations,
            passenger_groups=tuple(passenger_groups),
            patterns=tuple(patterns),
            old_aircraft_usage_minutes=old_aircraft,
            old_passenger_travel_minutes=sum(summaries[uid].passenger_travel_minutes for uid in triple),
            old_fuel_kg=sum(summaries[uid].fuel_kg for uid in triple),
        )
        return candidate if candidate.aircraft_savings_minutes > 0 else None

    @staticmethod
    def _triple_key(candidate: _TripleCountCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            candidate.new_flight_count,
            candidate.source_uids,
            tuple(pattern.route.service_order for pattern in candidate.patterns),
        )

    @staticmethod
    def _apply_triple_candidate(
        problem: ProblemData,
        solution: Solution,
        candidate: _TripleCountCandidate,
        iteration: int,
    ) -> None:
        for uid in candidate.source_uids:
            for pid in [
                pid
                for pid, assignment in solution.assignments.items()
                if assignment.flight_uid == uid
            ]:
                del solution.assignments[pid]

        groups = list(zip(candidate.passenger_groups, candidate.patterns, strict=True))
        groups.sort(key=lambda item: (item[1].route.aircraft_type, item[1].route.service_order, item[0]))
        for index, uid in enumerate(candidate.source_uids):
            if index < len(groups):
                passengers, pattern = groups[index]
                Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
                    problem, solution, uid, passengers, pattern
                )
            else:
                Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
                    problem, solution, uid, (), None
                )

        for extra_index, (passengers, pattern) in enumerate(groups[len(candidate.source_uids):], start=1):
            serial = 0
            while True:
                uid = f"V11_{iteration:03d}_{extra_index:02d}_{serial:03d}"
                if uid not in solution.flights:
                    break
                serial += 1
            route = pattern.route
            solution.flights[uid] = FlightPlan(
                flight_uid=uid,
                base_airport=candidate.base_airport,
                aircraft_type=route.aircraft_type,
                sea_stops=list(route.sea_stops),
            )
            stop_index = dict(zip(route.service_order, route.service_stop_indices, strict=True))
            for pid in passengers:
                destination = problem.requests[pid].destination_id
                solution.assignments[pid] = Assignment(
                    person_id=pid,
                    flight_uid=uid,
                    pickup_index=0,
                    delivery_index=stop_index[destination],
                )
