from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.cross_airport import (
    CrossAirportDecision,
    Q1LandCrossAirportLocalSearchSolver,
)
from helicopter_planner.solver.q1.generalized_tail_elimination import (
    GeneralizedTailEliminationDecision,
    Q1RetypeAwareTailEliminationSolver,
)
from helicopter_planner.solver.q1.local_search import (
    LocalSearchDecision,
    Q1FacilityBlockLocalSearchSolver,
    _OptimizedPattern,
)
from helicopter_planner.solver.q1.partial_relocate import (
    PartialRelocateDecision,
    Q1PartialPassengerLocalSearchSolver,
)
from helicopter_planner.solver.q1.route_builder import Q1RouteBuilder


class _StartingSolver(Protocol):
    def solve(self, problem: ProblemData) -> Solution: ...


class _StaticStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


@dataclass(frozen=True)
class _Block:
    original_flight_uid: str
    destination_id: str
    passenger_ids: tuple[str, ...]

    @property
    def passenger_count(self) -> int:
        return len(self.passenger_ids)


@dataclass(frozen=True)
class RecombinationDecision:
    iteration: int
    base_airport: str
    source_flight_uids: tuple[str, str, str]
    source_block_count: int
    source_passenger_count: int
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
    block_polish_move_count: int
    partial_polish_move_count: int
    generalized_tail_polish_count: int
    cross_airport_polish_count: int


@dataclass(frozen=True)
class RecombinationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[RecombinationDecision, ...]
    block_polish_decisions: tuple[LocalSearchDecision, ...]
    partial_polish_decisions: tuple[PartialRelocateDecision, ...]
    generalized_tail_polish_decisions: tuple[GeneralizedTailEliminationDecision, ...]
    cross_airport_polish_decisions: tuple[CrossAirportDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _TripleCandidate:
    source_uids: tuple[str, str, str]
    base_airport: str
    block_count: int
    passenger_count: int
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


class Q1ThreeRouteRecombinationSolver(Q1FacilityBlockLocalSearchSolver):
    """Q1 V7: exact small-block recombination of three nearby same-base routes.

    V3-V6 are all pairwise or donor-based neighborhoods, so they can stop at a
    local optimum where a three-way cyclic exchange is beneficial even though
    every single relocate/swap is non-improving. V7 destroys three routes at a
    time and repartitions their *current destination blocks* into one, two, or
    three rebuilt routes.

    To keep the neighborhood exact but tractable, a triple is searched when it
    is spatially local (formed from nearest route neighbors) or when its total
    passenger count is small enough that a 3->2 reduction is capacity-feasible.
    Triples with more than ``max_blocks`` current destination blocks are skipped.

    For each searched triple, all canonical set partitions of its blocks are
    enumerated. Every new route is then optimized from scratch across aircraft
    type, service order, and legal technical refueling. The globally best
    strictly positive aircraft-time move is accepted.

    After each accepted V7 move, V3, V4, V5, and V6 are run as lower-neighborhood
    polishing steps before the three-route neighborhood is searched again.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 20,
        polish_max_iterations: int = 200,
        neighbor_count: int = 6,
        max_blocks: int = 8,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1LandCrossAirportLocalSearchSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=50,
                polish_max_iterations=polish_max_iterations,
            )
        super().__init__(
            starting_solver=starting_solver,
            route_builder=builder,
            max_service_destinations=max_service_destinations,
            max_iterations=max_iterations,
        )
        self.polish_max_iterations = polish_max_iterations
        self.neighbor_count = neighbor_count
        self.max_blocks = max_blocks

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> RecombinationResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[RecombinationDecision] = []
        block_polish_decisions: list[LocalSearchDecision] = []
        partial_polish_decisions: list[PartialRelocateDecision] = []
        generalized_tail_polish_decisions: list[GeneralizedTailEliminationDecision] = []
        cross_airport_polish_decisions: list[CrossAirportDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _TripleCandidate | None = None

            for triple in self._candidate_triples(problem, solution, summaries):
                candidate = self._best_for_triple(
                    problem,
                    solution,
                    summaries,
                    triple,
                    route_cache,
                    pattern_cache,
                )
                if candidate is None:
                    continue
                if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                    best = candidate

            if best is None:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            self._apply_candidate(problem, solution, best)

            block_polish = Q1FacilityBlockLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not block_polish.converged:
                raise RuntimeError("V7 V3 polish hit max_iterations")
            solution = block_polish.solution
            block_polish_decisions.extend(block_polish.decisions)

            partial_polish = Q1PartialPassengerLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
                polish_max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not partial_polish.converged:
                raise RuntimeError("V7 V4 polish hit max_iterations")
            solution = partial_polish.solution
            partial_polish_decisions.extend(partial_polish.decisions)
            block_polish_decisions.extend(partial_polish.polish_decisions)

            generalized_polish = Q1RetypeAwareTailEliminationSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
                polish_max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not generalized_polish.converged:
                raise RuntimeError("V7 V5 polish hit max_iterations")
            solution = generalized_polish.solution
            generalized_tail_polish_decisions.extend(generalized_polish.decisions)
            block_polish_decisions.extend(generalized_polish.block_polish_decisions)
            partial_polish_decisions.extend(generalized_polish.partial_polish_decisions)

            cross_polish = Q1LandCrossAirportLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=50,
                polish_max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not cross_polish.converged:
                raise RuntimeError("V7 V6 polish hit max_iterations")
            solution = cross_polish.solution
            cross_airport_polish_decisions.extend(cross_polish.decisions)
            block_polish_decisions.extend(cross_polish.block_polish_decisions)
            partial_polish_decisions.extend(cross_polish.partial_polish_decisions)
            generalized_tail_polish_decisions.extend(
                cross_polish.generalized_tail_polish_decisions
            )

            after = evaluate_solution(problem, solution)
            total_savings = (
                before.total_aircraft_usage_minutes
                - after.total_aircraft_usage_minutes
            )
            if total_savings <= 0:
                raise AssertionError("accepted V7 iteration did not improve aircraft time")

            decisions.append(
                RecombinationDecision(
                    iteration=iteration,
                    base_airport=best.base_airport,
                    source_flight_uids=best.source_uids,
                    source_block_count=best.block_count,
                    source_passenger_count=best.passenger_count,
                    old_flight_count=3,
                    new_flight_count=best.new_flight_count,
                    old_aircraft_usage_minutes=best.old_aircraft_usage_minutes,
                    new_aircraft_usage_minutes=best.new_aircraft_usage_minutes,
                    immediate_aircraft_savings_minutes=best.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    immediate_passenger_savings_minutes=best.passenger_savings_minutes,
                    immediate_fuel_savings_kg=best.fuel_savings_kg,
                    new_aircraft_types=tuple(
                        pattern.route.aircraft_type for pattern in best.patterns
                    ),
                    new_service_orders=tuple(
                        pattern.route.service_order for pattern in best.patterns
                    ),
                    block_polish_move_count=len(block_polish.decisions),
                    partial_polish_move_count=len(partial_polish.decisions),
                    generalized_tail_polish_count=len(generalized_polish.decisions),
                    cross_airport_polish_count=len(cross_polish.decisions),
                )
            )

        return RecombinationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            block_polish_decisions=tuple(block_polish_decisions),
            partial_polish_decisions=tuple(partial_polish_decisions),
            generalized_tail_polish_decisions=tuple(generalized_tail_polish_decisions),
            cross_airport_polish_decisions=tuple(cross_airport_polish_decisions),
            converged=converged,
        )

    def _candidate_triples(self, problem, solution, summaries) -> tuple[tuple[str, str, str], ...]:
        by_base: dict[str, list[str]] = {}
        for uid, flight in solution.flights.items():
            by_base.setdefault(flight.base_airport, []).append(uid)

        triples: set[tuple[str, str, str]] = set()
        for base, raw_uids in by_base.items():
            uids = sorted(raw_uids)
            if len(uids) < 3:
                continue

            destination_sets = {
                uid: tuple(destination for destination, _ in summaries[uid].blocks)
                for uid in uids
            }
            for uid in uids:
                ranked = sorted(
                    (self._route_distance(problem, destination_sets[uid], destination_sets[other]), other)
                    for other in uids
                    if other != uid
                )
                neighbors = [other for _, other in ranked[: self.neighbor_count]]
                for left, right in itertools.combinations(neighbors, 2):
                    triples.add(tuple(sorted((uid, left, right))))

            # Never miss an immediately capacity-feasible 3->2 opportunity.
            for triple in itertools.combinations(uids, 3):
                if sum(len(summaries[uid].passenger_ids) for uid in triple) <= 38:
                    triples.add(tuple(sorted(triple)))

        return tuple(sorted(triples))

    @staticmethod
    def _route_distance(problem: ProblemData, left: tuple[str, ...], right: tuple[str, ...]) -> float:
        return min(problem.distance(a, b) for a in left for b in right)

    def _best_for_triple(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries,
        triple: tuple[str, str, str],
        route_cache,
        pattern_cache,
    ) -> _TripleCandidate | None:
        flights = [solution.flights[uid] for uid in triple]
        if len({flight.base_airport for flight in flights}) != 1:
            return None
        base = flights[0].base_airport

        blocks = [
            _Block(uid, destination, tuple(block))
            for uid in triple
            for destination, block in summaries[uid].blocks
        ]
        blocks.sort(
            key=lambda block: (
                -block.passenger_count,
                block.destination_id,
                block.original_flight_uid,
                block.passenger_ids,
            )
        )
        if not blocks or len(blocks) > self.max_blocks:
            return None

        passenger_count = sum(block.passenger_count for block in blocks)
        max_seats = max(spec.seats for spec in problem.aircraft_types.values())
        old_aircraft = sum(summaries[uid].aircraft_usage_minutes for uid in triple)
        old_passenger = sum(summaries[uid].passenger_travel_minutes for uid in triple)
        old_fuel = sum(summaries[uid].fuel_kg for uid in triple)

        best: _TripleCandidate | None = None
        min_routes = max(1, (passenger_count + max_seats - 1) // max_seats)
        for route_count in range(min_routes, 4):
            if route_count > len(blocks):
                continue
            for passenger_groups in self._partitions(blocks, route_count, max_seats):
                patterns: list[_OptimizedPattern] = []
                feasible = True
                for group in passenger_groups:
                    pattern = self._optimize_pattern(
                        problem,
                        base,
                        group,
                        route_cache,
                        pattern_cache,
                    )
                    if pattern is None:
                        feasible = False
                        break
                    patterns.append(pattern)
                if not feasible:
                    continue

                candidate = _TripleCandidate(
                    source_uids=triple,
                    base_airport=base,
                    block_count=len(blocks),
                    passenger_count=passenger_count,
                    passenger_groups=passenger_groups,
                    patterns=tuple(patterns),
                    old_aircraft_usage_minutes=old_aircraft,
                    old_passenger_travel_minutes=old_passenger,
                    old_fuel_kg=old_fuel,
                )
                if candidate.aircraft_savings_minutes <= 0:
                    continue
                if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                    best = candidate

        return best

    def _partitions(
        self,
        blocks: list[_Block],
        route_count: int,
        max_seats: int,
    ):
        bins: list[list[_Block]] = [[] for _ in range(route_count)]
        loads = [0] * route_count
        destinations: list[set[str]] = [set() for _ in range(route_count)]

        first = blocks[0]
        bins[0].append(first)
        loads[0] = first.passenger_count
        destinations[0].add(first.destination_id)

        def recurse(index: int, max_used_label: int):
            if index == len(blocks):
                if max_used_label != route_count - 1:
                    return
                groups = tuple(
                    tuple(
                        sorted(
                            pid
                            for block in bucket
                            for pid in block.passenger_ids
                        )
                    )
                    for bucket in bins
                )
                yield groups
                return

            block = blocks[index]
            upper = min(max_used_label + 1, route_count - 1)
            for label in range(upper + 1):
                if loads[label] + block.passenger_count > max_seats:
                    continue
                adds_destination = block.destination_id not in destinations[label]
                if adds_destination and len(destinations[label]) >= self.max_service_destinations:
                    continue

                bins[label].append(block)
                loads[label] += block.passenger_count
                if adds_destination:
                    destinations[label].add(block.destination_id)

                yield from recurse(index + 1, max(max_used_label, label))

                if adds_destination:
                    destinations[label].remove(block.destination_id)
                loads[label] -= block.passenger_count
                bins[label].pop()

        yield from recurse(1, 0)

    @staticmethod
    def _candidate_key(candidate: _TripleCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            candidate.new_flight_count,
            candidate.source_uids,
            tuple(
                tuple(
                    sorted(
                        (
                            pattern.route.aircraft_type,
                            *pattern.route.service_order,
                        )
                    )
                )
                for pattern in candidate.patterns
            ),
        )

    @staticmethod
    def _apply_candidate(
        problem: ProblemData,
        solution: Solution,
        candidate: _TripleCandidate,
    ) -> None:
        for uid in candidate.source_uids:
            for pid in [
                pid
                for pid, assignment in solution.assignments.items()
                if assignment.flight_uid == uid
            ]:
                del solution.assignments[pid]

        ordered_groups = list(zip(candidate.passenger_groups, candidate.patterns, strict=True))
        ordered_groups.sort(
            key=lambda item: (
                item[1].route.aircraft_type,
                item[1].route.service_order,
                item[0],
            )
        )
        for index, uid in enumerate(candidate.source_uids):
            if index < len(ordered_groups):
                passengers, pattern = ordered_groups[index]
                Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
                    problem,
                    solution,
                    uid,
                    passengers,
                    pattern,
                )
            else:
                Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
                    problem,
                    solution,
                    uid,
                    (),
                    None,
                )
