from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import Assignment, FlightPlan, ProblemData, Solution
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
from helicopter_planner.solver.q1.recombination import (
    Q1ThreeRouteRecombinationSolver,
    RecombinationDecision,
    _Block,
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
class VariableRecombinationDecision:
    iteration: int
    base_airport: str
    source_flight_uids: tuple[str, ...]
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
    three_route_polish_count: int


@dataclass(frozen=True)
class VariableRecombinationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[VariableRecombinationDecision, ...]
    block_polish_decisions: tuple[LocalSearchDecision, ...]
    partial_polish_decisions: tuple[PartialRelocateDecision, ...]
    generalized_tail_polish_decisions: tuple[GeneralizedTailEliminationDecision, ...]
    cross_airport_polish_decisions: tuple[CrossAirportDecision, ...]
    three_route_polish_decisions: tuple[RecombinationDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _VariableCandidate:
    source_uids: tuple[str, ...]
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


class Q1VariableFlightCountRecombinationSolver(Q1ThreeRouteRecombinationSolver):
    """Q1 V8: block recombination that may *increase* the number of flights.

    V1-V7 inherited a strong monotonic bias toward fewer flights. That is not
    implied by Q1's objective: one slow multi-stop T3 route can be worse in
    total aircraft time than two shorter T1/T2 routes. V8 explicitly removes
    this bias.

    Every same-airport pair is searched, plus the spatially local triples from
    V7. Current destination blocks are pooled and repartitioned exactly. A pair
    may rebuild into up to three routes; a triple may rebuild into up to four.
    Thus the neighborhood contains 2->3 and 3->4 moves as well as the previous
    route-count-preserving/reducing moves.

    Each rebuilt route independently re-optimizes aircraft type, service order,
    and refueling. Only strict improvements in total aircraft usage are accepted.
    Lower neighborhoods V3-V7 are then polished before V8 searches again.
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
        max_extra_routes: int = 1,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1ThreeRouteRecombinationSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=20,
                polish_max_iterations=polish_max_iterations,
                neighbor_count=neighbor_count,
                max_blocks=max_blocks,
            )
        super().__init__(
            starting_solver=starting_solver,
            route_builder=builder,
            max_service_destinations=max_service_destinations,
            max_iterations=max_iterations,
            polish_max_iterations=polish_max_iterations,
            neighbor_count=neighbor_count,
            max_blocks=max_blocks,
        )
        self.max_extra_routes = max_extra_routes

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> VariableRecombinationResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[VariableRecombinationDecision] = []
        block_polish_decisions: list[LocalSearchDecision] = []
        partial_polish_decisions: list[PartialRelocateDecision] = []
        generalized_tail_polish_decisions: list[GeneralizedTailEliminationDecision] = []
        cross_airport_polish_decisions: list[CrossAirportDecision] = []
        three_route_polish_decisions: list[RecombinationDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _VariableCandidate | None = None

            for source_uids in self._source_sets(problem, solution, summaries):
                candidate = self._best_for_source_set(
                    problem,
                    solution,
                    summaries,
                    source_uids,
                    route_cache,
                    pattern_cache,
                )
                if candidate is None:
                    continue
                if best is None or self._variable_key(candidate) < self._variable_key(best):
                    best = candidate

            if best is None:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            self._apply_variable_candidate(problem, solution, best, iteration)

            block_polish = Q1FacilityBlockLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not block_polish.converged:
                raise RuntimeError("V8 V3 polish hit max_iterations")
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
                raise RuntimeError("V8 V4 polish hit max_iterations")
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
                raise RuntimeError("V8 V5 polish hit max_iterations")
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
                raise RuntimeError("V8 V6 polish hit max_iterations")
            solution = cross_polish.solution
            cross_airport_polish_decisions.extend(cross_polish.decisions)
            block_polish_decisions.extend(cross_polish.block_polish_decisions)
            partial_polish_decisions.extend(cross_polish.partial_polish_decisions)
            generalized_tail_polish_decisions.extend(
                cross_polish.generalized_tail_polish_decisions
            )

            three_polish = Q1ThreeRouteRecombinationSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=20,
                polish_max_iterations=self.polish_max_iterations,
                neighbor_count=self.neighbor_count,
                max_blocks=self.max_blocks,
            ).solve_with_diagnostics(problem)
            if not three_polish.converged:
                raise RuntimeError("V8 V7 polish hit max_iterations")
            solution = three_polish.solution
            three_route_polish_decisions.extend(three_polish.decisions)
            block_polish_decisions.extend(three_polish.block_polish_decisions)
            partial_polish_decisions.extend(three_polish.partial_polish_decisions)
            generalized_tail_polish_decisions.extend(
                three_polish.generalized_tail_polish_decisions
            )
            cross_airport_polish_decisions.extend(
                three_polish.cross_airport_polish_decisions
            )

            after = evaluate_solution(problem, solution)
            total_savings = before.total_aircraft_usage_minutes - after.total_aircraft_usage_minutes
            if total_savings <= 0:
                raise AssertionError("accepted V8 iteration did not improve aircraft time")

            decisions.append(
                VariableRecombinationDecision(
                    iteration=iteration,
                    base_airport=best.base_airport,
                    source_flight_uids=best.source_uids,
                    source_block_count=best.block_count,
                    source_passenger_count=best.passenger_count,
                    old_flight_count=len(best.source_uids),
                    new_flight_count=best.new_flight_count,
                    old_aircraft_usage_minutes=best.old_aircraft_usage_minutes,
                    new_aircraft_usage_minutes=best.new_aircraft_usage_minutes,
                    immediate_aircraft_savings_minutes=best.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    immediate_passenger_savings_minutes=best.passenger_savings_minutes,
                    immediate_fuel_savings_kg=best.fuel_savings_kg,
                    new_aircraft_types=tuple(pattern.route.aircraft_type for pattern in best.patterns),
                    new_service_orders=tuple(pattern.route.service_order for pattern in best.patterns),
                    block_polish_move_count=len(block_polish.decisions),
                    partial_polish_move_count=len(partial_polish.decisions),
                    generalized_tail_polish_count=len(generalized_polish.decisions),
                    cross_airport_polish_count=len(cross_polish.decisions),
                    three_route_polish_count=len(three_polish.decisions),
                )
            )

        return VariableRecombinationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            block_polish_decisions=tuple(block_polish_decisions),
            partial_polish_decisions=tuple(partial_polish_decisions),
            generalized_tail_polish_decisions=tuple(generalized_tail_polish_decisions),
            cross_airport_polish_decisions=tuple(cross_airport_polish_decisions),
            three_route_polish_decisions=tuple(three_route_polish_decisions),
            converged=converged,
        )

    def _source_sets(self, problem, solution, summaries):
        by_base: dict[str, list[str]] = {}
        for uid, flight in solution.flights.items():
            by_base.setdefault(flight.base_airport, []).append(uid)

        source_sets: set[tuple[str, ...]] = set()
        for uids in by_base.values():
            uids = sorted(uids)
            source_sets.update(itertools.combinations(uids, 2))

        source_sets.update(self._candidate_triples(problem, solution, summaries))
        return tuple(sorted(source_sets, key=lambda item: (len(item), item)))

    def _best_for_source_set(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries,
        source_uids: tuple[str, ...],
        route_cache,
        pattern_cache,
    ) -> _VariableCandidate | None:
        flights = [solution.flights[uid] for uid in source_uids]
        if len({flight.base_airport for flight in flights}) != 1:
            return None
        base = flights[0].base_airport

        blocks = [
            _Block(uid, destination, tuple(block))
            for uid in source_uids
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
        min_routes = max(1, (passenger_count + max_seats - 1) // max_seats)
        max_routes = min(
            len(blocks),
            len(source_uids) + self.max_extra_routes,
        )
        if min_routes > max_routes:
            return None

        old_aircraft = sum(summaries[uid].aircraft_usage_minutes for uid in source_uids)
        old_passenger = sum(summaries[uid].passenger_travel_minutes for uid in source_uids)
        old_fuel = sum(summaries[uid].fuel_kg for uid in source_uids)
        best: _VariableCandidate | None = None

        for route_count in range(min_routes, max_routes + 1):
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

                candidate = _VariableCandidate(
                    source_uids=source_uids,
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
                if best is None or self._variable_key(candidate) < self._variable_key(best):
                    best = candidate

        return best

    @staticmethod
    def _variable_key(candidate: _VariableCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            candidate.new_flight_count,
            candidate.source_uids,
            tuple(
                (pattern.route.aircraft_type, pattern.route.service_order)
                for pattern in candidate.patterns
            ),
        )

    def _apply_variable_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        candidate: _VariableCandidate,
        iteration: int,
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

        existing_count = len(candidate.source_uids)
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

        for extra_index, (passengers, pattern) in enumerate(
            ordered_groups[existing_count:],
            start=1,
        ):
            uid = self._fresh_uid(solution, iteration, extra_index)
            route = pattern.route
            solution.flights[uid] = FlightPlan(
                flight_uid=uid,
                base_airport=candidate.base_airport,
                aircraft_type=route.aircraft_type,
                sea_stops=list(route.sea_stops),
            )
            stop_index = dict(
                zip(route.service_order, route.service_stop_indices, strict=True)
            )
            for pid in passengers:
                destination = problem.requests[pid].destination_id
                solution.assignments[pid] = Assignment(
                    person_id=pid,
                    flight_uid=uid,
                    pickup_index=0,
                    delivery_index=stop_index[destination],
                )

    @staticmethod
    def _fresh_uid(solution: Solution, iteration: int, extra_index: int) -> str:
        serial = 0
        while True:
            uid = f"V8_{iteration:03d}_{extra_index:02d}_{serial:03d}"
            if uid not in solution.flights:
                return uid
            serial += 1
