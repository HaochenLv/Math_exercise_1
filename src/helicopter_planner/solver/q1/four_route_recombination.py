from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.local_search import Q1FacilityBlockLocalSearchSolver
from helicopter_planner.solver.q1.route_builder import Q1RouteBuilder
from helicopter_planner.solver.q1.variable_recombination import (
    Q1VariableFlightCountRecombinationSolver,
    VariableRecombinationDecision,
    _VariableCandidate,
)


class _StartingSolver(Protocol):
    def solve(self, problem: ProblemData) -> Solution: ...


class _StaticStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


@dataclass(frozen=True)
class FourRouteDecision:
    iteration: int
    base_airport: str
    source_flight_uids: tuple[str, str, str, str]
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
    lower_neighborhood_polish_moves: int


@dataclass(frozen=True)
class FourRouteResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[FourRouteDecision, ...]
    lower_polish_decisions: tuple[VariableRecombinationDecision, ...]
    converged: bool


class Q1FourRouteRecombinationSolver(Q1VariableFlightCountRecombinationSolver):
    """Q1 V9: exact four-route block recombination.

    Pair/triple neighborhoods V3-V8 may still miss a four-way cyclic exchange.
    V9 therefore destroys four spatially related routes at one base, pools all
    of their current destination blocks, and exactly repartitions those blocks.

    A four-route neighborhood may rebuild into the capacity-minimum number of
    routes through five routes, so both route-count reduction and 4->5 splitting
    are allowed. Every rebuilt route independently re-optimizes aircraft type,
    service order and refueling using the shared route optimizer.

    Candidate quadruples are generated from each route and its nearest route
    neighbors, avoiding the O(n^4) full enumeration while preserving the most
    geometrically plausible exchanges. After every accepted move, V8 is run to
    convergence; V8 in turn polishes all lower neighborhoods V3-V7.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 10,
        polish_max_iterations: int = 200,
        neighbor_count: int = 7,
        max_blocks: int = 9,
        max_extra_routes: int = 1,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1VariableFlightCountRecombinationSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=20,
                polish_max_iterations=polish_max_iterations,
                neighbor_count=6,
                max_blocks=8,
                max_extra_routes=1,
            )
        super().__init__(
            starting_solver=starting_solver,
            route_builder=builder,
            max_service_destinations=max_service_destinations,
            max_iterations=max_iterations,
            polish_max_iterations=polish_max_iterations,
            neighbor_count=neighbor_count,
            max_blocks=max_blocks,
            max_extra_routes=max_extra_routes,
        )

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> FourRouteResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[FourRouteDecision] = []
        lower_polish_decisions: list[VariableRecombinationDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _VariableCandidate | None = None

            for source_uids in self._candidate_quadruples(problem, solution, summaries):
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

            lower = Q1VariableFlightCountRecombinationSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=20,
                polish_max_iterations=self.polish_max_iterations,
                neighbor_count=6,
                max_blocks=8,
                max_extra_routes=1,
            ).solve_with_diagnostics(problem)
            if not lower.converged:
                raise RuntimeError("V9 V8 polish hit max_iterations")
            solution = lower.solution
            lower_polish_decisions.extend(lower.decisions)

            after = evaluate_solution(problem, solution)
            total_savings = before.total_aircraft_usage_minutes - after.total_aircraft_usage_minutes
            if total_savings <= 0:
                raise AssertionError("accepted V9 iteration did not improve aircraft time")

            decisions.append(
                FourRouteDecision(
                    iteration=iteration,
                    base_airport=best.base_airport,
                    source_flight_uids=(
                        best.source_uids[0],
                        best.source_uids[1],
                        best.source_uids[2],
                        best.source_uids[3],
                    ),
                    source_block_count=best.block_count,
                    source_passenger_count=best.passenger_count,
                    old_flight_count=4,
                    new_flight_count=best.new_flight_count,
                    old_aircraft_usage_minutes=best.old_aircraft_usage_minutes,
                    new_aircraft_usage_minutes=best.new_aircraft_usage_minutes,
                    immediate_aircraft_savings_minutes=best.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    immediate_passenger_savings_minutes=best.passenger_savings_minutes,
                    immediate_fuel_savings_kg=best.fuel_savings_kg,
                    new_aircraft_types=tuple(pattern.route.aircraft_type for pattern in best.patterns),
                    new_service_orders=tuple(pattern.route.service_order for pattern in best.patterns),
                    lower_neighborhood_polish_moves=len(lower.decisions),
                )
            )

        return FourRouteResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            lower_polish_decisions=tuple(lower_polish_decisions),
            converged=converged,
        )

    def _candidate_quadruples(self, problem, solution, summaries):
        by_base: dict[str, list[str]] = {}
        for uid, flight in solution.flights.items():
            by_base.setdefault(flight.base_airport, []).append(uid)

        quadruples: set[tuple[str, str, str, str]] = set()
        for raw_uids in by_base.values():
            uids = sorted(raw_uids)
            if len(uids) < 4:
                continue
            destination_sets = {
                uid: tuple(destination for destination, _ in summaries[uid].blocks)
                for uid in uids
            }
            for uid in uids:
                ranked = sorted(
                    (
                        self._route_distance(
                            problem,
                            destination_sets[uid],
                            destination_sets[other],
                        ),
                        other,
                    )
                    for other in uids
                    if other != uid
                )
                neighbors = [other for _, other in ranked[: self.neighbor_count]]
                for others in itertools.combinations(neighbors, 3):
                    quadruples.add(tuple(sorted((uid, *others))))

        return tuple(sorted(quadruples))
