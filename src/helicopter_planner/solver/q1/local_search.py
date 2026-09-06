from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import Assignment, FlightPlan, ProblemData, Solution
from helicopter_planner.evaluation.metrics import (
    SolutionMetrics,
    evaluate_solution,
    leg_minutes,
    stop_minutes,
)
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder
from helicopter_planner.solver.q1.tail_elimination import Q1TailEliminationSolver


class _StartingSolver(Protocol):
    def solve(self, problem: ProblemData) -> Solution: ...


@dataclass(frozen=True)
class LocalSearchDecision:
    iteration: int
    move_type: str
    left_flight_uid: str
    right_flight_uid: str
    left_block_destination: str
    right_block_destination: str | None
    old_aircraft_usage_minutes: int
    new_aircraft_usage_minutes: int
    aircraft_savings_minutes: int
    passenger_savings_minutes: int
    fuel_savings_kg: float
    left_old_aircraft_type: str
    left_new_aircraft_type: str | None
    right_old_aircraft_type: str
    right_new_aircraft_type: str | None
    left_new_service_order: tuple[str, ...]
    right_new_service_order: tuple[str, ...]
    left_new_passenger_count: int
    right_new_passenger_count: int


@dataclass(frozen=True)
class LocalSearchResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[LocalSearchDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _FlightSummary:
    passenger_ids: tuple[str, ...]
    blocks: tuple[tuple[str, tuple[str, ...]], ...]
    aircraft_usage_minutes: int
    passenger_travel_minutes: int
    fuel_kg: float

    def block_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.blocks)


@dataclass(frozen=True)
class _OptimizedPattern:
    route: BuiltRoute
    passenger_travel_minutes: int
    fuel_kg: float


@dataclass(frozen=True)
class _PairCandidate:
    move_type: str
    left_uid: str
    right_uid: str
    left_block_destination: str
    right_block_destination: str | None
    left_passenger_ids: tuple[str, ...]
    right_passenger_ids: tuple[str, ...]
    left_pattern: _OptimizedPattern | None
    right_pattern: _OptimizedPattern | None
    old_aircraft_usage_minutes: int
    old_passenger_travel_minutes: int
    old_fuel_kg: float

    @property
    def new_aircraft_usage_minutes(self) -> int:
        return sum(
            pattern.route.trip_minutes
            for pattern in (self.left_pattern, self.right_pattern)
            if pattern is not None
        )

    @property
    def new_passenger_travel_minutes(self) -> int:
        return sum(
            pattern.passenger_travel_minutes
            for pattern in (self.left_pattern, self.right_pattern)
            if pattern is not None
        )

    @property
    def new_fuel_kg(self) -> float:
        return sum(
            pattern.fuel_kg
            for pattern in (self.left_pattern, self.right_pattern)
            if pattern is not None
        )

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
        return int(self.left_pattern is not None) + int(self.right_pattern is not None)


class Q1FacilityBlockLocalSearchSolver:
    """Q1 V3: best-improvement local search on facility-demand blocks.

    The starting point is Tail-Elimination V2.  A block is all passengers on a
    flight whose destination is the same offshore facility.  LAND airport
    choices and fixed-origin airport choices are frozen: only flights with the
    same base airport are paired.

    Two neighborhoods are searched exhaustively at every iteration:

    * RELOCATE: move one whole destination block from one flight to another;
    * SWAP: exchange one whole destination block between two flights.

    After every hypothetical move, BOTH affected flights are re-optimized from
    scratch.  The route optimizer enumerates every service order (at most five
    destinations), all available aircraft types, and lets Q1RouteBuilder insert
    legal technical refuel stops.  Thus reorder/retype/refuel optimization is
    embedded in each neighborhood evaluation instead of being a separate move.

    The globally best strictly positive aircraft-time saving is applied.  Ties
    prefer passenger-time saving, then fuel saving, then fewer surviving
    flights and deterministic identifiers.  Search stops at a local optimum for
    these two neighborhoods.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 200,
    ) -> None:
        self.starting_solver = starting_solver or Q1TailEliminationSolver()
        self.route_builder = route_builder or Q1RouteBuilder()
        self.max_service_destinations = max_service_destinations
        self.max_iterations = max_iterations

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> LocalSearchResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[LocalSearchDecision] = []
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}
        pattern_cache: dict[
            tuple[str, tuple[tuple[str, int], ...]], _OptimizedPattern | None
        ] = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _PairCandidate | None = None
            uids = sorted(solution.flights)

            for i, left_uid in enumerate(uids):
                left_flight = solution.flights[left_uid]
                left = summaries[left_uid]
                left_blocks = left.block_map()
                for right_uid in uids[i + 1 :]:
                    right_flight = solution.flights[right_uid]
                    if left_flight.base_airport != right_flight.base_airport:
                        continue
                    right = summaries[right_uid]
                    right_blocks = right.block_map()
                    base = left_flight.base_airport

                    # Relocate one complete facility block left -> right.
                    for destination, block in left.blocks:
                        candidate = self._make_candidate(
                            problem,
                            solution,
                            summaries,
                            move_type="relocate",
                            left_uid=left_uid,
                            right_uid=right_uid,
                            left_block_destination=destination,
                            right_block_destination=None,
                            left_passenger_ids=self._without_block(
                                left.passenger_ids, set(block)
                            ),
                            right_passenger_ids=tuple(
                                sorted((*right.passenger_ids, *block))
                            ),
                            base_airport=base,
                            route_cache=route_cache,
                            pattern_cache=pattern_cache,
                        )
                        best = self._better(best, candidate)

                    # Relocate one complete facility block right -> left.  We
                    # keep the pair identifiers stable and encode the moved
                    # block in right_block_destination.
                    for destination, block in right.blocks:
                        candidate = self._make_candidate(
                            problem,
                            solution,
                            summaries,
                            move_type="relocate_reverse",
                            left_uid=left_uid,
                            right_uid=right_uid,
                            left_block_destination=destination,
                            right_block_destination=None,
                            left_passenger_ids=tuple(
                                sorted((*left.passenger_ids, *block))
                            ),
                            right_passenger_ids=self._without_block(
                                right.passenger_ids, set(block)
                            ),
                            base_airport=base,
                            route_cache=route_cache,
                            pattern_cache=pattern_cache,
                        )
                        best = self._better(best, candidate)

                    # Exchange complete blocks between the two routes.
                    for left_destination, left_block in left.blocks:
                        left_set = set(left_block)
                        for right_destination, right_block in right.blocks:
                            right_set = set(right_block)
                            candidate = self._make_candidate(
                                problem,
                                solution,
                                summaries,
                                move_type="swap",
                                left_uid=left_uid,
                                right_uid=right_uid,
                                left_block_destination=left_destination,
                                right_block_destination=right_destination,
                                left_passenger_ids=tuple(
                                    sorted(
                                        (*self._without_block(left.passenger_ids, left_set),
                                         *right_block)
                                    )
                                ),
                                right_passenger_ids=tuple(
                                    sorted(
                                        (*self._without_block(right.passenger_ids, right_set),
                                         *left_block)
                                    )
                                ),
                                base_airport=base,
                                route_cache=route_cache,
                                pattern_cache=pattern_cache,
                            )
                            best = self._better(best, candidate)

            if best is None:
                converged = True
                break

            left_old_type = solution.flights[best.left_uid].aircraft_type
            right_old_type = solution.flights[best.right_uid].aircraft_type
            self._apply_candidate(problem, solution, best)
            decisions.append(
                LocalSearchDecision(
                    iteration=iteration,
                    move_type=best.move_type,
                    left_flight_uid=best.left_uid,
                    right_flight_uid=best.right_uid,
                    left_block_destination=best.left_block_destination,
                    right_block_destination=best.right_block_destination,
                    old_aircraft_usage_minutes=best.old_aircraft_usage_minutes,
                    new_aircraft_usage_minutes=best.new_aircraft_usage_minutes,
                    aircraft_savings_minutes=best.aircraft_savings_minutes,
                    passenger_savings_minutes=best.passenger_savings_minutes,
                    fuel_savings_kg=best.fuel_savings_kg,
                    left_old_aircraft_type=left_old_type,
                    left_new_aircraft_type=(
                        None if best.left_pattern is None
                        else best.left_pattern.route.aircraft_type
                    ),
                    right_old_aircraft_type=right_old_type,
                    right_new_aircraft_type=(
                        None if best.right_pattern is None
                        else best.right_pattern.route.aircraft_type
                    ),
                    left_new_service_order=(
                        () if best.left_pattern is None
                        else best.left_pattern.route.service_order
                    ),
                    right_new_service_order=(
                        () if best.right_pattern is None
                        else best.right_pattern.route.service_order
                    ),
                    left_new_passenger_count=len(best.left_passenger_ids),
                    right_new_passenger_count=len(best.right_passenger_ids),
                )
            )

        return LocalSearchResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            converged=converged,
        )

    def _make_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries: dict[str, _FlightSummary],
        *,
        move_type: str,
        left_uid: str,
        right_uid: str,
        left_block_destination: str,
        right_block_destination: str | None,
        left_passenger_ids: tuple[str, ...],
        right_passenger_ids: tuple[str, ...],
        base_airport: str,
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None],
        pattern_cache: dict[
            tuple[str, tuple[tuple[str, int], ...]], _OptimizedPattern | None
        ],
    ) -> _PairCandidate | None:
        left_pattern = self._optimize_pattern(
            problem,
            base_airport,
            left_passenger_ids,
            route_cache,
            pattern_cache,
        )
        if left_passenger_ids and left_pattern is None:
            return None
        right_pattern = self._optimize_pattern(
            problem,
            base_airport,
            right_passenger_ids,
            route_cache,
            pattern_cache,
        )
        if right_passenger_ids and right_pattern is None:
            return None

        left = summaries[left_uid]
        right = summaries[right_uid]
        candidate = _PairCandidate(
            move_type=move_type,
            left_uid=left_uid,
            right_uid=right_uid,
            left_block_destination=left_block_destination,
            right_block_destination=right_block_destination,
            left_passenger_ids=left_passenger_ids,
            right_passenger_ids=right_passenger_ids,
            left_pattern=left_pattern,
            right_pattern=right_pattern,
            old_aircraft_usage_minutes=(
                left.aircraft_usage_minutes + right.aircraft_usage_minutes
            ),
            old_passenger_travel_minutes=(
                left.passenger_travel_minutes + right.passenger_travel_minutes
            ),
            old_fuel_kg=left.fuel_kg + right.fuel_kg,
        )
        return candidate if candidate.aircraft_savings_minutes > 0 else None

    def _optimize_pattern(
        self,
        problem: ProblemData,
        base_airport: str,
        passenger_ids: tuple[str, ...],
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None],
        pattern_cache: dict[
            tuple[str, tuple[tuple[str, int], ...]], _OptimizedPattern | None
        ],
    ) -> _OptimizedPattern | None:
        if not passenger_ids:
            return None

        counts: dict[str, int] = {}
        for pid in passenger_ids:
            destination = problem.requests[pid].destination_id
            counts[destination] = counts.get(destination, 0) + 1
        count_key = tuple(sorted(counts.items()))
        cache_key = (base_airport, count_key)
        if cache_key in pattern_cache:
            return pattern_cache[cache_key]
        if len(counts) > self.max_service_destinations:
            pattern_cache[cache_key] = None
            return None

        total_people = len(passenger_ids)
        best: tuple[tuple, _OptimizedPattern] | None = None
        destinations = tuple(sorted(counts))
        for aircraft_type in sorted(problem.aircraft_types):
            spec = problem.aircraft_types[aircraft_type]
            if total_people > spec.seats:
                continue
            for order in itertools.permutations(destinations):
                route_key = (base_airport, order, aircraft_type)
                if route_key not in route_cache:
                    route_cache[route_key] = self.route_builder.build(
                        problem,
                        base_airport,
                        order,
                        aircraft_type,
                    )
                route = route_cache[route_key]
                if route is None:
                    continue
                arrivals = dict(
                    zip(route.service_order, route.service_arrival_minutes, strict=True)
                )
                passenger_minutes = sum(
                    count * arrivals[destination]
                    for destination, count in counts.items()
                )
                fuel = route.total_distance_km * spec.fuel_rate_kg_per_km
                pattern = _OptimizedPattern(
                    route=route,
                    passenger_travel_minutes=passenger_minutes,
                    fuel_kg=fuel,
                )
                objective = (
                    route.trip_minutes,
                    passenger_minutes,
                    round(fuel, 9),
                    aircraft_type,
                    order,
                )
                if best is None or objective < best[0]:
                    best = (objective, pattern)

        pattern_cache[cache_key] = None if best is None else best[1]
        return pattern_cache[cache_key]

    @staticmethod
    def _candidate_key(candidate: _PairCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            candidate.new_flight_count,
            candidate.move_type,
            candidate.left_uid,
            candidate.right_uid,
            candidate.left_block_destination,
            candidate.right_block_destination or "",
        )

    @classmethod
    def _better(
        cls,
        current: _PairCandidate | None,
        candidate: _PairCandidate | None,
    ) -> _PairCandidate | None:
        if candidate is None:
            return current
        if current is None or cls._candidate_key(candidate) < cls._candidate_key(current):
            return candidate
        return current

    @staticmethod
    def _without_block(
        passenger_ids: tuple[str, ...],
        block: set[str],
    ) -> tuple[str, ...]:
        return tuple(pid for pid in passenger_ids if pid not in block)

    @staticmethod
    def _apply_candidate(
        problem: ProblemData,
        solution: Solution,
        candidate: _PairCandidate,
    ) -> None:
        for uid in (candidate.left_uid, candidate.right_uid):
            for pid in [
                pid
                for pid, assignment in solution.assignments.items()
                if assignment.flight_uid == uid
            ]:
                del solution.assignments[pid]

        Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
            problem,
            solution,
            candidate.left_uid,
            candidate.left_passenger_ids,
            candidate.left_pattern,
        )
        Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
            problem,
            solution,
            candidate.right_uid,
            candidate.right_passenger_ids,
            candidate.right_pattern,
        )

    @staticmethod
    def _materialize_affected_flight(
        problem: ProblemData,
        solution: Solution,
        flight_uid: str,
        passenger_ids: tuple[str, ...],
        pattern: _OptimizedPattern | None,
    ) -> None:
        if pattern is None:
            if passenger_ids:
                raise AssertionError("nonempty flight has no optimized pattern")
            del solution.flights[flight_uid]
            return

        old = solution.flights[flight_uid]
        route = pattern.route
        solution.flights[flight_uid] = FlightPlan(
            flight_uid=flight_uid,
            base_airport=old.base_airport,
            aircraft_type=route.aircraft_type,
            sea_stops=list(route.sea_stops),
        )
        stop_index = dict(
            zip(route.service_order, route.service_stop_indices, strict=True)
        )
        for pid in passenger_ids:
            destination = problem.requests[pid].destination_id
            solution.assignments[pid] = Assignment(
                person_id=pid,
                flight_uid=flight_uid,
                pickup_index=0,
                delivery_index=stop_index[destination],
            )

    @staticmethod
    def _summarize_all(
        problem: ProblemData,
        solution: Solution,
    ) -> dict[str, _FlightSummary]:
        people_by_flight: dict[str, list[str]] = {uid: [] for uid in solution.flights}
        for assignment in solution.assignments.values():
            people_by_flight.setdefault(assignment.flight_uid, []).append(assignment.person_id)
        return {
            uid: Q1FacilityBlockLocalSearchSolver._summarize_flight(
                problem,
                solution,
                uid,
                tuple(sorted(people_by_flight[uid])),
            )
            for uid in sorted(solution.flights)
        }

    @staticmethod
    def _summarize_flight(
        problem: ProblemData,
        solution: Solution,
        flight_uid: str,
        passenger_ids: tuple[str, ...],
    ) -> _FlightSummary:
        if not passenger_ids:
            raise ValueError(f"flight {flight_uid} carries no passengers")
        flight = solution.flights[flight_uid]
        spec = problem.aircraft_types[flight.aircraft_type]
        route = flight.full_route()
        distances = [problem.distance(a, b) for a, b in zip(route, route[1:])]
        leg_times = [leg_minutes(distance, spec.speed_kmh) for distance in distances]
        stop_times = [stop_minutes(stop.refuel) for stop in flight.sea_stops]
        aircraft_usage = sum(leg_times) + sum(stop_times)
        fuel = sum(distances) * spec.fuel_rate_kg_per_km
        passenger_time = 0
        blocks: dict[str, list[str]] = {}
        for pid in passenger_ids:
            assignment = solution.assignments[pid]
            passenger_time += sum(
                leg_times[assignment.pickup_index : assignment.delivery_index]
            )
            passenger_time += sum(
                stop_times[
                    assignment.pickup_index : assignment.delivery_index - 1
                ]
            )
            destination = problem.requests[pid].destination_id
            blocks.setdefault(destination, []).append(pid)

        return _FlightSummary(
            passenger_ids=passenger_ids,
            blocks=tuple(
                (destination, tuple(sorted(pids)))
                for destination, pids in sorted(blocks.items())
            ),
            aircraft_usage_minutes=aircraft_usage,
            passenger_travel_minutes=passenger_time,
            fuel_kg=fuel,
        )

    @staticmethod
    def _copy_solution(solution: Solution) -> Solution:
        return Solution(
            flights={
                uid: FlightPlan(
                    flight_uid=flight.flight_uid,
                    base_airport=flight.base_airport,
                    aircraft_type=flight.aircraft_type,
                    sea_stops=list(flight.sea_stops),
                )
                for uid, flight in solution.flights.items()
            },
            assignments=dict(solution.assignments),
        )
