from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
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
class CrossAirportDecision:
    iteration: int
    source_flight_uid: str
    target_flight_uid: str
    source_airport: str
    target_airport: str
    destination_id: str
    moved_passenger_count: int
    immediate_aircraft_savings_minutes: int
    total_iteration_aircraft_savings_minutes: int
    immediate_passenger_savings_minutes: int
    immediate_fuel_savings_kg: float
    source_old_aircraft_type: str
    source_new_aircraft_type: str | None
    target_old_aircraft_type: str
    target_new_aircraft_type: str | None
    block_polish_move_count: int
    partial_polish_move_count: int
    generalized_tail_polish_count: int


@dataclass(frozen=True)
class CrossAirportResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[CrossAirportDecision, ...]
    block_polish_decisions: tuple[LocalSearchDecision, ...]
    partial_polish_decisions: tuple[PartialRelocateDecision, ...]
    generalized_tail_polish_decisions: tuple[GeneralizedTailEliminationDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _CrossCandidate:
    source_uid: str
    target_uid: str
    source_airport: str
    target_airport: str
    destination_id: str
    moved_passenger_ids: tuple[str, ...]
    source_passenger_ids: tuple[str, ...]
    target_passenger_ids: tuple[str, ...]
    source_pattern: _OptimizedPattern | None
    target_pattern: _OptimizedPattern | None
    old_aircraft_usage_minutes: int
    old_passenger_travel_minutes: int
    old_fuel_kg: float

    @property
    def new_aircraft_usage_minutes(self) -> int:
        return sum(
            pattern.route.trip_minutes
            for pattern in (self.source_pattern, self.target_pattern)
            if pattern is not None
        )

    @property
    def new_passenger_travel_minutes(self) -> int:
        return sum(
            pattern.passenger_travel_minutes
            for pattern in (self.source_pattern, self.target_pattern)
            if pattern is not None
        )

    @property
    def new_fuel_kg(self) -> float:
        return sum(
            pattern.fuel_kg
            for pattern in (self.source_pattern, self.target_pattern)
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
        return int(self.source_pattern is not None) + int(self.target_pattern is not None)


class Q1LandCrossAirportLocalSearchSolver(Q1FacilityBlockLocalSearchSolver):
    """Q1 V6: reassign only LAND passengers between different airports.

    The starting point is the converged V5 solution. For every ordered pair of
    flights based at different airports, each LAND subset of a destination
    block is considered for relocation. Fixed-origin A01/A02/A03 passengers are
    never moved across airports.

    For a block containing m movable LAND passengers, q=1,...,m is tested.
    Both affected flights are rebuilt independently at their own airports,
    jointly reconsidering aircraft type, service order, and legal refueling.
    The globally best strictly positive aircraft-time move is accepted.

    After every accepted cross-airport move, the solution is polished by the
    existing same-airport neighborhoods: V3 block relocate/swap, V4 partial
    passenger relocation, and V5 retype-aware generalized tail elimination.
    The process repeats until no improving LAND cross-airport relocation remains.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 50,
        polish_max_iterations: int = 200,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1RetypeAwareTailEliminationSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=max_iterations,
                polish_max_iterations=polish_max_iterations,
            )
        super().__init__(
            starting_solver=starting_solver,
            route_builder=builder,
            max_service_destinations=max_service_destinations,
            max_iterations=max_iterations,
        )
        self.polish_max_iterations = polish_max_iterations

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> CrossAirportResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[CrossAirportDecision] = []
        block_polish_decisions: list[LocalSearchDecision] = []
        partial_polish_decisions: list[PartialRelocateDecision] = []
        generalized_tail_polish_decisions: list[GeneralizedTailEliminationDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _CrossCandidate | None = None
            uids = sorted(solution.flights)

            for source_uid in uids:
                source_flight = solution.flights[source_uid]
                source = summaries[source_uid]
                for destination_id, block in source.blocks:
                    movable = tuple(
                        pid
                        for pid in block
                        if problem.requests[pid].origin_id == "LAND"
                    )
                    if not movable:
                        continue

                    for target_uid in uids:
                        if target_uid == source_uid:
                            continue
                        target_flight = solution.flights[target_uid]
                        if target_flight.base_airport == source_flight.base_airport:
                            continue
                        target = summaries[target_uid]

                        for q in range(1, len(movable) + 1):
                            moved = movable[:q]
                            moved_set = set(moved)
                            source_people = tuple(
                                pid
                                for pid in source.passenger_ids
                                if pid not in moved_set
                            )
                            target_people = tuple(
                                sorted((*target.passenger_ids, *moved))
                            )

                            source_pattern = self._optimize_pattern(
                                problem,
                                source_flight.base_airport,
                                source_people,
                                route_cache,
                                pattern_cache,
                            )
                            if source_people and source_pattern is None:
                                continue
                            target_pattern = self._optimize_pattern(
                                problem,
                                target_flight.base_airport,
                                target_people,
                                route_cache,
                                pattern_cache,
                            )
                            if target_pattern is None:
                                continue

                            candidate = _CrossCandidate(
                                source_uid=source_uid,
                                target_uid=target_uid,
                                source_airport=source_flight.base_airport,
                                target_airport=target_flight.base_airport,
                                destination_id=destination_id,
                                moved_passenger_ids=moved,
                                source_passenger_ids=source_people,
                                target_passenger_ids=target_people,
                                source_pattern=source_pattern,
                                target_pattern=target_pattern,
                                old_aircraft_usage_minutes=(
                                    source.aircraft_usage_minutes
                                    + target.aircraft_usage_minutes
                                ),
                                old_passenger_travel_minutes=(
                                    source.passenger_travel_minutes
                                    + target.passenger_travel_minutes
                                ),
                                old_fuel_kg=source.fuel_kg + target.fuel_kg,
                            )
                            if candidate.aircraft_savings_minutes <= 0:
                                continue
                            if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                                best = candidate

            if best is None:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            source_old_type = solution.flights[best.source_uid].aircraft_type
            target_old_type = solution.flights[best.target_uid].aircraft_type
            self._apply_candidate(problem, solution, best)

            block_polish = Q1FacilityBlockLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not block_polish.converged:
                raise RuntimeError("V6 V3 polish hit max_iterations")
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
                raise RuntimeError("V6 V4 polish hit max_iterations")
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
                raise RuntimeError("V6 V5 polish hit max_iterations")
            solution = generalized_polish.solution
            generalized_tail_polish_decisions.extend(generalized_polish.decisions)
            block_polish_decisions.extend(generalized_polish.block_polish_decisions)
            partial_polish_decisions.extend(generalized_polish.partial_polish_decisions)

            after = evaluate_solution(problem, solution)
            total_savings = (
                before.total_aircraft_usage_minutes
                - after.total_aircraft_usage_minutes
            )
            if total_savings <= 0:
                raise AssertionError("accepted V6 iteration did not improve aircraft time")

            source_new_type = (
                None if best.source_pattern is None
                else best.source_pattern.route.aircraft_type
            )
            target_new_type = (
                None if best.target_pattern is None
                else best.target_pattern.route.aircraft_type
            )
            decisions.append(
                CrossAirportDecision(
                    iteration=iteration,
                    source_flight_uid=best.source_uid,
                    target_flight_uid=best.target_uid,
                    source_airport=best.source_airport,
                    target_airport=best.target_airport,
                    destination_id=best.destination_id,
                    moved_passenger_count=len(best.moved_passenger_ids),
                    immediate_aircraft_savings_minutes=best.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    immediate_passenger_savings_minutes=best.passenger_savings_minutes,
                    immediate_fuel_savings_kg=best.fuel_savings_kg,
                    source_old_aircraft_type=source_old_type,
                    source_new_aircraft_type=source_new_type,
                    target_old_aircraft_type=target_old_type,
                    target_new_aircraft_type=target_new_type,
                    block_polish_move_count=len(block_polish.decisions),
                    partial_polish_move_count=len(partial_polish.decisions),
                    generalized_tail_polish_count=len(generalized_polish.decisions),
                )
            )

        return CrossAirportResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            block_polish_decisions=tuple(block_polish_decisions),
            partial_polish_decisions=tuple(partial_polish_decisions),
            generalized_tail_polish_decisions=tuple(generalized_tail_polish_decisions),
            converged=converged,
        )

    @staticmethod
    def _candidate_key(candidate: _CrossCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            candidate.new_flight_count,
            len(candidate.moved_passenger_ids),
            candidate.source_airport,
            candidate.target_airport,
            candidate.source_uid,
            candidate.target_uid,
            candidate.destination_id,
        )

    @staticmethod
    def _apply_candidate(
        problem: ProblemData,
        solution: Solution,
        candidate: _CrossCandidate,
    ) -> None:
        for uid in (candidate.source_uid, candidate.target_uid):
            for pid in [
                pid
                for pid, assignment in solution.assignments.items()
                if assignment.flight_uid == uid
            ]:
                del solution.assignments[pid]

        Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
            problem,
            solution,
            candidate.source_uid,
            candidate.source_passenger_ids,
            candidate.source_pattern,
        )
        Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
            problem,
            solution,
            candidate.target_uid,
            candidate.target_passenger_ids,
            candidate.target_pattern,
        )
