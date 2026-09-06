from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.local_search import (
    LocalSearchDecision,
    Q1FacilityBlockLocalSearchSolver,
    _PairCandidate,
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
class PartialRelocateDecision:
    iteration: int
    source_flight_uid: str
    target_flight_uid: str
    destination_id: str
    moved_passenger_count: int
    immediate_aircraft_savings_minutes: int
    total_iteration_aircraft_savings_minutes: int
    immediate_passenger_savings_minutes: int
    immediate_fuel_savings_kg: float
    source_old_aircraft_type: str
    source_after_partial_aircraft_type: str
    target_old_aircraft_type: str
    target_after_partial_aircraft_type: str
    polish_move_count: int


@dataclass(frozen=True)
class PartialRelocateResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[PartialRelocateDecision, ...]
    polish_decisions: tuple[LocalSearchDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _PartialChoice:
    candidate: _PairCandidate
    source_flight_uid: str
    target_flight_uid: str
    destination_id: str
    moved_passenger_count: int


class Q1PartialPassengerLocalSearchSolver(Q1FacilityBlockLocalSearchSolver):
    """Q1 V4: partial passenger relocation with V3 polishing.

    The starting point is the converged facility-block local-search V3 solution.
    At each iteration, every same-airport flight pair is inspected.  For every
    destination block of size n >= 2, the solver tries moving q=1,...,n-1
    passengers to the other flight.  Passengers in a block are interchangeable
    for Q1, so a deterministic prefix of passenger IDs represents each q.

    Each hypothetical move fully re-optimizes both affected routes: aircraft
    type, service order, and legal refueling are all reconsidered through the V3
    route optimizer.  The best strictly positive aircraft-time saving is
    applied.  After each accepted partial move, V3 whole-block relocate/swap
    local search is run to convergence as a polishing step.  Search stops when
    no improving partial relocation remains from a V3-polished solution.

    Base-airport assignments remain frozen, so explicit-origin and LAND choices
    from earlier stages stay valid.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 5,
        max_iterations: int = 200,
        polish_max_iterations: int = 200,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        if starting_solver is None:
            starting_solver = Q1FacilityBlockLocalSearchSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=polish_max_iterations,
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

    def solve_with_diagnostics(self, problem: ProblemData) -> PartialRelocateResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[PartialRelocateDecision] = []
        polish_decisions: list[LocalSearchDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            best: _PartialChoice | None = None
            uids = sorted(solution.flights)

            for i, left_uid in enumerate(uids):
                left_flight = solution.flights[left_uid]
                left = summaries[left_uid]
                for right_uid in uids[i + 1 :]:
                    right_flight = solution.flights[right_uid]
                    if left_flight.base_airport != right_flight.base_airport:
                        continue
                    right = summaries[right_uid]
                    base = left_flight.base_airport

                    for destination, block in left.blocks:
                        if len(block) < 2:
                            continue
                        for q in range(1, len(block)):
                            moved = tuple(block[:q])
                            moved_set = set(moved)
                            candidate = self._make_candidate(
                                problem,
                                solution,
                                summaries,
                                move_type="partial_relocate",
                                left_uid=left_uid,
                                right_uid=right_uid,
                                left_block_destination=destination,
                                right_block_destination=None,
                                left_passenger_ids=tuple(
                                    pid for pid in left.passenger_ids if pid not in moved_set
                                ),
                                right_passenger_ids=tuple(
                                    sorted((*right.passenger_ids, *moved))
                                ),
                                base_airport=base,
                                route_cache=route_cache,
                                pattern_cache=pattern_cache,
                            )
                            if candidate is not None:
                                best = self._better_choice(
                                    best,
                                    _PartialChoice(
                                        candidate=candidate,
                                        source_flight_uid=left_uid,
                                        target_flight_uid=right_uid,
                                        destination_id=destination,
                                        moved_passenger_count=q,
                                    ),
                                )

                    for destination, block in right.blocks:
                        if len(block) < 2:
                            continue
                        for q in range(1, len(block)):
                            moved = tuple(block[:q])
                            moved_set = set(moved)
                            candidate = self._make_candidate(
                                problem,
                                solution,
                                summaries,
                                move_type="partial_relocate_reverse",
                                left_uid=left_uid,
                                right_uid=right_uid,
                                left_block_destination=destination,
                                right_block_destination=None,
                                left_passenger_ids=tuple(
                                    sorted((*left.passenger_ids, *moved))
                                ),
                                right_passenger_ids=tuple(
                                    pid for pid in right.passenger_ids if pid not in moved_set
                                ),
                                base_airport=base,
                                route_cache=route_cache,
                                pattern_cache=pattern_cache,
                            )
                            if candidate is not None:
                                best = self._better_choice(
                                    best,
                                    _PartialChoice(
                                        candidate=candidate,
                                        source_flight_uid=right_uid,
                                        target_flight_uid=left_uid,
                                        destination_id=destination,
                                        moved_passenger_count=q,
                                    ),
                                )

            if best is None:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            candidate = best.candidate
            source_old_type = solution.flights[best.source_flight_uid].aircraft_type
            target_old_type = solution.flights[best.target_flight_uid].aircraft_type

            self._apply_candidate(problem, solution, candidate)

            if best.source_flight_uid == candidate.left_uid:
                source_pattern = candidate.left_pattern
                target_pattern = candidate.right_pattern
            else:
                source_pattern = candidate.right_pattern
                target_pattern = candidate.left_pattern
            if source_pattern is None or target_pattern is None:
                raise AssertionError("partial relocation must leave both flights nonempty")

            polish = Q1FacilityBlockLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not polish.converged:
                raise RuntimeError("V3 polish hit max_iterations before convergence")
            solution = polish.solution
            polish_decisions.extend(polish.decisions)
            after = evaluate_solution(problem, solution)

            total_iteration_savings = (
                before.total_aircraft_usage_minutes
                - after.total_aircraft_usage_minutes
            )
            if total_iteration_savings <= 0:
                raise AssertionError("accepted V4 iteration did not improve aircraft time")

            decisions.append(
                PartialRelocateDecision(
                    iteration=iteration,
                    source_flight_uid=best.source_flight_uid,
                    target_flight_uid=best.target_flight_uid,
                    destination_id=best.destination_id,
                    moved_passenger_count=best.moved_passenger_count,
                    immediate_aircraft_savings_minutes=candidate.aircraft_savings_minutes,
                    total_iteration_aircraft_savings_minutes=total_iteration_savings,
                    immediate_passenger_savings_minutes=candidate.passenger_savings_minutes,
                    immediate_fuel_savings_kg=candidate.fuel_savings_kg,
                    source_old_aircraft_type=source_old_type,
                    source_after_partial_aircraft_type=source_pattern.route.aircraft_type,
                    target_old_aircraft_type=target_old_type,
                    target_after_partial_aircraft_type=target_pattern.route.aircraft_type,
                    polish_move_count=len(polish.decisions),
                )
            )

        return PartialRelocateResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            polish_decisions=tuple(polish_decisions),
            converged=converged,
        )

    @staticmethod
    def _choice_key(choice: _PartialChoice) -> tuple:
        return (
            Q1FacilityBlockLocalSearchSolver._candidate_key(choice.candidate),
            choice.moved_passenger_count,
            choice.source_flight_uid,
            choice.target_flight_uid,
            choice.destination_id,
        )

    @classmethod
    def _better_choice(
        cls,
        current: _PartialChoice | None,
        candidate: _PartialChoice,
    ) -> _PartialChoice:
        if current is None or cls._choice_key(candidate) < cls._choice_key(current):
            return candidate
        return current
