from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
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
class GeneralizedTailEliminationDecision:
    iteration: int
    donor_flight_uid: str
    base_airport: str
    destination_id: str
    donor_passenger_count: int
    recipient_allocations: tuple[tuple[str, int], ...]
    recipient_type_changes: tuple[tuple[str, str, str], ...]
    donor_aircraft_usage_minutes: int
    recipient_increment_aircraft_minutes: int
    immediate_aircraft_savings_minutes: int
    total_iteration_aircraft_savings_minutes: int
    immediate_passenger_savings_minutes: int
    immediate_fuel_savings_kg: float
    block_polish_move_count: int
    partial_polish_move_count: int
    nested_block_polish_move_count: int


@dataclass(frozen=True)
class GeneralizedTailEliminationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[GeneralizedTailEliminationDecision, ...]
    block_polish_decisions: tuple[LocalSearchDecision, ...]
    partial_polish_decisions: tuple[PartialRelocateDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _RecipientOption:
    recipient_uid: str
    inserted_count: int
    existing_passenger_ids: tuple[str, ...]
    pattern: _OptimizedPattern
    old_aircraft_type: str
    delta_aircraft_minutes: int
    delta_passenger_minutes: int
    delta_fuel_kg: float

    @property
    def new_aircraft_type(self) -> str:
        return self.pattern.route.aircraft_type

    @property
    def retyped(self) -> bool:
        return self.old_aircraft_type != self.new_aircraft_type


@dataclass(frozen=True)
class _EliminationCandidate:
    donor_uid: str
    base_airport: str
    destination_id: str
    donor_passenger_ids: tuple[str, ...]
    donor_aircraft_usage_minutes: int
    donor_passenger_travel_minutes: int
    donor_fuel_kg: float
    recipient_options: tuple[_RecipientOption, ...]

    @property
    def recipient_increment_aircraft_minutes(self) -> int:
        return sum(option.delta_aircraft_minutes for option in self.recipient_options)

    @property
    def aircraft_savings_minutes(self) -> int:
        return (
            self.donor_aircraft_usage_minutes
            - self.recipient_increment_aircraft_minutes
        )

    @property
    def passenger_savings_minutes(self) -> int:
        return self.donor_passenger_travel_minutes - sum(
            option.delta_passenger_minutes for option in self.recipient_options
        )

    @property
    def fuel_savings_kg(self) -> float:
        return self.donor_fuel_kg - sum(
            option.delta_fuel_kg for option in self.recipient_options
        )

    @property
    def retyped_recipient_count(self) -> int:
        return sum(option.retyped for option in self.recipient_options)


class Q1RetypeAwareTailEliminationSolver(Q1FacilityBlockLocalSearchSolver):
    """Q1 V5: retype-aware multi-recipient elimination of tail flights.

    The starting point is the converged V4 partial-passenger local-search
    solution. Every single-destination flight is considered as a donor. Its
    passengers may be split over several flights at the SAME base airport.

    Unlike V2, a recipient is not restricted to seats left under its current
    aircraft type. For every possible inserted passenger count, the recipient
    is rebuilt from scratch: all aircraft types, all service orders, and legal
    technical refueling patterns are reconsidered. A dynamic program then
    finds the lexicographically best set of recipient insertions that absorbs
    every donor passenger.

    The best strictly positive aircraft-time elimination is applied. The
    resulting solution is polished first by V3 whole-block relocate/swap and
    then by V4 partial relocation (which itself invokes V3 after accepted
    moves). The process repeats until no positive generalized elimination
    remains. Base-airport assignments stay frozen.
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
            starting_solver = Q1PartialPassengerLocalSearchSolver(
                route_builder=builder,
                max_service_destinations=max_service_destinations,
                max_iterations=polish_max_iterations,
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

    def solve_with_diagnostics(
        self,
        problem: ProblemData,
    ) -> GeneralizedTailEliminationResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[GeneralizedTailEliminationDecision] = []
        block_polish_decisions: list[LocalSearchDecision] = []
        partial_polish_decisions: list[PartialRelocateDecision] = []
        route_cache = {}
        pattern_cache = {}
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summarize_all(problem, solution)
            candidates: list[_EliminationCandidate] = []

            for donor_uid in sorted(solution.flights):
                donor = summaries[donor_uid]
                if len(donor.blocks) != 1:
                    continue
                candidate = self._best_elimination_candidate(
                    problem,
                    solution,
                    summaries,
                    donor_uid,
                    route_cache,
                    pattern_cache,
                )
                if (
                    candidate is not None
                    and candidate.aircraft_savings_minutes > 0
                ):
                    candidates.append(candidate)

            if not candidates:
                converged = True
                break

            chosen = min(candidates, key=self._candidate_key)
            before = evaluate_solution(problem, solution)
            self._apply_elimination(problem, solution, chosen)

            block_polish = Q1FacilityBlockLocalSearchSolver(
                starting_solver=_StaticStartingSolver(solution),
                route_builder=self.route_builder,
                max_service_destinations=self.max_service_destinations,
                max_iterations=self.polish_max_iterations,
            ).solve_with_diagnostics(problem)
            if not block_polish.converged:
                raise RuntimeError("V3 block polish hit max_iterations")
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
                raise RuntimeError("V4 partial polish hit max_iterations")
            solution = partial_polish.solution
            partial_polish_decisions.extend(partial_polish.decisions)
            block_polish_decisions.extend(partial_polish.polish_decisions)

            after = evaluate_solution(problem, solution)
            total_iteration_savings = (
                before.total_aircraft_usage_minutes
                - after.total_aircraft_usage_minutes
            )
            if total_iteration_savings <= 0:
                raise AssertionError(
                    "accepted V5 iteration did not improve aircraft time"
                )

            decisions.append(
                GeneralizedTailEliminationDecision(
                    iteration=iteration,
                    donor_flight_uid=chosen.donor_uid,
                    base_airport=chosen.base_airport,
                    destination_id=chosen.destination_id,
                    donor_passenger_count=len(chosen.donor_passenger_ids),
                    recipient_allocations=tuple(
                        (option.recipient_uid, option.inserted_count)
                        for option in chosen.recipient_options
                    ),
                    recipient_type_changes=tuple(
                        (
                            option.recipient_uid,
                            option.old_aircraft_type,
                            option.new_aircraft_type,
                        )
                        for option in chosen.recipient_options
                        if option.retyped
                    ),
                    donor_aircraft_usage_minutes=(
                        chosen.donor_aircraft_usage_minutes
                    ),
                    recipient_increment_aircraft_minutes=(
                        chosen.recipient_increment_aircraft_minutes
                    ),
                    immediate_aircraft_savings_minutes=(
                        chosen.aircraft_savings_minutes
                    ),
                    total_iteration_aircraft_savings_minutes=(
                        total_iteration_savings
                    ),
                    immediate_passenger_savings_minutes=(
                        chosen.passenger_savings_minutes
                    ),
                    immediate_fuel_savings_kg=chosen.fuel_savings_kg,
                    block_polish_move_count=len(block_polish.decisions),
                    partial_polish_move_count=len(partial_polish.decisions),
                    nested_block_polish_move_count=len(
                        partial_polish.polish_decisions
                    ),
                )
            )

        return GeneralizedTailEliminationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            block_polish_decisions=tuple(block_polish_decisions),
            partial_polish_decisions=tuple(partial_polish_decisions),
            converged=converged,
        )

    def _best_elimination_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries,
        donor_uid: str,
        route_cache,
        pattern_cache,
    ) -> _EliminationCandidate | None:
        donor_flight = solution.flights[donor_uid]
        donor = summaries[donor_uid]
        destination_id, donor_passenger_ids = donor.blocks[0]
        need = len(donor_passenger_ids)
        max_seats = max(spec.seats for spec in problem.aircraft_types.values())

        recipient_options: list[tuple[str, tuple[_RecipientOption, ...]]] = []
        for recipient_uid in sorted(solution.flights):
            if recipient_uid == donor_uid:
                continue
            recipient_flight = solution.flights[recipient_uid]
            if recipient_flight.base_airport != donor_flight.base_airport:
                continue

            recipient = summaries[recipient_uid]
            max_insert = min(
                need,
                max_seats - len(recipient.passenger_ids),
            )
            if max_insert <= 0:
                continue

            options: list[_RecipientOption] = []
            for q in range(1, max_insert + 1):
                combined = tuple(
                    sorted(
                        (
                            *recipient.passenger_ids,
                            *donor_passenger_ids[:q],
                        )
                    )
                )
                pattern = self._optimize_pattern(
                    problem,
                    donor_flight.base_airport,
                    combined,
                    route_cache,
                    pattern_cache,
                )
                if pattern is None:
                    continue
                options.append(
                    _RecipientOption(
                        recipient_uid=recipient_uid,
                        inserted_count=q,
                        existing_passenger_ids=recipient.passenger_ids,
                        pattern=pattern,
                        old_aircraft_type=recipient_flight.aircraft_type,
                        delta_aircraft_minutes=(
                            pattern.route.trip_minutes
                            - recipient.aircraft_usage_minutes
                        ),
                        delta_passenger_minutes=(
                            pattern.passenger_travel_minutes
                            - recipient.passenger_travel_minutes
                        ),
                        delta_fuel_kg=pattern.fuel_kg - recipient.fuel_kg,
                    )
                )

            if options:
                recipient_options.append(
                    (recipient_uid, tuple(options))
                )

        if (
            sum(
                max(option.inserted_count for option in options)
                for _, options in recipient_options
            )
            < need
        ):
            return None

        # delivered -> (lexicographic cumulative cost, chosen options)
        dp: dict[int, tuple[tuple, tuple[_RecipientOption, ...]]] = {
            0: ((0, 0, 0.0, 0, 0, ()), ())
        }
        for _, options in recipient_options:
            next_dp = dict(dp)
            for delivered, (cost, selected) in dp.items():
                for option in options:
                    new_delivered = delivered + option.inserted_count
                    if new_delivered > need:
                        continue
                    new_selected = selected + (option,)
                    signature = tuple(
                        (item.recipient_uid, item.inserted_count)
                        for item in new_selected
                    )
                    new_cost = (
                        cost[0] + option.delta_aircraft_minutes,
                        cost[1] + option.delta_passenger_minutes,
                        round(cost[2] + option.delta_fuel_kg, 9),
                        cost[3] + 1,
                        cost[4] + int(option.retyped),
                        signature,
                    )
                    previous = next_dp.get(new_delivered)
                    if previous is None or new_cost < previous[0]:
                        next_dp[new_delivered] = (new_cost, new_selected)
            dp = next_dp

        if need not in dp:
            return None

        _, selected = dp[need]
        selected = tuple(
            sorted(selected, key=lambda option: option.recipient_uid)
        )
        candidate = _EliminationCandidate(
            donor_uid=donor_uid,
            base_airport=donor_flight.base_airport,
            destination_id=destination_id,
            donor_passenger_ids=donor_passenger_ids,
            donor_aircraft_usage_minutes=donor.aircraft_usage_minutes,
            donor_passenger_travel_minutes=donor.passenger_travel_minutes,
            donor_fuel_kg=donor.fuel_kg,
            recipient_options=selected,
        )
        return candidate if candidate.aircraft_savings_minutes > 0 else None

    @staticmethod
    def _candidate_key(candidate: _EliminationCandidate) -> tuple:
        return (
            -candidate.aircraft_savings_minutes,
            -candidate.passenger_savings_minutes,
            -round(candidate.fuel_savings_kg, 9),
            len(candidate.recipient_options),
            candidate.retyped_recipient_count,
            candidate.donor_uid,
            tuple(
                (option.recipient_uid, option.inserted_count)
                for option in candidate.recipient_options
            ),
        )

    @staticmethod
    def _apply_elimination(
        problem: ProblemData,
        solution: Solution,
        candidate: _EliminationCandidate,
    ) -> None:
        donor_people = list(candidate.donor_passenger_ids)
        del solution.flights[candidate.donor_uid]
        for pid in donor_people:
            del solution.assignments[pid]

        cursor = 0
        for option in candidate.recipient_options:
            chunk = tuple(
                donor_people[cursor : cursor + option.inserted_count]
            )
            cursor += option.inserted_count
            new_people = tuple(
                sorted((*option.existing_passenger_ids, *chunk))
            )
            Q1FacilityBlockLocalSearchSolver._materialize_affected_flight(
                problem,
                solution,
                option.recipient_uid,
                new_people,
                option.pattern,
            )

        if cursor != len(donor_people):
            raise AssertionError(
                "generalized tail elimination did not reassign every passenger"
            )
