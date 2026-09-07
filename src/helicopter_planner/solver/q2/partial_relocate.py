from __future__ import annotations

import copy
from dataclasses import dataclass

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q2.tail_elimination import (
    OptimizedPassengerRoute,
    ProfileKey,
    Q2PassengerSetRouteOptimizer,
    Q2ResidualTailEliminationSolver,
    TailEliminationDecision,
)
from helicopter_planner.solver.q2.warm_start import classify_q2_request


@dataclass(frozen=True)
class PartialRelocationDecision:
    donor_flight_uid: str
    host_flight_uid: str
    donor_kind: str
    moved_passenger_count: int
    donor_passenger_count_before: int
    donor_minutes_before: int
    host_minutes_before: int
    donor_minutes_after: int
    host_minutes_after: int
    aircraft_savings_minutes: int
    passenger_minutes_delta: int
    fuel_kg_delta: float
    donor_aircraft_type_after: str
    host_aircraft_type_after: str
    donor_base_airport_after: str
    host_base_airport_after: str
    donor_service_order_after: tuple[str, ...]
    host_service_order_after: tuple[str, ...]


@dataclass(frozen=True)
class Q2PartialRelocationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    partial_relocations: tuple[PartialRelocationDecision, ...]
    unlocked_tail_eliminations: tuple[TailEliminationDecision, ...]
    tail_polish_aircraft_savings_minutes: int
    tail_polish_count: int
    converged: bool


@dataclass(frozen=True)
class _PartialCandidate:
    donor_uid: str
    host_uid: str
    donor_kind: str
    donor_ids: tuple[str, ...]
    host_ids: tuple[str, ...]
    moved_count: int
    donor_before: tuple[int, int, float]
    host_before: tuple[int, int, float]
    donor_after: OptimizedPassengerRoute
    host_after: OptimizedPassengerRoute

    @property
    def savings(self) -> int:
        return (
            self.donor_before[0]
            + self.host_before[0]
            - self.donor_after.aircraft_minutes
            - self.host_after.aircraft_minutes
        )

    @property
    def passenger_delta(self) -> int:
        return (
            self.donor_after.passenger_minutes
            + self.host_after.passenger_minutes
            - self.donor_before[1]
            - self.host_before[1]
        )

    @property
    def fuel_delta(self) -> float:
        return (
            self.donor_after.fuel_kg
            + self.host_after.fuel_kg
            - self.donor_before[2]
            - self.host_before[2]
        )


class Q2PartialPassengerRelocationSolver:
    """Q2 B2: partial passenger relocation followed by B1 tail polish.

    B1 can delete a residual single-OD sortie only when *all* passengers on that
    sortie can be redistributed.  B2 targets the complementary threshold effect:
    move only q passengers (1 <= q < n) from a surviving B0B direct tail into a
    compatible host.  Both donor and host are then reoptimized exactly over base
    airport, T1/T2/T3, precedence-feasible service order, capacity, and legal
    refuelling.  This can trigger discontinuous savings such as T3->T2 or
    T2->T1 on the donor even when complete deletion is impossible.

    After every accepted strict aircraft-time improvement, the full B1
    retype-aware tail elimination is rerun as a polish step because a partial
    move may make another whole residual tail deletable.
    """

    def __init__(
        self,
        *,
        route_optimizer: Q2PassengerSetRouteOptimizer | None = None,
        max_hosts_per_donor: int = 32,
        max_iterations: int = 40,
        tail_max_hosts_per_donor: int = 32,
    ) -> None:
        self.route_optimizer = route_optimizer or Q2PassengerSetRouteOptimizer()
        self.max_hosts_per_donor = max_hosts_per_donor
        self.max_iterations = max_iterations
        self.tail_max_hosts_per_donor = tail_max_hosts_per_donor

    def solve(self, problem: ProblemData, starting_solution: Solution) -> Q2PartialRelocationResult:
        solution = copy.deepcopy(starting_solution)
        start_check = check_solution(problem, solution, require_all_requests=True)
        if not start_check.ok:
            raise ValueError("B2 starting solution invalid:\n" + "\n".join(start_check.errors))
        starting_metrics = evaluate_solution(problem, solution)

        moves: list[PartialRelocationDecision] = []
        unlocked_eliminations: list[TailEliminationDecision] = []
        tail_polish_aircraft_savings = 0
        tail_polish_count = 0
        converged = False

        for _ in range(self.max_iterations):
            candidate = self._best_partial(problem, solution)
            if candidate is None or candidate.savings <= 0:
                converged = True
                break

            decision = self._apply_partial(problem, solution, candidate)
            moves.append(decision)
            check = check_solution(problem, solution, require_all_requests=True)
            if not check.ok:
                raise AssertionError("B2 invalid after partial relocation:\n" + "\n".join(check.errors))

            before_polish = evaluate_solution(problem, solution)
            tail_result = Q2ResidualTailEliminationSolver(
                route_optimizer=self.route_optimizer,
                max_hosts_per_donor=self.tail_max_hosts_per_donor,
            ).solve(problem, solution)
            tail_polish_aircraft_savings += (
                before_polish.total_aircraft_usage_minutes
                - tail_result.metrics.total_aircraft_usage_minutes
            )
            tail_polish_count += 1
            unlocked_eliminations.extend(tail_result.tail_eliminations)
            solution = tail_result.solution

        final_check = check_solution(problem, solution, require_all_requests=True)
        if not final_check.ok:
            raise AssertionError("B2 final solution invalid:\n" + "\n".join(final_check.errors))

        return Q2PartialRelocationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            partial_relocations=tuple(moves),
            unlocked_tail_eliminations=tuple(unlocked_eliminations),
            tail_polish_aircraft_savings_minutes=tail_polish_aircraft_savings,
            tail_polish_count=tail_polish_count,
            converged=converged,
        )

    def _best_partial(self, problem: ProblemData, solution: Solution) -> _PartialCandidate | None:
        best: _PartialCandidate | None = None
        best_key: tuple | None = None

        for donor_uid in sorted(solution.flights):
            # B2 deliberately attacks the direct residual tails created by B0B.
            # Other routes may receive passengers but are not partial donors yet.
            if not donor_uid.startswith("q2b0b-"):
                continue
            donor_ids = tuple(self._person_ids(solution, donor_uid))
            if len(donor_ids) < 2:
                continue
            donor_profile = self.route_optimizer.profile(problem, donor_ids)
            if len(donor_profile) != 1:
                continue
            origin, destination, donor_count = donor_profile[0]
            request_key = (origin, destination)
            donor_kind = classify_q2_request(problem.requests[donor_ids[0]])
            if donor_kind not in {"return", "shuttle"}:
                continue

            donor_before = self._flight_cost(problem, solution, donor_uid)
            donor_sea = {x for x in request_key if x.startswith("F")}

            host_rows: list[tuple[tuple, str, tuple[str, ...], ProfileKey, tuple[int, int, float]]] = []
            for host_uid in sorted(solution.flights):
                if host_uid == donor_uid:
                    continue
                host_ids = tuple(self._person_ids(solution, host_uid))
                if not host_ids:
                    continue
                host_profile = self.route_optimizer.profile(problem, host_ids)
                host_facilities = {
                    location
                    for o, d, _ in host_profile
                    for location in (o, d)
                    if location.startswith("F")
                }
                overlap = len(host_facilities & donor_sea)
                if overlap == 0:
                    continue
                union = host_facilities | donor_sea
                if len(union) > 5:
                    continue
                host_before = self._flight_cost(problem, solution, host_uid)
                rank = (-overlap, len(union), host_before[0], host_uid)
                host_rows.append((rank, host_uid, host_ids, host_profile, host_before))
            host_rows.sort(key=lambda row: row[0])
            host_rows = host_rows[: self.max_hosts_per_donor]
            if not host_rows:
                continue

            donor_after_by_q: dict[int, OptimizedPassengerRoute] = {}
            for q in range(1, donor_count):
                remaining_profile: ProfileKey = ((origin, destination, donor_count - q),)
                donor_after = self.route_optimizer.optimize(problem, remaining_profile)
                if donor_after is not None:
                    donor_after_by_q[q] = donor_after

            for _, host_uid, host_ids, host_profile, host_before in host_rows:
                for q, donor_after in donor_after_by_q.items():
                    combined_profile = self.route_optimizer.add_to_profile(host_profile, request_key, q)
                    host_after = self.route_optimizer.optimize(problem, combined_profile)
                    if host_after is None:
                        continue
                    candidate = _PartialCandidate(
                        donor_uid=donor_uid,
                        host_uid=host_uid,
                        donor_kind=donor_kind,
                        donor_ids=donor_ids,
                        host_ids=host_ids,
                        moved_count=q,
                        donor_before=donor_before,
                        host_before=host_before,
                        donor_after=donor_after,
                        host_after=host_after,
                    )
                    if candidate.savings <= 0:
                        continue
                    key = (
                        -candidate.savings,
                        candidate.passenger_delta,
                        candidate.fuel_delta,
                        candidate.moved_count,
                        candidate.donor_uid,
                        candidate.host_uid,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = candidate

        return best

    def _apply_partial(
        self,
        problem: ProblemData,
        solution: Solution,
        candidate: _PartialCandidate,
    ) -> PartialRelocationDecision:
        moved = list(candidate.donor_ids[: candidate.moved_count])
        remaining = list(candidate.donor_ids[candidate.moved_count :])
        if not moved or not remaining:
            raise AssertionError("B2 partial relocation must leave both donor and moved sets nonempty")

        self.route_optimizer.install(
            problem,
            solution,
            candidate.donor_uid,
            remaining,
            candidate.donor_after,
        )
        self.route_optimizer.install(
            problem,
            solution,
            candidate.host_uid,
            [*candidate.host_ids, *moved],
            candidate.host_after,
        )

        return PartialRelocationDecision(
            donor_flight_uid=candidate.donor_uid,
            host_flight_uid=candidate.host_uid,
            donor_kind=candidate.donor_kind,
            moved_passenger_count=candidate.moved_count,
            donor_passenger_count_before=len(candidate.donor_ids),
            donor_minutes_before=candidate.donor_before[0],
            host_minutes_before=candidate.host_before[0],
            donor_minutes_after=candidate.donor_after.aircraft_minutes,
            host_minutes_after=candidate.host_after.aircraft_minutes,
            aircraft_savings_minutes=candidate.savings,
            passenger_minutes_delta=candidate.passenger_delta,
            fuel_kg_delta=candidate.fuel_delta,
            donor_aircraft_type_after=candidate.donor_after.route.aircraft_type,
            host_aircraft_type_after=candidate.host_after.route.aircraft_type,
            donor_base_airport_after=candidate.donor_after.route.base_airport,
            host_base_airport_after=candidate.host_after.route.base_airport,
            donor_service_order_after=candidate.donor_after.route.service_order,
            host_service_order_after=candidate.host_after.route.service_order,
        )

    @staticmethod
    def _person_ids(solution: Solution, flight_uid: str) -> list[str]:
        return sorted(
            pid
            for pid, assignment in solution.assignments.items()
            if assignment.flight_uid == flight_uid
        )

    @staticmethod
    def _flight_cost(problem: ProblemData, solution: Solution, flight_uid: str) -> tuple[int, int, float]:
        assignments = {
            pid: assignment
            for pid, assignment in solution.assignments.items()
            if assignment.flight_uid == flight_uid
        }
        metrics = evaluate_solution(
            problem,
            Solution(
                flights={flight_uid: solution.flights[flight_uid]},
                assignments=assignments,
            ),
        )
        return (
            metrics.total_aircraft_usage_minutes,
            metrics.total_passenger_travel_minutes,
            metrics.total_fuel_consumption_kg,
        )
