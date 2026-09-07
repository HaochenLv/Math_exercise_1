from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Assignment, FlightPlan, ProblemData, SeaStop, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder
from helicopter_planner.solver.q2.warm_start import Q2WarmStartPacker, classify_q2_request


@dataclass(frozen=True)
class ResidualRouteDecision:
    kind: str
    flight_uid: str
    base_airport: str
    aircraft_type: str
    service_order: tuple[str, ...]
    trip_minutes: int
    passenger_counts: tuple[int, ...]
    separate_direct_minutes: int | None = None
    aircraft_savings_minutes: int | None = None


@dataclass(frozen=True)
class Q2ResidualBaselineResult:
    solution: Solution
    b0a_metrics: SolutionMetrics
    metrics: SolutionMetrics
    b0a_assigned_count: int
    b0a_remaining_count: int
    chain_decisions: tuple[ResidualRouteDecision, ...]
    direct_shuttle_decisions: tuple[ResidualRouteDecision, ...]
    direct_return_decisions: tuple[ResidualRouteDecision, ...]
    returns_repacked_zero_cost: int


@dataclass(frozen=True)
class _ChainCandidate:
    first_edge: tuple[str, str]
    second_edge: tuple[str, str]
    first_count: int
    second_count: int
    route: BuiltRoute
    separate_direct_minutes: int

    @property
    def savings(self) -> int:
        return self.separate_direct_minutes - self.route.trip_minutes


class Q2ResidualFlowChainingBaseline:
    """Q2 B0B: complete residual baseline on top of the frozen Q1+B0A skeleton.

    The baseline deliberately keeps the construction interpretable:

    1. Run exact B0A zero-cost packing on the frozen Q1 milestone.
    2. For remaining sea-to-sea demand, greedily combine adjacent OD flows
       ``u->v`` and ``v->w`` into one sortie ``base->u->v->w->base`` whenever
       this strictly reduces aircraft usage versus serving those passenger
       batches independently. Batch sizes are evaluated at the natural
       12/16/19-seat thresholds, so a chain is not forced onto T3 merely because
       a few passengers can be left for a later route.
    3. Serve residual shuttle demand by minimum-aircraft-time direct OD sorties.
    4. Re-run the exact zero-cost packer to absorb as many remaining return
       passengers as possible into all old and new routes.
    5. Serve any final return demand by direct pickup sorties.

    No passenger transfers are introduced. Every new route is built by the
    common fuel-feasible route builder, so the five-sea-landing and reserve-fuel
    rules remain enforced.
    """

    def __init__(
        self,
        *,
        route_builder: Q1RouteBuilder | None = None,
        packing_time_seconds: float = 30.0,
    ) -> None:
        self.route_builder = route_builder or Q1RouteBuilder()
        self.packing_time_seconds = packing_time_seconds
        self._route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}
        self._best_cache: dict[tuple[tuple[str, ...], int, tuple[str, ...]], BuiltRoute | None] = {}
        self._direct_cost_cache: dict[tuple[tuple[str, ...], int, tuple[str, ...]], int] = {}
        self._next_flight_no = 1

    def solve(self, problem: ProblemData, q1_solution: Solution) -> Q2ResidualBaselineResult:
        b0a = Q2WarmStartPacker(max_time_seconds=self.packing_time_seconds).pack(
            problem, q1_solution
        )
        solution = copy.deepcopy(b0a.solution)
        b0a_assigned_count = len(solution.assignments)
        b0a_remaining_count = len(problem.requests) - b0a_assigned_count

        shuttle_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for pid in sorted(set(problem.requests) - set(solution.assignments)):
            req = problem.requests[pid]
            if classify_q2_request(req) == "shuttle":
                shuttle_groups[(req.origin_id, req.destination_id)].append(pid)

        chain_decisions: list[ResidualRouteDecision] = []
        direct_shuttle_decisions: list[ResidualRouteDecision] = []
        direct_return_decisions: list[ResidualRouteDecision] = []

        while True:
            candidate = self._best_chain(problem, shuttle_groups)
            if candidate is None or candidate.savings <= 0:
                break
            uid = self._add_route(solution, candidate.route)
            first_ids = self._take(shuttle_groups, candidate.first_edge, candidate.first_count)
            second_ids = self._take(shuttle_groups, candidate.second_edge, candidate.second_count)
            self._assign_shuttle_batch(
                solution, candidate.route, uid, candidate.first_edge, first_ids
            )
            self._assign_shuttle_batch(
                solution, candidate.route, uid, candidate.second_edge, second_ids
            )
            chain_decisions.append(
                ResidualRouteDecision(
                    kind="two_edge_chain",
                    flight_uid=uid,
                    base_airport=candidate.route.base_airport,
                    aircraft_type=candidate.route.aircraft_type,
                    service_order=candidate.route.service_order,
                    trip_minutes=candidate.route.trip_minutes,
                    passenger_counts=(len(first_ids), len(second_ids)),
                    separate_direct_minutes=candidate.separate_direct_minutes,
                    aircraft_savings_minutes=candidate.savings,
                )
            )

        for edge in sorted(list(shuttle_groups)):
            while shuttle_groups.get(edge):
                remaining = len(shuttle_groups[edge])
                batch_count, route = self._best_direct_batch(problem, edge, remaining)
                uid = self._add_route(solution, route)
                ids = self._take(shuttle_groups, edge, batch_count)
                self._assign_shuttle_batch(solution, route, uid, edge, ids)
                direct_shuttle_decisions.append(
                    ResidualRouteDecision(
                        kind="direct_shuttle",
                        flight_uid=uid,
                        base_airport=route.base_airport,
                        aircraft_type=route.aircraft_type,
                        service_order=route.service_order,
                        trip_minutes=route.trip_minutes,
                        passenger_counts=(len(ids),),
                    )
                )

        shuttle_check = check_solution(problem, solution, require_all_requests=False)
        if not shuttle_check.ok:
            raise AssertionError(
                "B0B shuttle construction invalid:\n" + "\n".join(shuttle_check.errors)
            )

        repack = Q2WarmStartPacker(max_time_seconds=self.packing_time_seconds).pack(
            problem, solution
        )
        solution = repack.solution
        returns_repacked = repack.packed_by_kind.get("return", 0)

        return_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for pid in sorted(set(problem.requests) - set(solution.assignments)):
            req = problem.requests[pid]
            kind = classify_q2_request(req)
            if kind != "return":
                raise AssertionError(
                    f"B0B unexpected residual request after shuttle phase: {pid} ({kind})"
                )
            return_groups[(req.origin_id, req.destination_id)].append(pid)

        for key in sorted(return_groups):
            origin, destination = key
            allowed_bases = tuple(sorted(AIRPORTS)) if destination == "LAND" else (destination,)
            while return_groups.get(key):
                remaining = len(return_groups[key])
                batch_count, route = self._best_direct_batch(
                    problem,
                    (origin,),
                    remaining,
                    allowed_bases=allowed_bases,
                )
                uid = self._add_route(solution, route)
                ids = return_groups[key][:batch_count]
                del return_groups[key][:batch_count]
                if not return_groups[key]:
                    del return_groups[key]
                pickup_index = route.service_stop_index(origin)
                delivery_index = len(route.sea_stops) + 1
                for pid in ids:
                    solution.assignments[pid] = Assignment(
                        person_id=pid,
                        flight_uid=uid,
                        pickup_index=pickup_index,
                        delivery_index=delivery_index,
                    )
                direct_return_decisions.append(
                    ResidualRouteDecision(
                        kind="direct_return",
                        flight_uid=uid,
                        base_airport=route.base_airport,
                        aircraft_type=route.aircraft_type,
                        service_order=route.service_order,
                        trip_minutes=route.trip_minutes,
                        passenger_counts=(len(ids),),
                    )
                )

        final_check = check_solution(problem, solution, require_all_requests=True)
        if not final_check.ok:
            raise AssertionError(
                "B0B final solution invalid:\n" + "\n".join(final_check.errors)
            )

        return Q2ResidualBaselineResult(
            solution=solution,
            b0a_metrics=b0a.metrics,
            metrics=evaluate_solution(problem, solution),
            b0a_assigned_count=b0a_assigned_count,
            b0a_remaining_count=b0a_remaining_count,
            chain_decisions=tuple(chain_decisions),
            direct_shuttle_decisions=tuple(direct_shuttle_decisions),
            direct_return_decisions=tuple(direct_return_decisions),
            returns_repacked_zero_cost=returns_repacked,
        )

    def _best_chain(
        self,
        problem: ProblemData,
        groups: dict[tuple[str, str], list[str]],
    ) -> _ChainCandidate | None:
        active = {edge: len(ids) for edge, ids in groups.items() if ids}
        if len(active) < 2:
            return None
        outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in active:
            outgoing[edge[0]].append(edge)
            incoming[edge[1]].append(edge)

        best: _ChainCandidate | None = None
        best_key: tuple | None = None
        for hub in sorted(set(incoming) & set(outgoing)):
            for first in sorted(incoming[hub]):
                for second in sorted(outgoing[hub]):
                    if first == second or first[0] == second[1]:
                        continue
                    order = (first[0], hub, second[1])
                    if len(set(order)) != 3:
                        continue
                    for first_count in self._batch_sizes(active[first]):
                        direct_first = self._minimum_direct_cost(
                            problem, first, first_count
                        )
                        for second_count in self._batch_sizes(active[second]):
                            chain = self._best_route(
                                problem,
                                order,
                                max(first_count, second_count),
                                tuple(sorted(AIRPORTS)),
                            )
                            if chain is None:
                                continue
                            direct_second = self._minimum_direct_cost(
                                problem, second, second_count
                            )
                            candidate = _ChainCandidate(
                                first_edge=first,
                                second_edge=second,
                                first_count=first_count,
                                second_count=second_count,
                                route=chain,
                                separate_direct_minutes=direct_first + direct_second,
                            )
                            if candidate.savings <= 0:
                                continue
                            key = (
                                -candidate.savings,
                                candidate.route.trip_minutes,
                                -(first_count + second_count),
                                order,
                                candidate.route.aircraft_type,
                                candidate.route.base_airport,
                            )
                            if best_key is None or key < best_key:
                                best_key = key
                                best = candidate
        return best

    @staticmethod
    def _batch_sizes(count: int) -> tuple[int, ...]:
        return tuple(
            sorted({min(count, seats) for seats in (12, 16, 19) if count > 0})
        )

    def _best_direct_batch(
        self,
        problem: ProblemData,
        order: tuple[str, ...],
        count: int,
        *,
        allowed_bases: tuple[str, ...] | None = None,
    ) -> tuple[int, BuiltRoute]:
        bases = allowed_bases or tuple(sorted(AIRPORTS))
        best: tuple | None = None
        choice: tuple[int, BuiltRoute] | None = None
        for batch in self._batch_sizes(count):
            route = self._best_route(problem, order, batch, bases)
            if route is None:
                continue
            future = (
                self._minimum_direct_cost(problem, order, count - batch, bases)
                if count > batch
                else 0
            )
            key = (
                route.trip_minutes + future,
                route.trip_minutes,
                -batch,
                route.aircraft_type,
                route.base_airport,
            )
            if best is None or key < best:
                best = key
                choice = (batch, route)
        if choice is None:
            raise RuntimeError(
                f"no feasible direct residual route for {order} count={count}"
            )
        return choice

    def _minimum_direct_cost(
        self,
        problem: ProblemData,
        order: tuple[str, ...],
        count: int,
        allowed_bases: tuple[str, ...] | None = None,
    ) -> int:
        if count <= 0:
            return 0
        bases = allowed_bases or tuple(sorted(AIRPORTS))
        cache_key = (order, count, bases)
        cached = self._direct_cost_cache.get(cache_key)
        if cached is not None:
            return cached
        best: int | None = None
        for batch in self._batch_sizes(count):
            route = self._best_route(problem, order, batch, bases)
            if route is None:
                continue
            remainder = count - batch
            cost = route.trip_minutes + self._minimum_direct_cost(
                problem, order, remainder, bases
            )
            if best is None or cost < best:
                best = cost
        if best is None:
            raise RuntimeError(
                f"no feasible direct route cost for {order} count={count}"
            )
        self._direct_cost_cache[cache_key] = best
        return best

    def _best_route(
        self,
        problem: ProblemData,
        order: tuple[str, ...],
        min_seats: int,
        allowed_bases: tuple[str, ...],
    ) -> BuiltRoute | None:
        key = (order, min_seats, allowed_bases)
        if key in self._best_cache:
            return self._best_cache[key]
        best: BuiltRoute | None = None
        best_key: tuple | None = None
        for base in allowed_bases:
            if base not in AIRPORTS:
                continue
            for aircraft_type in ("T1", "T2", "T3"):
                spec = problem.aircraft_types[aircraft_type]
                if spec.seats < min_seats:
                    continue
                route_key = (base, order, aircraft_type)
                if route_key not in self._route_cache:
                    self._route_cache[route_key] = self.route_builder.build(
                        problem, base, order, aircraft_type
                    )
                route = self._route_cache[route_key]
                if route is None:
                    continue
                candidate_key = (
                    route.trip_minutes,
                    route.total_distance_km * spec.fuel_rate_kg_per_km,
                    len(route.sea_stops),
                    aircraft_type,
                    base,
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best = route
        self._best_cache[key] = best
        return best

    def _add_route(self, solution: Solution, route: BuiltRoute) -> str:
        while True:
            uid = f"q2b0b-{self._next_flight_no:04d}"
            self._next_flight_no += 1
            if uid not in solution.flights:
                break
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport=route.base_airport,
            aircraft_type=route.aircraft_type,
            sea_stops=[
                SeaStop(stop.facility_id, stop.refuel) for stop in route.sea_stops
            ],
        )
        return uid

    @staticmethod
    def _take(
        groups: dict[tuple[str, str], list[str]],
        edge: tuple[str, str],
        count: int,
    ) -> list[str]:
        ids = groups[edge][:count]
        del groups[edge][:count]
        if not groups[edge]:
            del groups[edge]
        return ids

    @staticmethod
    def _assign_shuttle_batch(
        solution: Solution,
        route: BuiltRoute,
        flight_uid: str,
        edge: tuple[str, str],
        person_ids: Iterable[str],
    ) -> None:
        pickup = route.service_stop_index(edge[0])
        delivery = route.service_stop_index(edge[1])
        if pickup >= delivery:
            raise AssertionError(
                f"invalid shuttle order {edge} on {route.service_order}"
            )
        for pid in person_ids:
            solution.assignments[pid] = Assignment(
                person_id=pid,
                flight_uid=flight_uid,
                pickup_index=pickup,
                delivery_index=delivery,
            )
