from __future__ import annotations

import copy
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Assignment, FlightPlan, ProblemData, SeaStop, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution, leg_minutes, stop_minutes
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder
from helicopter_planner.solver.q2.warm_start import classify_q2_request

RequestKey = tuple[str, str]
ProfileKey = tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class OptimizedPassengerRoute:
    route: BuiltRoute
    aircraft_minutes: int
    passenger_minutes: int
    fuel_kg: float


@dataclass(frozen=True)
class SingleRoutePolishDecision:
    flight_uid: str
    before_minutes: int
    after_minutes: int
    aircraft_savings_minutes: int
    before_aircraft_type: str
    after_aircraft_type: str
    before_base_airport: str
    after_base_airport: str
    after_service_order: tuple[str, ...]


@dataclass(frozen=True)
class TailRecipientDecision:
    flight_uid: str
    passenger_count: int
    added_aircraft_minutes: int
    after_aircraft_type: str
    after_base_airport: str
    after_service_order: tuple[str, ...]


@dataclass(frozen=True)
class TailEliminationDecision:
    donor_flight_uid: str
    donor_kind: str
    donor_passenger_count: int
    donor_minutes: int
    added_recipient_minutes: int
    aircraft_savings_minutes: int
    passenger_minutes_delta: int
    fuel_kg_delta: float
    recipients: tuple[TailRecipientDecision, ...]


@dataclass(frozen=True)
class Q2ResidualTailEliminationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics_after_single_route_polish: SolutionMetrics
    metrics: SolutionMetrics
    single_route_polish: tuple[SingleRoutePolishDecision, ...]
    tail_eliminations: tuple[TailEliminationDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _InsertionOption:
    host_uid: str
    count: int
    optimized: OptimizedPassengerRoute
    added_aircraft_minutes: int
    added_passenger_minutes: int
    added_fuel_kg: float


@dataclass(frozen=True)
class _EliminationCandidate:
    donor_uid: str
    donor_kind: str
    donor_ids: tuple[str, ...]
    donor_minutes: int
    donor_passenger_minutes: int
    donor_fuel_kg: float
    options: tuple[_InsertionOption, ...]

    @property
    def added_aircraft_minutes(self) -> int:
        return sum(option.added_aircraft_minutes for option in self.options)

    @property
    def savings(self) -> int:
        return self.donor_minutes - self.added_aircraft_minutes

    @property
    def passenger_delta(self) -> int:
        return sum(option.added_passenger_minutes for option in self.options) - self.donor_passenger_minutes

    @property
    def fuel_delta(self) -> float:
        return sum(option.added_fuel_kg for option in self.options) - self.donor_fuel_kg


class Q2PassengerSetRouteOptimizer:
    """Exact single-sortie optimizer for a fixed multiset of Q2 requests.

    There are at most five distinct required sea facilities in a feasible sortie.
    We enumerate all precedence-feasible service orders, all legal base airports,
    and T1/T2/T3.  The common route builder supplies the minimum-time legal
    refuelling pattern for every fixed (base, order, type).  Capacity is checked
    leg-by-leg using the complete pickup/delivery profile.
    """

    def __init__(self, route_builder: Q1RouteBuilder | None = None) -> None:
        self.route_builder = route_builder or Q1RouteBuilder()
        self._built_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}
        self._profile_cache: dict[ProfileKey, OptimizedPassengerRoute | None] = {}

    @staticmethod
    def profile(problem: ProblemData, person_ids: list[str] | tuple[str, ...]) -> ProfileKey:
        counts = Counter(
            (problem.requests[pid].origin_id, problem.requests[pid].destination_id)
            for pid in person_ids
        )
        return tuple(sorted((origin, destination, count) for (origin, destination), count in counts.items()))

    @staticmethod
    def add_to_profile(profile: ProfileKey, request_key: RequestKey, count: int) -> ProfileKey:
        counts = {(origin, destination): n for origin, destination, n in profile}
        counts[request_key] = counts.get(request_key, 0) + count
        return tuple(sorted((origin, destination, n) for (origin, destination), n in counts.items() if n))

    def optimize(self, problem: ProblemData, profile: ProfileKey) -> OptimizedPassengerRoute | None:
        if profile in self._profile_cache:
            return self._profile_cache[profile]

        explicit_airports: set[str] = set()
        facilities: set[str] = set()
        precedence: set[tuple[str, str]] = set()
        for origin, destination, count in profile:
            if count <= 0:
                continue
            for location in (origin, destination):
                if location in AIRPORTS:
                    explicit_airports.add(location)
                elif location.startswith("F"):
                    facilities.add(location)
            if origin.startswith("F") and destination.startswith("F"):
                precedence.add((origin, destination))

        if len(explicit_airports) > 1 or not facilities or len(facilities) > 5:
            self._profile_cache[profile] = None
            return None
        bases = tuple(explicit_airports) if explicit_airports else tuple(sorted(AIRPORTS))

        best: OptimizedPassengerRoute | None = None
        best_key: tuple | None = None
        for order in itertools.permutations(sorted(facilities)):
            positions = {facility: index for index, facility in enumerate(order)}
            if any(positions[a] >= positions[b] for a, b in precedence):
                continue
            for base in bases:
                for aircraft_type in ("T1", "T2", "T3"):
                    cache_key = (base, order, aircraft_type)
                    if cache_key not in self._built_cache:
                        self._built_cache[cache_key] = self.route_builder.build(
                            problem, base, order, aircraft_type
                        )
                    route = self._built_cache[cache_key]
                    if route is None:
                        continue
                    scored = self._score_profile(problem, profile, route)
                    if scored is None:
                        continue
                    key = (
                        scored.aircraft_minutes,
                        scored.passenger_minutes,
                        scored.fuel_kg,
                        len(route.sea_stops),
                        aircraft_type,
                        base,
                        order,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = scored

        self._profile_cache[profile] = best
        return best

    @staticmethod
    def _score_profile(
        problem: ProblemData,
        profile: ProfileKey,
        route: BuiltRoute,
    ) -> OptimizedPassengerRoute | None:
        spec = problem.aircraft_types[route.aircraft_type]
        full_route = [route.base_airport, *(stop.facility_id for stop in route.sea_stops), route.base_airport]
        distances = [problem.distance(a, b) for a, b in zip(full_route, full_route[1:])]
        flight_mins = [leg_minutes(distance, spec.speed_kmh) for distance in distances]
        stop_mins = [stop_minutes(stop.refuel) for stop in route.sea_stops]
        leg_loads = [0] * len(distances)
        passenger_minutes = 0

        for origin, destination, count in profile:
            if origin == "LAND" or origin in AIRPORTS:
                pickup = 0
            else:
                pickup = route.service_stop_index(origin)
            if destination == "LAND" or destination in AIRPORTS:
                delivery = len(route.sea_stops) + 1
            else:
                delivery = route.service_stop_index(destination)
            if pickup >= delivery:
                return None
            for leg in range(pickup, delivery):
                leg_loads[leg] += count
            passenger_minutes += count * (
                sum(flight_mins[pickup:delivery])
                + sum(stop_mins[pickup : delivery - 1])
            )

        if any(load > spec.seats for load in leg_loads):
            return None
        return OptimizedPassengerRoute(
            route=route,
            aircraft_minutes=route.trip_minutes,
            passenger_minutes=passenger_minutes,
            fuel_kg=route.total_distance_km * spec.fuel_rate_kg_per_km,
        )

    def install(
        self,
        problem: ProblemData,
        solution: Solution,
        flight_uid: str,
        person_ids: list[str] | tuple[str, ...],
        optimized: OptimizedPassengerRoute,
    ) -> None:
        route = optimized.route
        solution.flights[flight_uid] = FlightPlan(
            flight_uid=flight_uid,
            base_airport=route.base_airport,
            aircraft_type=route.aircraft_type,
            sea_stops=[SeaStop(stop.facility_id, stop.refuel) for stop in route.sea_stops],
        )
        for pid in person_ids:
            req = problem.requests[pid]
            pickup = 0 if req.origin_id == "LAND" or req.origin_id in AIRPORTS else route.service_stop_index(req.origin_id)
            delivery = (
                len(route.sea_stops) + 1
                if req.destination_id == "LAND" or req.destination_id in AIRPORTS
                else route.service_stop_index(req.destination_id)
            )
            solution.assignments[pid] = Assignment(
                person_id=pid,
                flight_uid=flight_uid,
                pickup_index=pickup,
                delivery_index=delivery,
            )


class Q2ResidualTailEliminationSolver:
    """Q2 B1: single-route exact polish plus retype-aware tail elimination.

    Direct residual shuttle/return sorties created by B0B are treated as donors.
    A donor may be deleted by splitting its symmetric OD group over several host
    flights.  Each affected host is reoptimized exactly over base airport,
    aircraft type, service order and legal refuelling.  A tiny DP chooses at
    most one insertion quantity per host and accepts only a strict reduction in
    total aircraft usage time.
    """

    def __init__(
        self,
        *,
        route_optimizer: Q2PassengerSetRouteOptimizer | None = None,
        max_hosts_per_donor: int = 24,
        max_iterations: int = 30,
    ) -> None:
        self.route_optimizer = route_optimizer or Q2PassengerSetRouteOptimizer()
        self.max_hosts_per_donor = max_hosts_per_donor
        self.max_iterations = max_iterations

    def solve(self, problem: ProblemData, starting_solution: Solution) -> Q2ResidualTailEliminationResult:
        solution = copy.deepcopy(starting_solution)
        start_check = check_solution(problem, solution, require_all_requests=True)
        if not start_check.ok:
            raise ValueError("B1 starting solution invalid:\n" + "\n".join(start_check.errors))
        starting_metrics = evaluate_solution(problem, solution)

        donor_uids = {
            uid
            for uid in solution.flights
            if uid.startswith("q2b0b-") and len(self._profile_for_flight(problem, solution, uid)) == 1
        }

        polish_decisions = self._single_route_polish(problem, solution)
        metrics_after_polish = evaluate_solution(problem, solution)

        eliminations: list[TailEliminationDecision] = []
        converged = False
        for _ in range(self.max_iterations):
            candidate = self._best_elimination(problem, solution, donor_uids)
            if candidate is None or candidate.savings <= 0:
                converged = True
                break
            decision = self._apply_elimination(problem, solution, candidate)
            eliminations.append(decision)
            donor_uids.discard(candidate.donor_uid)
            check = check_solution(problem, solution, require_all_requests=True)
            if not check.ok:
                raise AssertionError("B1 invalid after donor elimination:\n" + "\n".join(check.errors))

        final_check = check_solution(problem, solution, require_all_requests=True)
        if not final_check.ok:
            raise AssertionError("B1 final solution invalid:\n" + "\n".join(final_check.errors))
        return Q2ResidualTailEliminationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics_after_single_route_polish=metrics_after_polish,
            metrics=evaluate_solution(problem, solution),
            single_route_polish=tuple(polish_decisions),
            tail_eliminations=tuple(eliminations),
            converged=converged,
        )

    def _single_route_polish(self, problem: ProblemData, solution: Solution) -> list[SingleRoutePolishDecision]:
        decisions: list[SingleRoutePolishDecision] = []
        for uid in sorted(solution.flights):
            person_ids = self._person_ids(solution, uid)
            if not person_ids:
                continue
            before = self._flight_cost(problem, solution, uid)
            before_flight = solution.flights[uid]
            profile = self.route_optimizer.profile(problem, person_ids)
            optimized = self.route_optimizer.optimize(problem, profile)
            if optimized is None:
                continue
            candidate_key = (optimized.aircraft_minutes, optimized.passenger_minutes, optimized.fuel_kg)
            before_key = before
            if candidate_key >= before_key:
                continue
            self.route_optimizer.install(problem, solution, uid, person_ids, optimized)
            decisions.append(
                SingleRoutePolishDecision(
                    flight_uid=uid,
                    before_minutes=before[0],
                    after_minutes=optimized.aircraft_minutes,
                    aircraft_savings_minutes=before[0] - optimized.aircraft_minutes,
                    before_aircraft_type=before_flight.aircraft_type,
                    after_aircraft_type=optimized.route.aircraft_type,
                    before_base_airport=before_flight.base_airport,
                    after_base_airport=optimized.route.base_airport,
                    after_service_order=optimized.route.service_order,
                )
            )
        return decisions

    def _best_elimination(
        self,
        problem: ProblemData,
        solution: Solution,
        donor_uids: set[str],
    ) -> _EliminationCandidate | None:
        best: _EliminationCandidate | None = None
        best_key: tuple | None = None
        for donor_uid in sorted(donor_uids):
            if donor_uid not in solution.flights:
                continue
            candidate = self._evaluate_donor(problem, solution, donor_uid)
            if candidate is None or candidate.savings <= 0:
                continue
            key = (
                -candidate.savings,
                candidate.passenger_delta,
                candidate.fuel_delta,
                len(candidate.options),
                donor_uid,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = candidate
        return best

    def _evaluate_donor(
        self,
        problem: ProblemData,
        solution: Solution,
        donor_uid: str,
    ) -> _EliminationCandidate | None:
        donor_ids = tuple(self._person_ids(solution, donor_uid))
        if not donor_ids:
            return None
        donor_profile = self.route_optimizer.profile(problem, donor_ids)
        if len(donor_profile) != 1:
            return None
        origin, destination, donor_count = donor_profile[0]
        request_key = (origin, destination)
        donor_kind = classify_q2_request(problem.requests[donor_ids[0]])
        donor_cost = self._flight_cost(problem, solution, donor_uid)
        donor_sea = {location for location in request_key if location.startswith("F")}

        host_rows: list[tuple[tuple, str, ProfileKey, tuple[int, int, float]]] = []
        for host_uid in sorted(solution.flights):
            if host_uid == donor_uid:
                continue
            host_ids = self._person_ids(solution, host_uid)
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
            host_cost = self._flight_cost(problem, solution, host_uid)
            rank = (-overlap, len(union), host_cost[0], host_uid)
            host_rows.append((rank, host_uid, host_profile, host_cost))
        host_rows.sort(key=lambda row: row[0])
        host_rows = host_rows[: self.max_hosts_per_donor]
        if not host_rows:
            return None

        options_by_host: dict[str, list[_InsertionOption]] = defaultdict(list)
        for _, host_uid, host_profile, host_cost in host_rows:
            for count in range(1, donor_count + 1):
                combined_profile = self.route_optimizer.add_to_profile(host_profile, request_key, count)
                optimized = self.route_optimizer.optimize(problem, combined_profile)
                if optimized is None:
                    continue
                added_aircraft = optimized.aircraft_minutes - host_cost[0]
                if added_aircraft >= donor_cost[0]:
                    continue
                options_by_host[host_uid].append(
                    _InsertionOption(
                        host_uid=host_uid,
                        count=count,
                        optimized=optimized,
                        added_aircraft_minutes=added_aircraft,
                        added_passenger_minutes=optimized.passenger_minutes - host_cost[1],
                        added_fuel_kg=optimized.fuel_kg - host_cost[2],
                    )
                )

        # quantity -> ((aircraft, passenger, fuel, hosts, deterministic choices), options)
        states: dict[int, tuple[tuple, tuple[_InsertionOption, ...]]] = {0: ((0, 0, 0.0, 0, ()), ())}
        for host_uid in sorted(options_by_host):
            new_states = dict(states)
            for quantity, (_, chosen) in states.items():
                for option in options_by_host[host_uid]:
                    new_quantity = quantity + option.count
                    if new_quantity > donor_count:
                        continue
                    new_chosen = chosen + (option,)
                    key = (
                        sum(x.added_aircraft_minutes for x in new_chosen),
                        sum(x.added_passenger_minutes for x in new_chosen),
                        sum(x.added_fuel_kg for x in new_chosen),
                        len(new_chosen),
                        tuple((x.host_uid, x.count) for x in new_chosen),
                    )
                    previous = new_states.get(new_quantity)
                    if previous is None or key < previous[0]:
                        new_states[new_quantity] = (key, new_chosen)
            states = new_states

        if donor_count not in states:
            return None
        _, chosen = states[donor_count]
        candidate = _EliminationCandidate(
            donor_uid=donor_uid,
            donor_kind=donor_kind,
            donor_ids=donor_ids,
            donor_minutes=donor_cost[0],
            donor_passenger_minutes=donor_cost[1],
            donor_fuel_kg=donor_cost[2],
            options=chosen,
        )
        return candidate if candidate.savings > 0 else None

    def _apply_elimination(
        self,
        problem: ProblemData,
        solution: Solution,
        candidate: _EliminationCandidate,
    ) -> TailEliminationDecision:
        donor_ids = list(candidate.donor_ids)
        cursor = 0
        recipient_decisions: list[TailRecipientDecision] = []
        for option in sorted(candidate.options, key=lambda item: item.host_uid):
            host_ids = self._person_ids(solution, option.host_uid)
            moved = donor_ids[cursor : cursor + option.count]
            cursor += option.count
            self.route_optimizer.install(
                problem,
                solution,
                option.host_uid,
                [*host_ids, *moved],
                option.optimized,
            )
            recipient_decisions.append(
                TailRecipientDecision(
                    flight_uid=option.host_uid,
                    passenger_count=option.count,
                    added_aircraft_minutes=option.added_aircraft_minutes,
                    after_aircraft_type=option.optimized.route.aircraft_type,
                    after_base_airport=option.optimized.route.base_airport,
                    after_service_order=option.optimized.route.service_order,
                )
            )
        if cursor != len(donor_ids):
            raise AssertionError(f"donor reassignment incomplete: {cursor}!={len(donor_ids)}")
        del solution.flights[candidate.donor_uid]
        return TailEliminationDecision(
            donor_flight_uid=candidate.donor_uid,
            donor_kind=candidate.donor_kind,
            donor_passenger_count=len(donor_ids),
            donor_minutes=candidate.donor_minutes,
            added_recipient_minutes=candidate.added_aircraft_minutes,
            aircraft_savings_minutes=candidate.savings,
            passenger_minutes_delta=candidate.passenger_delta,
            fuel_kg_delta=candidate.fuel_delta,
            recipients=tuple(recipient_decisions),
        )

    @staticmethod
    def _person_ids(solution: Solution, flight_uid: str) -> list[str]:
        return sorted(
            pid for pid, assignment in solution.assignments.items() if assignment.flight_uid == flight_uid
        )

    def _profile_for_flight(self, problem: ProblemData, solution: Solution, flight_uid: str) -> ProfileKey:
        return self.route_optimizer.profile(problem, self._person_ids(solution, flight_uid))

    @staticmethod
    def _flight_cost(problem: ProblemData, solution: Solution, flight_uid: str) -> tuple[int, int, float]:
        assignments = {
            pid: assignment
            for pid, assignment in solution.assignments.items()
            if assignment.flight_uid == flight_uid
        }
        metrics = evaluate_solution(
            problem,
            Solution(flights={flight_uid: solution.flights[flight_uid]}, assignments=assignments),
        )
        return (
            metrics.total_aircraft_usage_minutes,
            metrics.total_passenger_travel_minutes,
            metrics.total_fuel_consumption_kg,
        )
