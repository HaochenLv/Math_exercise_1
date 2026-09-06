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
from helicopter_planner.solver.q1.pair_merge import Q1GreedyPairSavingsSolver
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder


class _StartingSolver(Protocol):
    def solve(self, problem: ProblemData) -> Solution: ...


@dataclass(frozen=True)
class TailEliminationDecision:
    donor_flight_uid: str
    base_airport: str
    destination_id: str
    donor_passenger_count: int
    recipient_allocations: tuple[tuple[str, int], ...]
    donor_aircraft_usage_minutes: int
    insertion_aircraft_minutes: int
    aircraft_savings_minutes: int
    passenger_savings_minutes: int
    fuel_savings_kg: float


@dataclass(frozen=True)
class TailEliminationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[TailEliminationDecision, ...]


@dataclass(frozen=True)
class _FlightSummary:
    passenger_ids: tuple[str, ...]
    destination_counts: tuple[tuple[str, int], ...]
    aircraft_usage_minutes: int
    passenger_travel_minutes: int
    fuel_kg: float
    arrival_by_destination: tuple[tuple[str, int], ...]
    delivery_index_by_destination: tuple[tuple[str, int], ...]

    @property
    def destinations(self) -> tuple[str, ...]:
        return tuple(destination for destination, _ in self.destination_counts)

    @property
    def passenger_count(self) -> int:
        return len(self.passenger_ids)

    def arrival_map(self) -> dict[str, int]:
        return dict(self.arrival_by_destination)

    def delivery_index_map(self) -> dict[str, int]:
        return dict(self.delivery_index_by_destination)


@dataclass(frozen=True)
class _RecipientOption:
    recipient_uid: str
    inserted_count: int
    route: BuiltRoute | None
    delivery_index_if_unchanged: int | None
    delta_aircraft_minutes: int
    delta_passenger_minutes: int
    delta_fuel_kg: float


@dataclass(frozen=True)
class _EliminationCandidate:
    donor_uid: str
    base_airport: str
    destination_id: str
    donor_passengers: tuple[str, ...]
    donor_aircraft_usage_minutes: int
    donor_passenger_travel_minutes: int
    donor_fuel_kg: float
    recipient_options: tuple[_RecipientOption, ...]

    @property
    def insertion_aircraft_minutes(self) -> int:
        return sum(option.delta_aircraft_minutes for option in self.recipient_options)

    @property
    def aircraft_savings_minutes(self) -> int:
        return self.donor_aircraft_usage_minutes - self.insertion_aircraft_minutes

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


class Q1TailEliminationSolver:
    """Q1 V2: eliminate single-destination tail flights by passenger insertion.

    The solver starts from Pair-Merge V1.  At each iteration it considers every
    *single-destination* flight as a donor.  Its passengers may be split across
    several other flights at the SAME base airport, using only seats that are
    already empty under each recipient's current aircraft type.

    If the donor destination is new to a recipient, the recipient route is
    rebuilt with the same aircraft type and all service-order permutations are
    tested through Q1RouteBuilder.  Recipient flights are limited to at most
    three passenger service destinations in this V2, keeping the experiment
    deliberately interpretable.  LAND airport choices remain frozen.

    A small dynamic program finds the least aircraft-time insertion plan that
    moves all donor passengers.  The best positive-saving donor elimination is
    applied, summaries are recomputed, and the process repeats until no further
    positive aircraft-time saving exists.
    """

    def __init__(
        self,
        *,
        starting_solver: _StartingSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
        max_service_destinations: int = 3,
    ) -> None:
        self.starting_solver = starting_solver or Q1GreedyPairSavingsSolver()
        self.route_builder = route_builder or Q1RouteBuilder()
        self.max_service_destinations = max_service_destinations

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> TailEliminationResult:
        solution = self._copy_solution(self.starting_solver.solve(problem))
        starting_metrics = evaluate_solution(problem, solution)
        decisions: list[TailEliminationDecision] = []
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}

        while True:
            summaries = self._summarize_all(problem, solution)
            candidates: list[_EliminationCandidate] = []

            for donor_uid in sorted(solution.flights):
                donor = summaries[donor_uid]
                if len(donor.destinations) != 1:
                    continue
                candidate = self._best_elimination_candidate(
                    problem,
                    solution,
                    summaries,
                    donor_uid,
                    route_cache,
                )
                if candidate is not None and candidate.aircraft_savings_minutes > 0:
                    candidates.append(candidate)

            if not candidates:
                break

            candidates.sort(
                key=lambda c: (
                    -c.aircraft_savings_minutes,
                    -c.passenger_savings_minutes,
                    -round(c.fuel_savings_kg, 9),
                    len(c.recipient_options),
                    c.donor_uid,
                )
            )
            chosen = candidates[0]
            self._apply_candidate(problem, solution, chosen)
            decisions.append(
                TailEliminationDecision(
                    donor_flight_uid=chosen.donor_uid,
                    base_airport=chosen.base_airport,
                    destination_id=chosen.destination_id,
                    donor_passenger_count=len(chosen.donor_passengers),
                    recipient_allocations=tuple(
                        (option.recipient_uid, option.inserted_count)
                        for option in chosen.recipient_options
                    ),
                    donor_aircraft_usage_minutes=chosen.donor_aircraft_usage_minutes,
                    insertion_aircraft_minutes=chosen.insertion_aircraft_minutes,
                    aircraft_savings_minutes=chosen.aircraft_savings_minutes,
                    passenger_savings_minutes=chosen.passenger_savings_minutes,
                    fuel_savings_kg=chosen.fuel_savings_kg,
                )
            )

        return TailEliminationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
        )

    def _best_elimination_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries: dict[str, _FlightSummary],
        donor_uid: str,
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None],
    ) -> _EliminationCandidate | None:
        donor_flight = solution.flights[donor_uid]
        donor = summaries[donor_uid]
        donor_destination = donor.destinations[0]
        need = donor.passenger_count

        recipient_options: list[tuple[str, tuple[_RecipientOption, ...]]] = []
        for recipient_uid in sorted(solution.flights):
            if recipient_uid == donor_uid:
                continue
            recipient_flight = solution.flights[recipient_uid]
            if recipient_flight.base_airport != donor_flight.base_airport:
                continue
            recipient = summaries[recipient_uid]
            spec = problem.aircraft_types[recipient_flight.aircraft_type]
            spare = spec.seats - recipient.passenger_count
            if spare <= 0:
                continue
            if (
                donor_destination not in recipient.destinations
                and len(recipient.destinations) >= self.max_service_destinations
            ):
                continue

            options = self._recipient_options(
                problem,
                solution,
                recipient_uid,
                recipient,
                donor_destination,
                min(spare, need),
                route_cache,
            )
            if options:
                recipient_options.append((recipient_uid, options))

        if sum(max(o.inserted_count for o in options) for _, options in recipient_options) < need:
            return None

        # delivered -> (lexicographic cost, selected recipient options)
        # Cost is insertion aircraft time first, then passenger/fuel increments.
        dp: dict[int, tuple[tuple[int, int, float, int], tuple[_RecipientOption, ...]]] = {
            0: ((0, 0, 0.0, 0), ())
        }
        for _, options in recipient_options:
            next_dp = dict(dp)  # q=0: leave this recipient untouched
            for delivered, (cost, selected) in dp.items():
                for option in options:
                    new_delivered = delivered + option.inserted_count
                    if new_delivered > need:
                        continue
                    new_cost = (
                        cost[0] + option.delta_aircraft_minutes,
                        cost[1] + option.delta_passenger_minutes,
                        cost[2] + option.delta_fuel_kg,
                        cost[3] + 1,
                    )
                    old = next_dp.get(new_delivered)
                    candidate = (new_cost, selected + (option,))
                    if old is None or new_cost < old[0]:
                        next_dp[new_delivered] = candidate
            dp = next_dp

        if need not in dp:
            return None
        _, selected = dp[need]
        selected = tuple(sorted(selected, key=lambda option: option.recipient_uid))
        candidate = _EliminationCandidate(
            donor_uid=donor_uid,
            base_airport=donor_flight.base_airport,
            destination_id=donor_destination,
            donor_passengers=donor.passenger_ids,
            donor_aircraft_usage_minutes=donor.aircraft_usage_minutes,
            donor_passenger_travel_minutes=donor.passenger_travel_minutes,
            donor_fuel_kg=donor.fuel_kg,
            recipient_options=selected,
        )
        return candidate if candidate.aircraft_savings_minutes > 0 else None

    def _recipient_options(
        self,
        problem: ProblemData,
        solution: Solution,
        recipient_uid: str,
        recipient: _FlightSummary,
        donor_destination: str,
        max_insert: int,
        route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None],
    ) -> tuple[_RecipientOption, ...]:
        flight = solution.flights[recipient_uid]
        spec = problem.aircraft_types[flight.aircraft_type]
        existing_counts = dict(recipient.destination_counts)

        # If this flight already serves the donor destination, filling spare
        # seats needs no route change at all.
        if donor_destination in existing_counts:
            delivery_index = recipient.delivery_index_map()[donor_destination]
            arrival = recipient.arrival_map()[donor_destination]
            return tuple(
                _RecipientOption(
                    recipient_uid=recipient_uid,
                    inserted_count=q,
                    route=None,
                    delivery_index_if_unchanged=delivery_index,
                    delta_aircraft_minutes=0,
                    delta_passenger_minutes=q * arrival,
                    delta_fuel_kg=0.0,
                )
                for q in range(1, max_insert + 1)
            )

        destinations = tuple(sorted((*recipient.destinations, donor_destination)))
        if len(destinations) > self.max_service_destinations:
            return ()

        routes: list[BuiltRoute] = []
        for order in itertools.permutations(destinations):
            key = (flight.base_airport, order, flight.aircraft_type)
            if key not in route_cache:
                route_cache[key] = self.route_builder.build(
                    problem,
                    flight.base_airport,
                    order,
                    flight.aircraft_type,
                )
            route = route_cache[key]
            if route is not None:
                routes.append(route)
        if not routes:
            return ()

        options: list[_RecipientOption] = []
        for q in range(1, max_insert + 1):
            best: tuple[tuple, BuiltRoute, int, float] | None = None
            counts = dict(existing_counts)
            counts[donor_destination] = q
            for route in routes:
                arrivals = dict(
                    zip(route.service_order, route.service_arrival_minutes, strict=True)
                )
                passenger_minutes = sum(
                    count * arrivals[destination]
                    for destination, count in counts.items()
                )
                fuel = route.total_distance_km * spec.fuel_rate_kg_per_km
                delta_aircraft = route.trip_minutes - recipient.aircraft_usage_minutes
                delta_passenger = passenger_minutes - recipient.passenger_travel_minutes
                delta_fuel = fuel - recipient.fuel_kg
                objective = (
                    delta_aircraft,
                    delta_passenger,
                    round(delta_fuel, 9),
                    route.service_order,
                )
                if best is None or objective < best[0]:
                    best = (objective, route, delta_passenger, delta_fuel)
            assert best is not None
            objective, route, delta_passenger, delta_fuel = best
            options.append(
                _RecipientOption(
                    recipient_uid=recipient_uid,
                    inserted_count=q,
                    route=route,
                    delivery_index_if_unchanged=None,
                    delta_aircraft_minutes=objective[0],
                    delta_passenger_minutes=delta_passenger,
                    delta_fuel_kg=delta_fuel,
                )
            )
        return tuple(options)

    def _apply_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        candidate: _EliminationCandidate,
    ) -> None:
        donor_people = list(sorted(candidate.donor_passengers))
        del solution.flights[candidate.donor_uid]
        for pid in donor_people:
            del solution.assignments[pid]

        cursor = 0
        for option in candidate.recipient_options:
            chunk = donor_people[cursor : cursor + option.inserted_count]
            cursor += option.inserted_count
            flight = solution.flights[option.recipient_uid]

            if option.route is None:
                assert option.delivery_index_if_unchanged is not None
                for pid in chunk:
                    solution.assignments[pid] = Assignment(
                        person_id=pid,
                        flight_uid=option.recipient_uid,
                        pickup_index=0,
                        delivery_index=option.delivery_index_if_unchanged,
                    )
                continue

            existing_people = sorted(
                pid
                for pid, assignment in solution.assignments.items()
                if assignment.flight_uid == option.recipient_uid
            )
            route = option.route
            solution.flights[option.recipient_uid] = FlightPlan(
                flight_uid=flight.flight_uid,
                base_airport=flight.base_airport,
                aircraft_type=flight.aircraft_type,
                sea_stops=list(route.sea_stops),
            )
            stop_index = dict(
                zip(route.service_order, route.service_stop_indices, strict=True)
            )
            for pid in (*existing_people, *chunk):
                destination = problem.requests[pid].destination_id
                solution.assignments[pid] = Assignment(
                    person_id=pid,
                    flight_uid=option.recipient_uid,
                    pickup_index=0,
                    delivery_index=stop_index[destination],
                )

        if cursor != len(donor_people):
            raise AssertionError("tail elimination did not reassign every donor passenger")

    @staticmethod
    def _summarize_all(
        problem: ProblemData,
        solution: Solution,
    ) -> dict[str, _FlightSummary]:
        people_by_flight: dict[str, list[str]] = {uid: [] for uid in solution.flights}
        for assignment in solution.assignments.values():
            people_by_flight.setdefault(assignment.flight_uid, []).append(assignment.person_id)
        return {
            uid: Q1TailEliminationSolver._summarize_flight(
                problem, solution, uid, tuple(sorted(people_by_flight[uid]))
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

        destination_counts: dict[str, int] = {}
        for pid in passenger_ids:
            destination = problem.requests[pid].destination_id
            destination_counts[destination] = destination_counts.get(destination, 0) + 1

        arrivals: dict[str, int] = {}
        delivery_indices: dict[str, int] = {}
        elapsed = 0
        for leg_idx, flight_minutes in enumerate(leg_times):
            elapsed += flight_minutes
            arrival_index = leg_idx + 1
            arrival_location = route[arrival_index]
            if arrival_location in destination_counts and arrival_location not in arrivals:
                arrivals[arrival_location] = elapsed
                delivery_indices[arrival_location] = arrival_index
            if leg_idx < len(stop_times):
                elapsed += stop_times[leg_idx]

        missing = set(destination_counts) - set(arrivals)
        if missing:
            raise ValueError(f"flight {flight_uid}: missing passenger destinations {missing}")
        passenger_travel = sum(
            count * arrivals[destination]
            for destination, count in destination_counts.items()
        )
        return _FlightSummary(
            passenger_ids=passenger_ids,
            destination_counts=tuple(sorted(destination_counts.items())),
            aircraft_usage_minutes=aircraft_usage,
            passenger_travel_minutes=passenger_travel,
            fuel_kg=fuel,
            arrival_by_destination=tuple(sorted(arrivals.items())),
            delivery_index_by_destination=tuple(sorted(delivery_indices.items())),
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
