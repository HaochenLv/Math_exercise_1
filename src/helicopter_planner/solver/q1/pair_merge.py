from __future__ import annotations

from dataclasses import dataclass

from helicopter_planner.domain import Assignment, FlightPlan, ProblemData, Solution
from helicopter_planner.evaluation.metrics import (
    SolutionMetrics,
    evaluate_solution,
    leg_minutes,
    stop_minutes,
)
from helicopter_planner.solver.q1.baseline import Q1SingleFacilityBaselineSolver
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder


@dataclass(frozen=True)
class PairMergeDecision:
    left_flight_uid: str
    right_flight_uid: str
    base_airport: str
    left_destination: str
    right_destination: str
    aircraft_type: str
    service_order: tuple[str, str]
    old_aircraft_usage_minutes: int
    new_aircraft_usage_minutes: int
    aircraft_savings_minutes: int
    old_passenger_travel_minutes: int
    new_passenger_travel_minutes: int
    old_fuel_kg: float
    new_fuel_kg: float


@dataclass(frozen=True)
class PairMergeResult:
    solution: Solution
    baseline_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[PairMergeDecision, ...]


@dataclass(frozen=True)
class _FlightSummary:
    passenger_ids: tuple[str, ...]
    destination_id: str
    aircraft_usage_minutes: int
    passenger_travel_minutes: int
    fuel_kg: float


@dataclass(frozen=True)
class _Candidate:
    left_uid: str
    right_uid: str
    base_airport: str
    left_destination: str
    right_destination: str
    left_passengers: tuple[str, ...]
    right_passengers: tuple[str, ...]
    route: BuiltRoute
    old_aircraft_usage_minutes: int
    old_passenger_travel_minutes: int
    old_fuel_kg: float
    new_passenger_travel_minutes: int
    new_fuel_kg: float

    @property
    def aircraft_savings_minutes(self) -> int:
        return self.old_aircraft_usage_minutes - self.route.trip_minutes

    @property
    def passenger_savings_minutes(self) -> int:
        return self.old_passenger_travel_minutes - self.new_passenger_travel_minutes

    @property
    def fuel_savings_kg(self) -> float:
        return self.old_fuel_kg - self.new_fuel_kg


class Q1GreedyPairSavingsSolver:
    """First Q1 improvement: disjoint same-airport pairwise savings merges.

    The solver starts from the single-facility baseline. Only two ORIGINAL
    baseline flights may be merged, so every new flight serves at most two
    passenger destinations. LAND airport assignments from the baseline are
    frozen in this version.

    For every eligible same-airport pair, all aircraft types and both service
    orders are tested. Q1RouteBuilder inserts technical refuel stops if needed.
    A merge is eligible only if it strictly reduces aircraft usage time.
    Positive-savings candidates are then selected greedily in descending
    aircraft-time savings order, with passenger-time and fuel savings as
    deterministic secondary tie-breaks.
    """

    def __init__(
        self,
        *,
        baseline_solver: Q1SingleFacilityBaselineSolver | None = None,
        route_builder: Q1RouteBuilder | None = None,
    ) -> None:
        self.baseline_solver = baseline_solver or Q1SingleFacilityBaselineSolver()
        self.route_builder = route_builder or Q1RouteBuilder()

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> PairMergeResult:
        baseline = self.baseline_solver.solve(problem)
        baseline_metrics = evaluate_solution(problem, baseline)

        people_by_flight: dict[str, list[str]] = {uid: [] for uid in baseline.flights}
        for assignment in baseline.assignments.values():
            people_by_flight[assignment.flight_uid].append(assignment.person_id)

        summaries: dict[str, _FlightSummary] = {}
        for uid in sorted(baseline.flights):
            summaries[uid] = self._summarize_flight(
                problem,
                baseline,
                uid,
                tuple(sorted(people_by_flight.get(uid, []))),
            )

        route_cache: dict[tuple[str, tuple[str, str], str], BuiltRoute | None] = {}
        candidates: list[_Candidate] = []
        uids = sorted(baseline.flights)
        for i, left_uid in enumerate(uids):
            left_flight = baseline.flights[left_uid]
            left = summaries[left_uid]
            for right_uid in uids[i + 1 :]:
                right_flight = baseline.flights[right_uid]
                right = summaries[right_uid]
                if left_flight.base_airport != right_flight.base_airport:
                    continue
                if left.destination_id == right.destination_id:
                    continue

                candidate = self._best_pair_candidate(
                    problem,
                    left_uid,
                    right_uid,
                    left_flight.base_airport,
                    left,
                    right,
                    route_cache,
                )
                if candidate is not None and candidate.aircraft_savings_minutes > 0:
                    candidates.append(candidate)

        candidates.sort(
            key=lambda c: (
                -c.aircraft_savings_minutes,
                -c.passenger_savings_minutes,
                -round(c.fuel_savings_kg, 9),
                c.route.trip_minutes,
                c.route.aircraft_type,
                c.route.service_order,
                c.left_uid,
                c.right_uid,
            )
        )

        used: set[str] = set()
        selected: list[_Candidate] = []
        for candidate in candidates:
            if candidate.left_uid in used or candidate.right_uid in used:
                continue
            used.add(candidate.left_uid)
            used.add(candidate.right_uid)
            selected.append(candidate)

        solution = self._materialize_solution(problem, baseline, selected)
        metrics = evaluate_solution(problem, solution)
        decisions = tuple(
            PairMergeDecision(
                left_flight_uid=c.left_uid,
                right_flight_uid=c.right_uid,
                base_airport=c.base_airport,
                left_destination=c.left_destination,
                right_destination=c.right_destination,
                aircraft_type=c.route.aircraft_type,
                service_order=(c.route.service_order[0], c.route.service_order[1]),
                old_aircraft_usage_minutes=c.old_aircraft_usage_minutes,
                new_aircraft_usage_minutes=c.route.trip_minutes,
                aircraft_savings_minutes=c.aircraft_savings_minutes,
                old_passenger_travel_minutes=c.old_passenger_travel_minutes,
                new_passenger_travel_minutes=c.new_passenger_travel_minutes,
                old_fuel_kg=c.old_fuel_kg,
                new_fuel_kg=c.new_fuel_kg,
            )
            for c in selected
        )

        return PairMergeResult(
            solution=solution,
            baseline_metrics=baseline_metrics,
            metrics=metrics,
            decisions=decisions,
        )

    @staticmethod
    def _summarize_flight(
        problem: ProblemData,
        solution: Solution,
        flight_uid: str,
        passenger_ids: tuple[str, ...],
    ) -> _FlightSummary:
        if not passenger_ids:
            raise ValueError(f"baseline flight {flight_uid} carries no passengers")
        destinations = {problem.requests[pid].destination_id for pid in passenger_ids}
        if len(destinations) != 1:
            raise ValueError(
                f"baseline flight {flight_uid} is not single-destination: {destinations}"
            )
        destination = next(iter(destinations))
        flight = solution.flights[flight_uid]
        spec = problem.aircraft_types[flight.aircraft_type]
        route = flight.full_route()
        distances = [problem.distance(a, b) for a, b in zip(route, route[1:])]
        leg_times = [leg_minutes(d, spec.speed_kmh) for d in distances]
        stop_times = [stop_minutes(s.refuel) for s in flight.sea_stops]
        aircraft_usage = sum(leg_times) + sum(stop_times)
        fuel = sum(distances) * spec.fuel_rate_kg_per_km
        passenger_time = 0
        for pid in passenger_ids:
            a = solution.assignments[pid]
            passenger_time += sum(leg_times[a.pickup_index : a.delivery_index])
            passenger_time += sum(
                stop_times[a.pickup_index : a.delivery_index - 1]
            )
        return _FlightSummary(
            passenger_ids=passenger_ids,
            destination_id=destination,
            aircraft_usage_minutes=aircraft_usage,
            passenger_travel_minutes=passenger_time,
            fuel_kg=fuel,
        )

    def _best_pair_candidate(
        self,
        problem: ProblemData,
        left_uid: str,
        right_uid: str,
        base_airport: str,
        left: _FlightSummary,
        right: _FlightSummary,
        route_cache: dict[tuple[str, tuple[str, str], str], BuiltRoute | None],
    ) -> _Candidate | None:
        total_people = len(left.passenger_ids) + len(right.passenger_ids)
        old_aircraft = left.aircraft_usage_minutes + right.aircraft_usage_minutes
        old_passenger = left.passenger_travel_minutes + right.passenger_travel_minutes
        old_fuel = left.fuel_kg + right.fuel_kg

        best: tuple[tuple, BuiltRoute, int, float] | None = None
        orders = (
            (left.destination_id, right.destination_id),
            (right.destination_id, left.destination_id),
        )
        counts = {
            left.destination_id: len(left.passenger_ids),
            right.destination_id: len(right.passenger_ids),
        }

        for aircraft_type in sorted(problem.aircraft_types):
            spec = problem.aircraft_types[aircraft_type]
            if total_people > spec.seats:
                continue
            for order in orders:
                key = (base_airport, order, aircraft_type)
                if key not in route_cache:
                    route_cache[key] = self.route_builder.build(
                        problem,
                        base_airport,
                        order,
                        aircraft_type,
                    )
                route = route_cache[key]
                if route is None:
                    continue

                arrival_by_destination = {
                    facility: arrival
                    for facility, arrival in zip(
                        route.service_order,
                        route.service_arrival_minutes,
                        strict=True,
                    )
                }
                passenger_time = sum(
                    counts[facility] * arrival_by_destination[facility]
                    for facility in counts
                )
                fuel = (
                    route.total_distance_km
                    * spec.fuel_rate_kg_per_km
                )
                objective = (
                    route.trip_minutes,
                    passenger_time,
                    round(fuel, 9),
                    aircraft_type,
                    order,
                )
                if best is None or objective < best[0]:
                    best = (objective, route, passenger_time, fuel)

        if best is None:
            return None
        _, route, passenger_time, fuel = best
        if route.trip_minutes >= old_aircraft:
            return None

        return _Candidate(
            left_uid=left_uid,
            right_uid=right_uid,
            base_airport=base_airport,
            left_destination=left.destination_id,
            right_destination=right.destination_id,
            left_passengers=left.passenger_ids,
            right_passengers=right.passenger_ids,
            route=route,
            old_aircraft_usage_minutes=old_aircraft,
            old_passenger_travel_minutes=old_passenger,
            old_fuel_kg=old_fuel,
            new_passenger_travel_minutes=passenger_time,
            new_fuel_kg=fuel,
        )

    @staticmethod
    def _materialize_solution(
        problem: ProblemData,
        baseline: Solution,
        selected: list[_Candidate],
    ) -> Solution:
        merged_old_uids = {
            uid
            for candidate in selected
            for uid in (candidate.left_uid, candidate.right_uid)
        }

        solution = Solution(
            flights={
                uid: FlightPlan(
                    flight_uid=flight.flight_uid,
                    base_airport=flight.base_airport,
                    aircraft_type=flight.aircraft_type,
                    sea_stops=list(flight.sea_stops),
                )
                for uid, flight in baseline.flights.items()
                if uid not in merged_old_uids
            },
            assignments={
                pid: assignment
                for pid, assignment in baseline.assignments.items()
                if assignment.flight_uid not in merged_old_uids
            },
        )

        numeric_ids = [
            int(uid[3:])
            for uid in baseline.flights
            if uid.startswith("FLT") and uid[3:].isdigit()
        ]
        next_number = max(numeric_ids, default=0) + 1

        for candidate in selected:
            flight_uid = f"FLT{next_number:06d}"
            next_number += 1
            route = candidate.route
            solution.flights[flight_uid] = FlightPlan(
                flight_uid=flight_uid,
                base_airport=candidate.base_airport,
                aircraft_type=route.aircraft_type,
                sea_stops=list(route.sea_stops),
            )

            stop_index = {
                facility: idx
                for facility, idx in zip(
                    route.service_order,
                    route.service_stop_indices,
                    strict=True,
                )
            }
            for pid in (*candidate.left_passengers, *candidate.right_passengers):
                destination = problem.requests[pid].destination_id
                solution.assignments[pid] = Assignment(
                    person_id=pid,
                    flight_uid=flight_uid,
                    pickup_index=0,
                    delivery_index=stop_index[destination],
                )

        return solution
