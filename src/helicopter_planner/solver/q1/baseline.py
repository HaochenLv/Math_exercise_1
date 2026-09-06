from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from helicopter_planner.domain import (
    AIRPORTS,
    REFUEL_FACILITIES,
    Assignment,
    FlightPlan,
    ProblemData,
    SeaStop,
    Solution,
)
from helicopter_planner.evaluation.metrics import leg_minutes


@dataclass(frozen=True)
class ShuttleRoute:
    """Best feasible single-destination shuttle route for one airport/type pair."""

    base_airport: str
    destination_id: str
    aircraft_type: str
    sea_stops: tuple[SeaStop, ...]
    trip_minutes: int
    passenger_minutes_to_destination: int
    total_distance_km: float


@dataclass(frozen=True)
class FacilityDecision:
    facility_id: str
    aircraft_type: str
    land_allocation: tuple[int, int, int]  # A01, A02, A03
    trip_counts: tuple[int, int, int]  # A01, A02, A03
    total_aircraft_usage_minutes: int
    total_passenger_travel_minutes: int
    total_fuel_consumption_kg: float
    number_of_flights: int


@dataclass(frozen=True)
class BaselineResult:
    solution: Solution
    decisions: tuple[FacilityDecision, ...]


@dataclass(frozen=True)
class _TypePlan:
    aircraft_type: str
    land_allocation: tuple[int, int, int]
    trip_counts: tuple[int, int, int]
    routes: Mapping[str, ShuttleRoute]
    objective_key: tuple[int, int, float, int, int, int, int]


class Q1SingleFacilityBaselineSolver:
    """Q1 baseline: one destination per passenger-carrying shuttle.

    For each destination facility independently:
      1. choose one aircraft type for all passengers going to that facility;
      2. optimize LAND passengers over A01/A02/A03;
      3. split each airport's passengers into capacity-sized shuttles;
      4. allow only technical refuel stops at the eight official refuel
         facilities when needed/useful for a feasible minimum-time shuttle.
         No other destination is served.

    Lexicographic objective:
      total aircraft usage time,
      total passenger travel time,
      total fuel consumption,
      number of flights,
      deterministic LAND allocation/type tie-breaks.
    """

    def __init__(self, *, max_sea_landings: int = 5) -> None:
        self.max_sea_landings = max_sea_landings

    def solve(self, problem: ProblemData) -> Solution:
        return self.solve_with_diagnostics(problem).solution

    def solve_with_diagnostics(self, problem: ProblemData) -> BaselineResult:
        grouped = self._group_q1_requests(problem)
        route_cache: dict[tuple[str, str, str], ShuttleRoute | None] = {}

        chosen: dict[str, _TypePlan] = {}
        decisions: list[FacilityDecision] = []

        for facility_id in sorted(grouped):
            fixed, land_ids = grouped[facility_id]
            candidates: list[tuple[tuple, _TypePlan]] = []
            for aircraft_type in sorted(problem.aircraft_types):
                plan = self._plan_facility_type(
                    problem,
                    facility_id,
                    aircraft_type,
                    fixed,
                    len(land_ids),
                    route_cache,
                )
                if plan is not None:
                    candidates.append((plan.objective_key + (aircraft_type,), plan))

            if not candidates:
                raise ValueError(
                    f"no feasible single-destination baseline plan for {facility_id}"
                )

            _, plan = min(candidates, key=lambda item: item[0])
            chosen[facility_id] = plan
            decisions.append(
                FacilityDecision(
                    facility_id=facility_id,
                    aircraft_type=plan.aircraft_type,
                    land_allocation=plan.land_allocation,
                    trip_counts=plan.trip_counts,
                    total_aircraft_usage_minutes=plan.objective_key[0],
                    total_passenger_travel_minutes=plan.objective_key[1],
                    total_fuel_consumption_kg=plan.objective_key[2],
                    number_of_flights=plan.objective_key[3],
                )
            )

        solution = self._materialize_solution(problem, grouped, chosen)
        return BaselineResult(solution=solution, decisions=tuple(decisions))

    @staticmethod
    def _group_q1_requests(
        problem: ProblemData,
    ) -> dict[str, tuple[dict[str, list[str]], list[str]]]:
        grouped: dict[str, tuple[dict[str, list[str]], list[str]]] = {}
        mutable: dict[str, dict[str, list[str]]] = {}

        for req in problem.requests.values():
            if req.origin_id != "LAND" and req.origin_id not in AIRPORTS:
                raise ValueError(f"{req.person_id}: invalid Q1 origin {req.origin_id}")
            if not req.destination_id.startswith("F"):
                raise ValueError(
                    f"{req.person_id}: Q1 destination must be sea facility, "
                    f"got {req.destination_id}"
                )
            bucket = mutable.setdefault(
                req.destination_id,
                {"A01": [], "A02": [], "A03": [], "LAND": []},
            )
            bucket[req.origin_id].append(req.person_id)

        for facility_id, bucket in mutable.items():
            fixed = {a: sorted(bucket[a]) for a in sorted(AIRPORTS)}
            grouped[facility_id] = (fixed, sorted(bucket["LAND"]))
        return grouped

    def _plan_facility_type(
        self,
        problem: ProblemData,
        facility_id: str,
        aircraft_type: str,
        fixed: Mapping[str, list[str]],
        land_count: int,
        route_cache: dict[tuple[str, str, str], ShuttleRoute | None],
    ) -> _TypePlan | None:
        spec = problem.aircraft_types[aircraft_type]
        routes: dict[str, ShuttleRoute] = {}

        for airport in sorted(AIRPORTS):
            key = (airport, facility_id, aircraft_type)
            if key not in route_cache:
                route_cache[key] = self._best_shuttle_route(
                    problem, airport, facility_id, aircraft_type
                )
            route = route_cache[key]
            if route is not None:
                routes[airport] = route
            elif fixed[airport]:
                return None

        best: tuple[
            tuple[int, int, float, int, int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
        ] | None = None

        for x_a01 in range(land_count + 1):
            for x_a02 in range(land_count - x_a01 + 1):
                x_a03 = land_count - x_a01 - x_a02
                allocation = (x_a01, x_a02, x_a03)

                total_aircraft = 0
                total_passenger = 0
                total_fuel = 0.0
                total_flights = 0
                trip_counts: list[int] = []
                feasible = True

                for airport, land_assigned in zip(
                    ("A01", "A02", "A03"), allocation, strict=True
                ):
                    n_people = len(fixed[airport]) + land_assigned
                    if n_people == 0:
                        trip_counts.append(0)
                        continue
                    route = routes.get(airport)
                    if route is None:
                        feasible = False
                        break
                    n_trips = math.ceil(n_people / spec.seats)
                    trip_counts.append(n_trips)
                    total_aircraft += n_trips * route.trip_minutes
                    total_passenger += (
                        n_people * route.passenger_minutes_to_destination
                    )
                    total_fuel += (
                        n_trips
                        * route.total_distance_km
                        * spec.fuel_rate_kg_per_km
                    )
                    total_flights += n_trips

                if not feasible:
                    continue

                objective = (
                    total_aircraft,
                    total_passenger,
                    round(total_fuel, 9),
                    total_flights,
                    x_a01,
                    x_a02,
                    x_a03,
                )
                candidate = (objective, allocation, tuple(trip_counts))
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None:
            return None

        objective, allocation, trip_counts = best
        return _TypePlan(
            aircraft_type=aircraft_type,
            land_allocation=allocation,
            trip_counts=trip_counts,
            routes=routes,
            objective_key=objective,
        )

    def _best_shuttle_route(
        self,
        problem: ProblemData,
        base_airport: str,
        destination_id: str,
        aircraft_type: str,
    ) -> ShuttleRoute | None:
        """Shortest feasible closed route serving exactly one destination.

        Any extra sea stop is restricted to an official refuel facility and is
        always used for refueling. This keeps the baseline single-destination
        while still making distant fixed-origin requests fuel-feasible.
        """

        spec = problem.aircraft_types[aircraft_type]
        max_distance_since_refuel = (
            spec.tank_capacity_kg - spec.min_safe_fuel_kg
        ) / spec.fuel_rate_kg_per_km

        available_refuel = tuple(
            sorted(
                r
                for r in REFUEL_FACILITIES
                if r != destination_id and r in problem.distances
            )
        )

        # Heap fields:
        # elapsed, passenger-arrival-key, distance, stops, path, flags,
        # current, target_visited, distance_since_refuel, passenger_arrival
        heap: list[tuple] = [
            (0, -1, 0.0, 0, (), (), base_airport, False, 0.0, -1)
        ]
        best_state: dict[tuple, tuple] = {}
        complete: list[tuple] = []

        while heap:
            (
                elapsed,
                passenger_key,
                total_distance,
                stop_count,
                path,
                flags,
                current,
                target_visited,
                used_distance,
                passenger_arrival,
            ) = heapq.heappop(heap)

            state = (
                current,
                target_visited,
                round(used_distance, 9),
                stop_count,
            )
            state_rank = (
                elapsed,
                passenger_key,
                total_distance,
                path,
                flags,
            )
            previous = best_state.get(state)
            if previous is not None and previous <= state_rank:
                continue
            best_state[state] = state_rank

            if target_visited:
                return_distance = problem.distance(current, base_airport)
                if (
                    used_distance + return_distance
                    <= max_distance_since_refuel + 1e-9
                ):
                    total_minutes = elapsed + leg_minutes(
                        return_distance, spec.speed_kmh
                    )
                    complete.append(
                        (
                            total_minutes,
                            passenger_arrival,
                            total_distance + return_distance,
                            stop_count,
                            path,
                            flags,
                        )
                    )

            if stop_count >= self.max_sea_landings:
                continue

            next_nodes: list[str] = []
            if not target_visited:
                next_nodes.append(destination_id)
            next_nodes.extend(available_refuel)

            for next_node in next_nodes:
                if next_node in path:
                    continue

                distance = problem.distance(current, next_node)
                if used_distance + distance > max_distance_since_refuel + 1e-9:
                    continue

                arrival_elapsed = elapsed + leg_minutes(
                    distance, spec.speed_kmh
                )

                if next_node == destination_id and not target_visited:
                    target_refuel_options = (
                        (False, True)
                        if destination_id in REFUEL_FACILITIES
                        else (False,)
                    )
                    for refuel in target_refuel_options:
                        next_used = (
                            0.0 if refuel else used_distance + distance
                        )
                        next_elapsed = arrival_elapsed + (20 if refuel else 10)
                        heapq.heappush(
                            heap,
                            (
                                next_elapsed,
                                arrival_elapsed,
                                total_distance + distance,
                                stop_count + 1,
                                path + (next_node,),
                                flags + (refuel,),
                                next_node,
                                True,
                                next_used,
                                arrival_elapsed,
                            ),
                        )
                else:
                    # Technical stops are used only as fuel resets in this baseline.
                    next_elapsed = arrival_elapsed + 20
                    heapq.heappush(
                        heap,
                        (
                            next_elapsed,
                            passenger_key,
                            total_distance + distance,
                            stop_count + 1,
                            path + (next_node,),
                            flags + (True,),
                            next_node,
                            target_visited,
                            0.0,
                            passenger_arrival,
                        ),
                    )

        if not complete:
            return None

        (
            trip_minutes,
            passenger_minutes,
            total_distance,
            _,
            path,
            flags,
        ) = min(complete)

        return ShuttleRoute(
            base_airport=base_airport,
            destination_id=destination_id,
            aircraft_type=aircraft_type,
            sea_stops=tuple(
                SeaStop(facility_id=facility, refuel=refuel)
                for facility, refuel in zip(path, flags, strict=True)
            ),
            trip_minutes=trip_minutes,
            passenger_minutes_to_destination=passenger_minutes,
            total_distance_km=total_distance,
        )

    @staticmethod
    def _materialize_solution(
        problem: ProblemData,
        grouped: Mapping[
            str, tuple[dict[str, list[str]], list[str]]
        ],
        chosen: Mapping[str, _TypePlan],
    ) -> Solution:
        solution = Solution()
        next_flight_number = 1

        for facility_id in sorted(grouped):
            fixed, land_ids = grouped[facility_id]
            plan = chosen[facility_id]
            spec = problem.aircraft_types[plan.aircraft_type]

            cursor = 0
            passengers_by_airport: dict[str, list[str]] = {}
            for airport, count in zip(
                ("A01", "A02", "A03"), plan.land_allocation, strict=True
            ):
                assigned_land = land_ids[cursor : cursor + count]
                cursor += count
                passengers_by_airport[airport] = sorted(
                    [*fixed[airport], *assigned_land]
                )
            if cursor != len(land_ids):
                raise AssertionError("LAND allocation did not consume all passengers")

            for airport in ("A01", "A02", "A03"):
                people = passengers_by_airport[airport]
                if not people:
                    continue
                route = plan.routes[airport]
                target_index = 1 + next(
                    i
                    for i, stop in enumerate(route.sea_stops)
                    if stop.facility_id == facility_id
                )

                for start in range(0, len(people), spec.seats):
                    chunk = people[start : start + spec.seats]
                    flight_uid = f"FLT{next_flight_number:06d}"
                    next_flight_number += 1

                    solution.flights[flight_uid] = FlightPlan(
                        flight_uid=flight_uid,
                        base_airport=airport,
                        aircraft_type=plan.aircraft_type,
                        sea_stops=list(route.sea_stops),
                    )
                    for person_id in chunk:
                        solution.assignments[person_id] = Assignment(
                            person_id=person_id,
                            flight_uid=flight_uid,
                            pickup_index=0,
                            delivery_index=target_index,
                        )

        return solution
