from __future__ import annotations

import heapq
from dataclasses import dataclass

from helicopter_planner.domain import (
    AIRPORTS,
    REFUEL_FACILITIES,
    ProblemData,
    SeaStop,
)
from helicopter_planner.evaluation.metrics import leg_minutes


@dataclass(frozen=True)
class BuiltRoute:
    """Minimum-time feasible route for a fixed airport/type/service order.

    `service_stop_indices` are indices in the full route, i.e. the base airport
    is stop 0. `service_arrival_minutes` are elapsed minutes from airport
    departure to arrival at each service destination.
    """

    base_airport: str
    aircraft_type: str
    service_order: tuple[str, ...]
    sea_stops: tuple[SeaStop, ...]
    service_stop_indices: tuple[int, ...]
    service_arrival_minutes: tuple[int, ...]
    trip_minutes: int
    total_distance_km: float

    def service_stop_index(self, facility_id: str) -> int:
        return self.service_stop_indices[self.service_order.index(facility_id)]

    def service_arrival_minute(self, facility_id: str) -> int:
        return self.service_arrival_minutes[self.service_order.index(facility_id)]


class Q1RouteBuilder:
    """Build a fuel-feasible Q1 route for an ordered set of service facilities.

    The service order is fixed by the caller. The builder may insert official
    refuel facilities as technical stops and may optionally refuel when a
    service facility itself is an official refuel facility. Technical stops do
    not serve passengers. The route starts and ends at the same base airport.

    Primary objective: minimum aircraft usage time. Equal-time ties prefer
    earlier service arrivals, then shorter distance and fewer sea landings.
    """

    def __init__(self, *, max_sea_landings: int = 5) -> None:
        self.max_sea_landings = max_sea_landings

    def build(
        self,
        problem: ProblemData,
        base_airport: str,
        service_order: tuple[str, ...] | list[str],
        aircraft_type: str,
    ) -> BuiltRoute | None:
        order = tuple(service_order)
        if base_airport not in AIRPORTS:
            raise ValueError(f"invalid base airport: {base_airport}")
        if aircraft_type not in problem.aircraft_types:
            raise ValueError(f"invalid aircraft type: {aircraft_type}")
        if not order:
            raise ValueError("service_order must not be empty")
        if len(order) != len(set(order)):
            raise ValueError("service_order must contain distinct facilities")
        if len(order) > self.max_sea_landings:
            return None
        for facility in order:
            if not facility.startswith("F") or facility not in problem.distances:
                raise ValueError(f"invalid service facility: {facility}")

        spec = problem.aircraft_types[aircraft_type]
        max_distance_since_refuel = (
            spec.tank_capacity_kg - spec.min_safe_fuel_kg
        ) / spec.fuel_rate_kg_per_km

        service_set = set(order)
        technical_refuel = tuple(
            sorted(
                facility
                for facility in REFUEL_FACILITIES
                if facility not in service_set and facility in problem.distances
            )
        )

        # Heap fields:
        # elapsed, total_distance, stop_count, next_service_idx, current,
        # distance_since_refuel, path, refuel_flags, service_indices, arrivals.
        heap: list[tuple] = [
            (0, 0.0, 0, 0, base_airport, 0.0, (), (), (), ())
        ]
        best_state: dict[tuple, tuple] = {}
        completed: list[tuple] = []

        while heap:
            (
                elapsed,
                total_distance,
                stop_count,
                next_service_idx,
                current,
                used_distance,
                path,
                flags,
                service_indices,
                arrivals,
            ) = heapq.heappop(heap)

            state = (
                current,
                next_service_idx,
                round(used_distance, 9),
                stop_count,
            )
            rank = (elapsed, total_distance, path, flags)
            previous = best_state.get(state)
            if previous is not None and previous <= rank:
                continue
            best_state[state] = rank

            if next_service_idx == len(order):
                return_distance = problem.distance(current, base_airport)
                if used_distance + return_distance <= max_distance_since_refuel + 1e-9:
                    completed.append(
                        (
                            elapsed
                            + leg_minutes(return_distance, spec.speed_kmh),
                            sum(arrivals),
                            total_distance + return_distance,
                            stop_count,
                            path,
                            flags,
                            service_indices,
                            arrivals,
                        )
                    )

            if stop_count >= self.max_sea_landings:
                continue

            # Visit the next required service destination.
            if next_service_idx < len(order):
                next_service = order[next_service_idx]
                distance = problem.distance(current, next_service)
                if used_distance + distance <= max_distance_since_refuel + 1e-9:
                    arrival_elapsed = elapsed + leg_minutes(distance, spec.speed_kmh)
                    refuel_options = (
                        (False, True)
                        if next_service in REFUEL_FACILITIES
                        else (False,)
                    )
                    for refuel in refuel_options:
                        next_used = 0.0 if refuel else used_distance + distance
                        heapq.heappush(
                            heap,
                            (
                                arrival_elapsed + (20 if refuel else 10),
                                total_distance + distance,
                                stop_count + 1,
                                next_service_idx + 1,
                                next_service,
                                next_used,
                                path + (next_service,),
                                flags + (refuel,),
                                service_indices + (stop_count + 1,),
                                arrivals + (arrival_elapsed,),
                            ),
                        )

            # Technical refuel stops may be repeated; the five-landing cap keeps
            # the state space finite. Service destinations are never used as
            # technical stops, preserving the official first-arrival rule.
            for refuel_facility in technical_refuel:
                if refuel_facility == current:
                    continue
                distance = problem.distance(current, refuel_facility)
                if used_distance + distance > max_distance_since_refuel + 1e-9:
                    continue
                arrival_elapsed = elapsed + leg_minutes(distance, spec.speed_kmh)
                heapq.heappush(
                    heap,
                    (
                        arrival_elapsed + 20,
                        total_distance + distance,
                        stop_count + 1,
                        next_service_idx,
                        refuel_facility,
                        0.0,
                        path + (refuel_facility,),
                        flags + (True,),
                        service_indices,
                        arrivals,
                    ),
                )

        if not completed:
            return None

        (
            trip_minutes,
            _,
            total_distance,
            _,
            path,
            flags,
            service_indices,
            arrivals,
        ) = min(completed)

        return BuiltRoute(
            base_airport=base_airport,
            aircraft_type=aircraft_type,
            service_order=order,
            sea_stops=tuple(
                SeaStop(facility_id=facility, refuel=refuel)
                for facility, refuel in zip(path, flags, strict=True)
            ),
            service_stop_indices=tuple(service_indices),
            service_arrival_minutes=tuple(arrivals),
            trip_minutes=trip_minutes,
            total_distance_km=total_distance,
        )
