from __future__ import annotations

import math
from dataclasses import dataclass
from helicopter_planner.domain import ProblemData, Solution

@dataclass(frozen=True)
class SolutionMetrics:
    total_aircraft_usage_minutes: int
    total_passenger_travel_minutes: int
    number_of_flights: int
    total_fuel_consumption_kg: float
    passenger_km: float
    available_seat_km: float
    seat_utilization: float

def leg_minutes(distance_km: float, speed_kmh: float) -> int:
    return math.ceil(distance_km / speed_kmh * 60.0 - 1e-12)

def stop_minutes(refuel: bool) -> int:
    return 20 if refuel else 10

def evaluate_solution(problem: ProblemData, solution: Solution) -> SolutionMetrics:
    total_aircraft = 0
    total_passenger = 0
    total_fuel = 0.0
    passenger_km = 0.0
    available_seat_km = 0.0
    assignments_by_flight: dict[str, list] = {uid: [] for uid in solution.flights}
    for a in solution.assignments.values():
        assignments_by_flight.setdefault(a.flight_uid, []).append(a)
    for uid, flight in solution.flights.items():
        route = flight.full_route()
        spec = problem.aircraft_types[flight.aircraft_type]
        distances = [problem.distance(a, b) for a, b in zip(route, route[1:])]
        flight_mins = [leg_minutes(d, spec.speed_kmh) for d in distances]
        stop_mins = [stop_minutes(s.refuel) for s in flight.sea_stops]
        total_aircraft += sum(flight_mins) + sum(stop_mins)
        total_fuel += sum(distances) * spec.fuel_rate_kg_per_km
        available_seat_km += sum(distances) * spec.seats
        leg_loads = [0] * len(distances)
        for a in assignments_by_flight.get(uid, []):
            for leg in range(a.pickup_index, a.delivery_index):
                leg_loads[leg] += 1
            total_passenger += sum(flight_mins[a.pickup_index:a.delivery_index]) + sum(stop_mins[a.pickup_index:a.delivery_index - 1])
        passenger_km += sum(load * d for load, d in zip(leg_loads, distances))
    seat_utilization = passenger_km / available_seat_km if available_seat_km else 0.0
    return SolutionMetrics(total_aircraft, total_passenger, len(solution.flights), total_fuel, passenger_km, available_seat_km, seat_utilization)
