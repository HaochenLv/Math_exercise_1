from __future__ import annotations

from dataclasses import dataclass, field
from helicopter_planner.domain import AIRPORTS, REFUEL_FACILITIES, ProblemData, Solution

@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

def _actual_endpoint(request_location: str, base_airport: str) -> str:
    return base_airport if request_location == "LAND" else request_location

def check_solution(problem: ProblemData, solution: Solution, *, require_all_requests: bool = True) -> CheckResult:
    errors: list[str] = []
    if require_all_requests:
        missing = sorted(set(problem.requests) - set(solution.assignments))
        extra = sorted(set(solution.assignments) - set(problem.requests))
        if missing:
            errors.append(f"missing assignments: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            errors.append(f"unknown assigned persons: {extra[:10]}{'...' if len(extra) > 10 else ''}")

    assignments_by_flight: dict[str, list] = {uid: [] for uid in solution.flights}
    for person_id, a in solution.assignments.items():
        if a.person_id != person_id:
            errors.append(f"assignment key/person mismatch for {person_id}")
        if a.flight_uid not in solution.flights:
            errors.append(f"{person_id}: unknown flight {a.flight_uid}")
            continue
        assignments_by_flight[a.flight_uid].append(a)

    for uid, flight in solution.flights.items():
        if uid != flight.flight_uid:
            errors.append(f"flight key/uid mismatch: {uid} != {flight.flight_uid}")
        if flight.base_airport not in AIRPORTS:
            errors.append(f"{uid}: invalid base airport {flight.base_airport}")
        if flight.aircraft_type not in problem.aircraft_types:
            errors.append(f"{uid}: invalid aircraft type {flight.aircraft_type}")
            continue
        if len(flight.sea_stops) > 5:
            errors.append(f"{uid}: more than 5 sea landings")
        for idx, stop in enumerate(flight.sea_stops, start=1):
            if not stop.facility_id.startswith("F"):
                errors.append(f"{uid}: stop {idx} is not a sea facility")
            if stop.refuel and stop.facility_id not in REFUEL_FACILITIES:
                errors.append(f"{uid}: illegal refuel at {stop.facility_id}")

        route = flight.full_route()
        spec = problem.aircraft_types[flight.aircraft_type]
        fuel = spec.tank_capacity_kg
        for leg_idx, (origin, dest) in enumerate(zip(route, route[1:])):
            try:
                distance = problem.distance(origin, dest)
            except KeyError:
                errors.append(f"{uid}: unknown route location on leg {origin}->{dest}")
                continue
            fuel -= distance * spec.fuel_rate_kg_per_km
            if fuel + 1e-9 < spec.min_safe_fuel_kg:
                errors.append(f"{uid}: fuel below reserve on arrival {dest} after leg {leg_idx}: {fuel:.3f} kg")
            if leg_idx < len(flight.sea_stops) and flight.sea_stops[leg_idx].refuel:
                fuel = spec.tank_capacity_kg

        leg_loads = [0] * (len(route) - 1)
        for a in assignments_by_flight.get(uid, []):
            req = problem.requests.get(a.person_id)
            if req is None:
                continue
            if not (0 <= a.pickup_index < a.delivery_index < len(route)):
                errors.append(f"{a.person_id}: invalid pickup/delivery indexes")
                continue
            pickup_loc = route[a.pickup_index]
            delivery_loc = route[a.delivery_index]
            expected_pickup = _actual_endpoint(req.origin_id, flight.base_airport)
            expected_delivery = _actual_endpoint(req.destination_id, flight.base_airport)
            if pickup_loc != expected_pickup:
                errors.append(f"{a.person_id}: pickup {pickup_loc} != required {expected_pickup}")
            if delivery_loc != expected_delivery:
                errors.append(f"{a.person_id}: delivery {delivery_loc} != required {expected_delivery}")
            try:
                first_delivery = route.index(expected_delivery, a.pickup_index + 1)
            except ValueError:
                first_delivery = None
            if first_delivery != a.delivery_index:
                errors.append(f"{a.person_id}: delivery is not first stop at destination after pickup")
            for leg in range(a.pickup_index, a.delivery_index):
                leg_loads[leg] += 1

        for leg_idx, load in enumerate(leg_loads):
            if load > spec.seats:
                errors.append(f"{uid}: seat capacity exceeded on leg {leg_idx}: {load}>{spec.seats}")
    return CheckResult(ok=not errors, errors=errors)
