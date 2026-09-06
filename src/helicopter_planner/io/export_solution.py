from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from helicopter_planner.domain import Solution

def export_q1_solution(solution: Solution, routes_path: str | Path, assignments_path: str | Path) -> None:
    routes_path = Path(routes_path)
    assignments_path = Path(assignments_path)
    routes_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list] = defaultdict(list)
    for flight in solution.flights.values():
        by_type[flight.aircraft_type].append(flight)

    flight_no: dict[str, int] = {}
    for aircraft_type in ("T1", "T2", "T3"):
        for no, flight in enumerate(sorted(by_type[aircraft_type], key=lambda f: f.flight_uid), start=1):
            flight_no[flight.flight_uid] = no

    with routes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["aircraft_type", "flight_no", "stop_order", "facility_id", "refuel"])
        for aircraft_type in ("T1", "T2", "T3"):
            for flight in sorted(by_type[aircraft_type], key=lambda x: flight_no[x.flight_uid]):
                no = flight_no[flight.flight_uid]
                writer.writerow([aircraft_type, no, 0, flight.base_airport, 0])
                for idx, stop in enumerate(flight.sea_stops, start=1):
                    writer.writerow([aircraft_type, no, idx, stop.facility_id, int(stop.refuel)])
                writer.writerow([aircraft_type, no, len(flight.sea_stops) + 1, flight.base_airport, 0])

    with assignments_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "aircraft_type", "flight_no", "pickup_stop_order", "delivery_stop_order"])
        for person_id in sorted(solution.assignments):
            a = solution.assignments[person_id]
            flight = solution.flights[a.flight_uid]
            writer.writerow([person_id, flight.aircraft_type, flight_no[a.flight_uid], a.pickup_index, a.delivery_index])
