from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from helicopter_planner.domain import Assignment, FlightPlan, SeaStop, Solution


def load_exported_solution(
    routes_path: str | Path,
    assignments_path: str | Path,
    *,
    uid_prefix: str = "imported",
) -> Solution:
    """Load the official Q1/Q2 routes+assignments CSV format into ``Solution``.

    This is primarily used to freeze a previously validated milestone solution
    as the starting point of a later experiment, instead of re-running a
    heuristic construction whose tie-breaking may evolve.
    """

    routes_path = Path(routes_path)
    assignments_path = Path(assignments_path)

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with routes_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["aircraft_type"], int(row["flight_no"]))
            grouped[key].append(row)

    solution = Solution()
    uid_by_key: dict[tuple[str, int], str] = {}
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        aircraft_type, flight_no = key
        rows = sorted(grouped[key], key=lambda row: int(row["stop_order"]))
        orders = [int(row["stop_order"]) for row in rows]
        if orders != list(range(len(rows))):
            raise ValueError(f"{key}: stop_order must be contiguous from 0")
        if len(rows) < 3:
            raise ValueError(f"{key}: route must contain airport, sea stop, airport")
        base = rows[0]["facility_id"]
        if rows[-1]["facility_id"] != base:
            raise ValueError(f"{key}: route must return to its origin airport")

        uid = f"{uid_prefix}-{aircraft_type}-{flight_no:04d}"
        uid_by_key[key] = uid
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport=base,
            aircraft_type=aircraft_type,
            sea_stops=[
                SeaStop(
                    facility_id=row["facility_id"],
                    refuel=bool(int(row["refuel"])),
                )
                for row in rows[1:-1]
            ],
        )

    with assignments_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            person_id = row["person_id"]
            key = (row["aircraft_type"], int(row["flight_no"]))
            if key not in uid_by_key:
                raise ValueError(f"{person_id}: assignment references unknown flight {key}")
            if person_id in solution.assignments:
                raise ValueError(f"duplicate assignment for {person_id}")
            solution.assignments[person_id] = Assignment(
                person_id=person_id,
                flight_uid=uid_by_key[key],
                pickup_index=int(row["pickup_stop_order"]),
                delivery_index=int(row["delivery_stop_order"]),
            )

    return solution
