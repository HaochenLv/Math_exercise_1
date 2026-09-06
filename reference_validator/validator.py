"""Independent Q1/Q2-style submission validator.

This module intentionally does NOT import helicopter_planner checker/evaluator code.
It duplicates critical rules so a bug in the solver-side implementation is less likely
be shared by the final validator.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

AIRPORTS = {"A01", "A02", "A03"}
REFUEL = {"F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050"}
SPECS = {
    "T1": dict(seats=12, speed=250.0, rate=3.4, tank=1000.0, reserve=150.0),
    "T2": dict(seats=16, speed=220.0, rate=2.5, tank=1150.0, reserve=150.0),
    "T3": dict(seats=19, speed=190.0, rate=2.9, tank=1600.0, reserve=200.0),
}

@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]
    metrics: dict[str, float | int]

def _read_distances(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames[1:]
        return {row["from_id"]: {c: float(row[c]) for c in cols} for row in reader}

def _read_requests(path: str | Path) -> dict[str, tuple[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return {r["person_id"]: (r["origin_id"], r["destination_id"]) for r in csv.DictReader(f)}

def _ceil_minutes(distance: float, speed: float) -> int:
    return math.ceil(distance / speed * 60.0 - 1e-12)

def validate_submission(distances_path: str | Path, requests_path: str | Path, routes_path: str | Path, assignments_path: str | Path, *, require_all_requests: bool = True) -> ValidationReport:
    D = _read_distances(distances_path)
    requests = _read_requests(requests_path)
    errors: list[str] = []

    with Path(routes_path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for r in rows:
        try:
            key = (r["aircraft_type"], int(r["flight_no"]))
            order = int(r["stop_order"])
            refuel = int(r["refuel"])
        except Exception:
            errors.append(f"malformed route row: {r}")
            continue
        r = dict(r)
        r["_order"] = order
        r["_refuel"] = refuel
        groups.setdefault(key, []).append(r)

    routes: dict[tuple[str, int], list[str]] = {}
    refuel_flags: dict[tuple[str, int], list[int]] = {}
    for key, rs in groups.items():
        rs.sort(key=lambda x: x["_order"])
        orders = [r["_order"] for r in rs]
        if orders != list(range(len(rs))):
            errors.append(f"{key}: stop_order not contiguous from 0")
        locs = [r["facility_id"] for r in rs]
        flags = [r["_refuel"] for r in rs]
        routes[key] = locs
        refuel_flags[key] = flags
        if len(locs) < 2 or locs[0] != locs[-1] or locs[0] not in AIRPORTS:
            errors.append(f"{key}: first/last must be same airport")
        if len(locs) - 2 > 5:
            errors.append(f"{key}: more than 5 sea landings")
        for i, loc in enumerate(locs):
            if i in (0, len(locs) - 1):
                if flags[i] != 0:
                    errors.append(f"{key}: airport refuel must be 0")
            else:
                if not loc.startswith("F"):
                    errors.append(f"{key}: intermediate stop is not sea facility")
                if flags[i] not in (0, 1):
                    errors.append(f"{key}: refuel must be 0/1")
                if flags[i] == 1 and loc not in REFUEL:
                    errors.append(f"{key}: illegal refuel at {loc}")

        if key[0] not in SPECS:
            errors.append(f"{key}: invalid aircraft type")
            continue
        s = SPECS[key[0]]
        fuel = s["tank"]
        for i in range(len(locs) - 1):
            try:
                fuel -= D[locs[i]][locs[i + 1]] * s["rate"]
            except KeyError:
                errors.append(f"{key}: unknown location in distance matrix")
                continue
            if fuel + 1e-9 < s["reserve"]:
                errors.append(f"{key}: fuel below reserve on arrival {locs[i + 1]}")
            if i + 1 < len(locs) - 1 and flags[i + 1] == 1:
                fuel = s["tank"]

    with Path(assignments_path).open(newline="", encoding="utf-8-sig") as f:
        arows = list(csv.DictReader(f))
    seen: set[str] = set()
    by_flight: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
    for r in arows:
        pid = r["person_id"]
        if pid in seen:
            errors.append(f"duplicate assignment for {pid}")
        seen.add(pid)
        try:
            key = (r["aircraft_type"], int(r["flight_no"]))
            p = int(r["pickup_stop_order"])
            q = int(r["delivery_stop_order"])
        except Exception:
            errors.append(f"malformed assignment row: {r}")
            continue
        if key not in routes:
            errors.append(f"{pid}: references unknown flight {key}")
            continue
        if pid not in requests:
            errors.append(f"{pid}: unknown person")
            continue
        route = routes[key]
        if not (0 <= p < q < len(route)):
            errors.append(f"{pid}: invalid stop indexes")
            continue
        base = route[0]
        origin, dest = requests[pid]
        origin = base if origin == "LAND" else origin
        dest = base if dest == "LAND" else dest
        if route[p] != origin:
            errors.append(f"{pid}: pickup mismatch")
        if route[q] != dest:
            errors.append(f"{pid}: delivery mismatch")
        try:
            first_q = route.index(dest, p + 1)
        except ValueError:
            first_q = None
        if first_q != q:
            errors.append(f"{pid}: delivery is not first destination stop after pickup")
        by_flight.setdefault(key, []).append((pid, p, q))

    if require_all_requests:
        missing = set(requests) - seen
        extra = seen - set(requests)
        if missing:
            errors.append(f"missing persons: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            errors.append(f"extra persons: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")

    total_air = 0
    total_pass = 0
    fuel_total = 0.0
    passenger_km = 0.0
    available_km = 0.0
    for key, route in routes.items():
        if key[0] not in SPECS:
            continue
        s = SPECS[key[0]]
        distances = [D[route[i]][route[i + 1]] for i in range(len(route) - 1)]
        leg_times = [_ceil_minutes(d, s["speed"]) for d in distances]
        flags = refuel_flags[key]
        stop_times = [20 if flags[i] else 10 for i in range(1, len(route) - 1)]
        total_air += sum(leg_times) + sum(stop_times)
        fuel_total += sum(distances) * s["rate"]
        available_km += sum(distances) * s["seats"]
        loads = [0] * len(distances)
        for pid, p, q in by_flight.get(key, []):
            for leg in range(p, q):
                loads[leg] += 1
            total_pass += sum(leg_times[p:q]) + sum(stop_times[p:q - 1])
        for i, load in enumerate(loads):
            if load > s["seats"]:
                errors.append(f"{key}: seat capacity exceeded on leg {i}: {load}>{s['seats']}")
            passenger_km += load * distances[i]

    metrics = {
        "total_aircraft_usage_minutes": total_air,
        "total_passenger_travel_minutes": total_pass,
        "number_of_flights": len(routes),
        "total_fuel_consumption_kg": fuel_total,
        "passenger_km": passenger_km,
        "available_seat_km": available_km,
        "seat_utilization": passenger_km / available_km if available_km else 0.0,
    }
    return ValidationReport(ok=not errors, errors=tuple(errors), metrics=metrics)
