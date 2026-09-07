from __future__ import annotations

import csv
from pathlib import Path

from helicopter_planner.domain import AIRCRAFT_TYPES, PersonRequest, ProblemData


def load_distances(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0] != "from_id":
            raise ValueError("distances.csv must start with from_id")
        destinations = reader.fieldnames[1:]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            origin = row["from_id"]
            matrix[origin] = {d: float(row[d]) for d in destinations}
    if set(matrix) != set(destinations):
        raise ValueError("distance matrix row/column location sets differ")
    return matrix


def _load_basic_requests(path: str | Path, *, label: str) -> dict[str, PersonRequest]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expected = ["person_id", "origin_id", "destination_id"]
        if reader.fieldnames != expected:
            raise ValueError(f"{label} request header must be {expected}")
        requests: dict[str, PersonRequest] = {}
        for row in reader:
            req = PersonRequest(**row)
            if req.person_id in requests:
                raise ValueError(f"duplicate person_id: {req.person_id}")
            requests[req.person_id] = req
    return requests


def load_q1_requests(path: str | Path) -> dict[str, PersonRequest]:
    return _load_basic_requests(path, label="Q1")


def load_q2_requests(path: str | Path) -> dict[str, PersonRequest]:
    return _load_basic_requests(path, label="Q2")


def load_q1_problem(distances_path: str | Path, people_path: str | Path) -> ProblemData:
    return ProblemData(
        distances=load_distances(distances_path),
        requests=load_q1_requests(people_path),
        aircraft_types=AIRCRAFT_TYPES,
    )


def load_q2_problem(distances_path: str | Path, people_path: str | Path) -> ProblemData:
    return ProblemData(
        distances=load_distances(distances_path),
        requests=load_q2_requests(people_path),
        aircraft_types=AIRCRAFT_TYPES,
    )
