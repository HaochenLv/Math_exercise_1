from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

AIRPORTS = frozenset({"A01", "A02", "A03"})
REFUEL_FACILITIES = frozenset({"F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050"})

@dataclass(frozen=True)
class AircraftTypeSpec:
    type_id: str
    seats: int
    speed_kmh: float
    fuel_rate_kg_per_km: float
    tank_capacity_kg: float
    min_safe_fuel_kg: float

AIRCRAFT_TYPES: Mapping[str, AircraftTypeSpec] = {
    "T1": AircraftTypeSpec("T1", 12, 250.0, 3.4, 1000.0, 150.0),
    "T2": AircraftTypeSpec("T2", 16, 220.0, 2.5, 1150.0, 150.0),
    "T3": AircraftTypeSpec("T3", 19, 190.0, 2.9, 1600.0, 200.0),
}

@dataclass(frozen=True)
class PersonRequest:
    person_id: str
    origin_id: str
    destination_id: str

@dataclass(frozen=True)
class SeaStop:
    facility_id: str
    refuel: bool = False

@dataclass
class FlightPlan:
    flight_uid: str
    base_airport: str
    aircraft_type: str
    sea_stops: list[SeaStop] = field(default_factory=list)

    def full_route(self) -> list[str]:
        return [self.base_airport, *(s.facility_id for s in self.sea_stops), self.base_airport]

@dataclass(frozen=True)
class Assignment:
    person_id: str
    flight_uid: str
    pickup_index: int
    delivery_index: int

@dataclass
class Solution:
    flights: dict[str, FlightPlan] = field(default_factory=dict)
    assignments: dict[str, Assignment] = field(default_factory=dict)

@dataclass(frozen=True)
class ProblemData:
    distances: Mapping[str, Mapping[str, float]]
    requests: Mapping[str, PersonRequest]
    aircraft_types: Mapping[str, AircraftTypeSpec] = field(default_factory=lambda: AIRCRAFT_TYPES)

    def distance(self, origin: str, destination: str) -> float:
        return float(self.distances[origin][destination])
