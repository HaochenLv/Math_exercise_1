from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from ortools.sat.python import cp_model

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, Assignment, PersonRequest, ProblemData, Solution
from helicopter_planner.evaluation.metrics import (
    SolutionMetrics,
    evaluate_solution,
    leg_minutes,
    stop_minutes,
)


@dataclass(frozen=True)
class PackingOption:
    person_id: str
    flight_uid: str
    pickup_index: int
    delivery_index: int
    passenger_travel_minutes: int

    @property
    def covered_legs(self) -> tuple[int, ...]:
        return tuple(range(self.pickup_index, self.delivery_index))


@dataclass(frozen=True)
class WarmStartPackingResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    extra_request_count: int
    candidate_request_count: int
    candidate_option_count: int
    packed_extra_count: int
    packed_by_kind: dict[str, int]
    remaining_by_kind: dict[str, int]
    candidate_options_by_person: dict[str, int]
    cp_status: str


def classify_q2_request(request: PersonRequest) -> str:
    origin_land = request.origin_id == "LAND" or request.origin_id in AIRPORTS
    destination_land = (
        request.destination_id == "LAND" or request.destination_id in AIRPORTS
    )
    if origin_land and not destination_land:
        return "outbound"
    if not origin_land and destination_land:
        return "return"
    if not origin_land and not destination_land:
        return "shuttle"
    return "land_to_land"


def _actual_endpoint(location: str, base_airport: str) -> str:
    return base_airport if location == "LAND" else location


def _leg_loads(solution: Solution, flight_uid: str) -> list[int]:
    flight = solution.flights[flight_uid]
    loads = [0] * (len(flight.full_route()) - 1)
    for assignment in solution.assignments.values():
        if assignment.flight_uid != flight_uid:
            continue
        for leg in range(assignment.pickup_index, assignment.delivery_index):
            loads[leg] += 1
    return loads


def build_onboard_trace(problem: ProblemData, solution: Solution) -> list[dict[str, object]]:
    assignments_by_flight: dict[str, list[Assignment]] = defaultdict(list)
    for assignment in solution.assignments.values():
        assignments_by_flight[assignment.flight_uid].append(assignment)

    rows: list[dict[str, object]] = []
    for uid in sorted(solution.flights):
        flight = solution.flights[uid]
        route = flight.full_route()
        capacity = problem.aircraft_types[flight.aircraft_type].seats
        leg_loads = _leg_loads(solution, uid)
        assignments = assignments_by_flight.get(uid, [])

        for stop_index, location in enumerate(route):
            onboard_before = leg_loads[stop_index - 1] if stop_index > 0 else 0
            dropoffs = sum(a.delivery_index == stop_index for a in assignments)
            pickups = sum(a.pickup_index == stop_index for a in assignments)
            onboard_after = (
                leg_loads[stop_index] if stop_index < len(leg_loads) else 0
            )
            if onboard_after != onboard_before - dropoffs + pickups:
                raise AssertionError(
                    f"{uid} stop {stop_index}: load accounting mismatch "
                    f"{onboard_before}-{dropoffs}+{pickups}!={onboard_after}"
                )
            if onboard_after > capacity:
                raise AssertionError(
                    f"{uid} stop {stop_index}: capacity exceeded "
                    f"{onboard_after}>{capacity}"
                )
            rows.append(
                {
                    "flight_uid": uid,
                    "aircraft_type": flight.aircraft_type,
                    "base_airport": flight.base_airport,
                    "stop_index": stop_index,
                    "location": location,
                    "onboard_before": onboard_before,
                    "dropoffs": dropoffs,
                    "pickups": pickups,
                    "onboard_after": onboard_after,
                    "capacity": capacity,
                }
            )
    return rows


class Q2WarmStartPacker:
    """Q2 B0A: pack extra Q2 requests into a fixed Q1 route skeleton.

    The Q1 flights, aircraft types, sea-stop order, and refuel decisions are
    frozen. Only assignments for Q2 requests not already present in the Q1
    solution may be added. Therefore every accepted passenger has zero
    incremental aircraft time, distance, fuel, and flight count.

    Phase 1 maximizes the number of newly packed passengers globally under
    per-leg seat capacities. Phase 2 fixes that maximum count and minimizes
    added passenger travel time.
    """

    def __init__(
        self,
        *,
        max_time_seconds: float = 30.0,
        random_seed: int = 0,
    ) -> None:
        self.max_time_seconds = max_time_seconds
        self.random_seed = random_seed

    def pack(
        self,
        problem: ProblemData,
        q1_solution: Solution,
    ) -> WarmStartPackingResult:
        solution = copy.deepcopy(q1_solution)
        unknown = sorted(set(solution.assignments) - set(problem.requests))
        if unknown:
            raise ValueError(
                "Q1 warm-start contains passengers absent from Q2: "
                + ", ".join(unknown[:10])
            )

        starting_check = check_solution(
            problem, solution, require_all_requests=False
        )
        if not starting_check.ok:
            raise ValueError(
                "Q1 warm-start is not a valid partial Q2 solution:\n"
                + "\n".join(starting_check.errors)
            )

        starting_metrics = evaluate_solution(problem, solution)
        base_loads = {
            uid: _leg_loads(solution, uid) for uid in solution.flights
        }
        extra_ids = sorted(set(problem.requests) - set(solution.assignments))

        options: list[PackingOption] = []
        options_by_person: dict[str, list[int]] = defaultdict(list)
        options_by_flight_leg: dict[tuple[str, int], list[int]] = defaultdict(list)

        for person_id in extra_ids:
            request = problem.requests[person_id]
            for option in self._candidate_options(problem, solution, request):
                option_index = len(options)
                options.append(option)
                options_by_person[person_id].append(option_index)
                for leg in option.covered_legs:
                    options_by_flight_leg[(option.flight_uid, leg)].append(
                        option_index
                    )

        if not options:
            metrics = evaluate_solution(problem, solution)
            remaining = Counter(
                classify_q2_request(problem.requests[pid]) for pid in extra_ids
            )
            return WarmStartPackingResult(
                solution=solution,
                starting_metrics=starting_metrics,
                metrics=metrics,
                extra_request_count=len(extra_ids),
                candidate_request_count=0,
                candidate_option_count=0,
                packed_extra_count=0,
                packed_by_kind={},
                remaining_by_kind=dict(sorted(remaining.items())),
                candidate_options_by_person={},
                cp_status="NO_CANDIDATES",
            )

        model = cp_model.CpModel()
        variables = [
            model.new_bool_var(f"x_{index}") for index in range(len(options))
        ]

        for person_id, indices in options_by_person.items():
            model.add(sum(variables[i] for i in indices) <= 1)

        for uid, flight in solution.flights.items():
            capacity = problem.aircraft_types[flight.aircraft_type].seats
            for leg, base_load in enumerate(base_loads[uid]):
                candidate_indices = options_by_flight_leg.get((uid, leg), [])
                if not candidate_indices:
                    continue
                model.add(
                    base_load + sum(variables[i] for i in candidate_indices)
                    <= capacity
                )

        served_expr = sum(variables)
        model.maximize(served_expr)
        solver = self._new_solver()
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "Q2 B0A max-cardinality packing was not proven optimal: "
                + self._status_name(status)
            )
        max_served = int(round(solver.objective_value))

        model.add(served_expr == max_served)
        model.minimize(
            sum(
                option.passenger_travel_minutes * variables[index]
                for index, option in enumerate(options)
            )
        )
        solver = self._new_solver()
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "Q2 B0A travel-time tie-break was not proven optimal: "
                + self._status_name(status)
            )

        selected = [
            options[index]
            for index, variable in enumerate(variables)
            if solver.value(variable)
        ]
        for option in selected:
            solution.assignments[option.person_id] = Assignment(
                person_id=option.person_id,
                flight_uid=option.flight_uid,
                pickup_index=option.pickup_index,
                delivery_index=option.delivery_index,
            )

        final_check = check_solution(
            problem, solution, require_all_requests=False
        )
        if not final_check.ok:
            raise AssertionError(
                "Q2 B0A produced an invalid partial solution:\n"
                + "\n".join(final_check.errors)
            )

        metrics = evaluate_solution(problem, solution)
        self._assert_zero_aircraft_cost(starting_metrics, metrics)
        build_onboard_trace(problem, solution)

        packed_ids = {option.person_id for option in selected}
        packed_by_kind = Counter(
            classify_q2_request(problem.requests[pid]) for pid in packed_ids
        )
        remaining_by_kind = Counter(
            classify_q2_request(problem.requests[pid])
            for pid in extra_ids
            if pid not in packed_ids
        )

        return WarmStartPackingResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=metrics,
            extra_request_count=len(extra_ids),
            candidate_request_count=len(options_by_person),
            candidate_option_count=len(options),
            packed_extra_count=len(selected),
            packed_by_kind=dict(sorted(packed_by_kind.items())),
            remaining_by_kind=dict(sorted(remaining_by_kind.items())),
            candidate_options_by_person={
                pid: len(indices)
                for pid, indices in sorted(options_by_person.items())
            },
            cp_status=self._status_name(status),
        )

    @staticmethod
    def _status_name(status: int) -> str:
        names = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
        }
        return names.get(status, str(status))

    def _new_solver(self) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.random_seed
        return solver

    def _candidate_options(
        self,
        problem: ProblemData,
        solution: Solution,
        request: PersonRequest,
    ) -> Iterable[PackingOption]:
        for uid in sorted(solution.flights):
            flight = solution.flights[uid]
            route = flight.full_route()
            expected_origin = _actual_endpoint(
                request.origin_id, flight.base_airport
            )
            expected_destination = _actual_endpoint(
                request.destination_id, flight.base_airport
            )

            for pickup_index, location in enumerate(route[:-1]):
                if location != expected_origin:
                    continue
                delivery_index = None
                for index in range(pickup_index + 1, len(route)):
                    if route[index] == expected_destination:
                        delivery_index = index
                        break
                if delivery_index is None:
                    continue
                yield PackingOption(
                    person_id=request.person_id,
                    flight_uid=uid,
                    pickup_index=pickup_index,
                    delivery_index=delivery_index,
                    passenger_travel_minutes=self._passenger_minutes(
                        problem,
                        flight,
                        pickup_index,
                        delivery_index,
                    ),
                )

    @staticmethod
    def _passenger_minutes(
        problem: ProblemData,
        flight,
        pickup_index: int,
        delivery_index: int,
    ) -> int:
        route = flight.full_route()
        spec = problem.aircraft_types[flight.aircraft_type]
        flight_minutes = [
            leg_minutes(problem.distance(a, b), spec.speed_kmh)
            for a, b in zip(route, route[1:])
        ]
        stop_minutes_list = [stop_minutes(stop.refuel) for stop in flight.sea_stops]
        return sum(flight_minutes[pickup_index:delivery_index]) + sum(
            stop_minutes_list[pickup_index : delivery_index - 1]
        )

    @staticmethod
    def _assert_zero_aircraft_cost(
        start: SolutionMetrics,
        end: SolutionMetrics,
    ) -> None:
        exact_fields = (
            "total_aircraft_usage_minutes",
            "number_of_flights",
        )
        float_fields = (
            "total_fuel_consumption_kg",
            "available_seat_km",
        )
        for field in exact_fields:
            if getattr(start, field) != getattr(end, field):
                raise AssertionError(
                    f"Q2 B0A changed fixed-route metric {field}: "
                    f"{getattr(start, field)} -> {getattr(end, field)}"
                )
        for field in float_fields:
            if abs(float(getattr(start, field)) - float(getattr(end, field))) > 1e-6:
                raise AssertionError(
                    f"Q2 B0A changed fixed-route metric {field}: "
                    f"{getattr(start, field)} -> {getattr(end, field)}"
                )
