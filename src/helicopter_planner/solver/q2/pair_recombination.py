from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

from ortools.sat.python import cp_model

from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import AIRPORTS, ProblemData, Solution
from helicopter_planner.evaluation.metrics import SolutionMetrics, evaluate_solution
from helicopter_planner.solver.q1.route_builder import BuiltRoute, Q1RouteBuilder
from helicopter_planner.solver.q2.tail_elimination import (
    OptimizedPassengerRoute,
    Q2PassengerSetRouteOptimizer,
    Q2ResidualTailEliminationSolver,
    TailEliminationDecision,
)

RequestKey = tuple[str, str]


@dataclass(frozen=True)
class PairRecombinationDecision:
    iteration: int
    source_flight_uids: tuple[str, str]
    pooled_group_count: int
    union_sea_facilities: tuple[str, ...]
    cp_status: str
    cp_objective_aircraft_minutes: int
    old_aircraft_usage_minutes: int
    new_aircraft_usage_minutes: int
    immediate_aircraft_savings_minutes: int
    total_iteration_aircraft_savings_minutes: int
    passenger_minutes_delta: int
    fuel_kg_delta: float
    old_flight_count: int
    new_flight_count: int
    new_aircraft_types: tuple[str, ...]
    new_base_airports: tuple[str, ...]
    new_service_orders: tuple[tuple[str, ...], ...]
    tail_polish_moves: int


@dataclass(frozen=True)
class PairIterationDiagnostics:
    iteration: int
    residual_flight_count: int
    candidate_pair_count: int
    eligible_pair_count: int
    cp_pair_count: int
    cp_optimal_count: int
    cp_feasible_not_proven_count: int
    cp_no_solution_count: int
    improving_pair_count: int
    route_template_count: int


@dataclass(frozen=True)
class Q2PairRecombinationResult:
    solution: Solution
    starting_metrics: SolutionMetrics
    metrics: SolutionMetrics
    decisions: tuple[PairRecombinationDecision, ...]
    iteration_diagnostics: tuple[PairIterationDiagnostics, ...]
    unlocked_tail_eliminations: tuple[TailEliminationDecision, ...]
    converged: bool


@dataclass(frozen=True)
class _FlightSummary:
    flight_uid: str
    person_ids: tuple[str, ...]
    profile: tuple[tuple[str, str, int], ...]
    sea_facilities: frozenset[str]
    cost: tuple[int, int, float]


@dataclass(frozen=True)
class _RouteTemplate:
    route: BuiltRoute
    serviceable_groups: tuple[int, ...]
    leg_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _PairSolveStats:
    status: str
    template_count: int
    eligible: bool


@dataclass(frozen=True)
class _PairCandidate:
    source_uids: tuple[str, str]
    passenger_groups: tuple[tuple[str, ...], ...]
    optimized_routes: tuple[OptimizedPassengerRoute, ...]
    old_cost: tuple[int, int, float]
    cp_status: str
    cp_objective_aircraft_minutes: int
    template_count: int
    pooled_group_count: int
    union_sea_facilities: tuple[str, ...]

    @property
    def new_aircraft_minutes(self) -> int:
        return sum(route.aircraft_minutes for route in self.optimized_routes)

    @property
    def new_passenger_minutes(self) -> int:
        return sum(route.passenger_minutes for route in self.optimized_routes)

    @property
    def new_fuel_kg(self) -> float:
        return sum(route.fuel_kg for route in self.optimized_routes)

    @property
    def savings(self) -> int:
        return self.old_cost[0] - self.new_aircraft_minutes

    @property
    def passenger_delta(self) -> int:
        return self.new_passenger_minutes - self.old_cost[1]

    @property
    def fuel_delta(self) -> float:
        return self.new_fuel_kg - self.old_cost[2]


class Q2ExactPairRecombinationSolver:
    """Q2 B3A: exact two-sortie OD-count recombination on residual flights.

    B1/B2 keep most passenger composition inherited from B0B. B3A pools the
    complete OD demand carried by a pair of residual ``q2b0b-*`` sorties and
    repartitions integer OD counts into one or two rebuilt sorties.

    For a fixed pair, every relevant distinct service-facility set induced by a
    subset of the pooled OD groups is enumerated (up to five service
    facilities), as are every service order, base airport and T1/T2/T3. The
    common route builder supplies the minimum-time legal refuelling pattern. A
    CP-SAT model chooses up to two route templates and assigns integer OD counts
    while enforcing capacity on every actual flight leg. Passenger counts of a
    single OD may therefore be split arbitrarily between the rebuilt sorties.

    Candidate-pair generation is a neighborhood restriction, not a feasibility
    claim: shared-facility pairs are always included, together with the nearest
    residual neighbors, and pairs whose pooled sea-facility union exceeds the
    configured bound are skipped. Within every evaluated pair the primary model
    is exact over the common distinct-service route family whenever CP-SAT
    returns OPTIMAL.

    After every accepted strict improvement the common B1 single-route/tail
    polish is rerun, then the pair neighborhood is rebuilt from the new state.
    """

    def __init__(
        self,
        *,
        route_optimizer: Q2PassengerSetRouteOptimizer | None = None,
        route_builder: Q1RouteBuilder | None = None,
        neighbor_count: int = 8,
        max_union_facilities: int = 6,
        cp_time_limit_seconds: float = 0.75,
        max_iterations: int = 12,
        tail_max_hosts_per_donor: int = 32,
    ) -> None:
        builder = route_builder or Q1RouteBuilder()
        self.route_optimizer = route_optimizer or Q2PassengerSetRouteOptimizer(builder)
        self.route_builder = builder
        self.neighbor_count = neighbor_count
        self.max_union_facilities = max_union_facilities
        self.cp_time_limit_seconds = cp_time_limit_seconds
        self.max_iterations = max_iterations
        self.tail_max_hosts_per_donor = tail_max_hosts_per_donor
        self._route_cache: dict[tuple[str, tuple[str, ...], str], BuiltRoute | None] = {}
        self._template_cache: dict[tuple[RequestKey, ...], tuple[_RouteTemplate, ...]] = {}

    def solve(self, problem: ProblemData, starting_solution: Solution) -> Q2PairRecombinationResult:
        solution = copy.deepcopy(starting_solution)
        start_check = check_solution(problem, solution, require_all_requests=True)
        if not start_check.ok:
            raise ValueError("B3 starting solution invalid:\n" + "\n".join(start_check.errors))
        starting_metrics = evaluate_solution(problem, solution)

        decisions: list[PairRecombinationDecision] = []
        diagnostics: list[PairIterationDiagnostics] = []
        unlocked_eliminations: list[TailEliminationDecision] = []
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            summaries = self._summaries(problem, solution)
            pairs = self._candidate_pairs(problem, summaries)
            best: _PairCandidate | None = None
            eligible = 0
            cp_pair_count = 0
            cp_optimal = 0
            cp_feasible = 0
            cp_none = 0
            improving = 0
            template_total = 0

            for pair in pairs:
                candidate, stats = self._solve_pair(problem, solution, summaries, pair)
                if not stats.eligible:
                    continue
                eligible += 1
                template_total += stats.template_count
                if stats.status != "NOT_RUN":
                    cp_pair_count += 1
                if stats.status == "OPTIMAL":
                    cp_optimal += 1
                elif stats.status == "FEASIBLE":
                    cp_feasible += 1
                elif stats.status in {"INFEASIBLE", "UNKNOWN", "MODEL_INVALID"}:
                    cp_none += 1
                if candidate is None:
                    continue
                improving += 1
                if best is None or self._candidate_key(candidate) < self._candidate_key(best):
                    best = candidate

            diagnostics.append(
                PairIterationDiagnostics(
                    iteration=iteration,
                    residual_flight_count=len(summaries),
                    candidate_pair_count=len(pairs),
                    eligible_pair_count=eligible,
                    cp_pair_count=cp_pair_count,
                    cp_optimal_count=cp_optimal,
                    cp_feasible_not_proven_count=cp_feasible,
                    cp_no_solution_count=cp_none,
                    improving_pair_count=improving,
                    route_template_count=template_total,
                )
            )

            if best is None or best.savings <= 0:
                converged = True
                break

            before = evaluate_solution(problem, solution)
            self._apply_candidate(problem, solution, best)
            check = check_solution(problem, solution, require_all_requests=True)
            if not check.ok:
                raise AssertionError("B3 invalid after pair recombination:\n" + "\n".join(check.errors))

            before_polish = evaluate_solution(problem, solution)
            tail_result = Q2ResidualTailEliminationSolver(
                route_optimizer=self.route_optimizer,
                max_hosts_per_donor=self.tail_max_hosts_per_donor,
            ).solve(problem, solution)
            solution = tail_result.solution
            unlocked_eliminations.extend(tail_result.tail_eliminations)
            after = evaluate_solution(problem, solution)
            total_savings = before.total_aircraft_usage_minutes - after.total_aircraft_usage_minutes
            if total_savings <= 0:
                raise AssertionError("accepted B3 iteration did not improve aircraft time")

            decisions.append(
                PairRecombinationDecision(
                    iteration=iteration,
                    source_flight_uids=best.source_uids,
                    pooled_group_count=best.pooled_group_count,
                    union_sea_facilities=best.union_sea_facilities,
                    cp_status=best.cp_status,
                    cp_objective_aircraft_minutes=best.cp_objective_aircraft_minutes,
                    old_aircraft_usage_minutes=best.old_cost[0],
                    new_aircraft_usage_minutes=best.new_aircraft_minutes,
                    immediate_aircraft_savings_minutes=best.savings,
                    total_iteration_aircraft_savings_minutes=total_savings,
                    passenger_minutes_delta=best.passenger_delta,
                    fuel_kg_delta=best.fuel_delta,
                    old_flight_count=2,
                    new_flight_count=len(best.optimized_routes),
                    new_aircraft_types=tuple(x.route.aircraft_type for x in best.optimized_routes),
                    new_base_airports=tuple(x.route.base_airport for x in best.optimized_routes),
                    new_service_orders=tuple(x.route.service_order for x in best.optimized_routes),
                    tail_polish_moves=len(tail_result.tail_eliminations),
                )
            )

            if before_polish.total_aircraft_usage_minutes < after.total_aircraft_usage_minutes:
                raise AssertionError("B3 tail polish increased aircraft time")

        final_check = check_solution(problem, solution, require_all_requests=True)
        if not final_check.ok:
            raise AssertionError("B3 final solution invalid:\n" + "\n".join(final_check.errors))
        return Q2PairRecombinationResult(
            solution=solution,
            starting_metrics=starting_metrics,
            metrics=evaluate_solution(problem, solution),
            decisions=tuple(decisions),
            iteration_diagnostics=tuple(diagnostics),
            unlocked_tail_eliminations=tuple(unlocked_eliminations),
            converged=converged,
        )

    def _summaries(self, problem: ProblemData, solution: Solution) -> dict[str, _FlightSummary]:
        summaries: dict[str, _FlightSummary] = {}
        for uid in sorted(solution.flights):
            if not uid.startswith("q2b0b-"):
                continue
            person_ids = tuple(self._person_ids(solution, uid))
            if not person_ids:
                continue
            profile = self.route_optimizer.profile(problem, person_ids)
            sea = frozenset(
                location
                for origin, destination, _ in profile
                for location in (origin, destination)
                if location.startswith("F")
            )
            if not sea:
                continue
            summaries[uid] = _FlightSummary(
                flight_uid=uid,
                person_ids=person_ids,
                profile=profile,
                sea_facilities=sea,
                cost=self._flight_cost(problem, solution, uid),
            )
        return summaries

    def _candidate_pairs(
        self,
        problem: ProblemData,
        summaries: dict[str, _FlightSummary],
    ) -> tuple[tuple[str, str], ...]:
        uids = sorted(summaries)
        pairs: set[tuple[str, str]] = set()
        for uid in uids:
            left = summaries[uid]
            ranked: list[tuple[float, str]] = []
            for other_uid in uids:
                if other_uid == uid:
                    continue
                right = summaries[other_uid]
                distance = min(
                    problem.distance(a, b)
                    for a in left.sea_facilities
                    for b in right.sea_facilities
                )
                ranked.append((distance, other_uid))
            for _, other_uid in sorted(ranked)[: self.neighbor_count]:
                pairs.add(tuple(sorted((uid, other_uid))))

        for left_index, left_uid in enumerate(uids):
            left = summaries[left_uid]
            for right_uid in uids[left_index + 1 :]:
                right = summaries[right_uid]
                if left.sea_facilities & right.sea_facilities:
                    pairs.add((left_uid, right_uid))
        return tuple(sorted(pairs))

    def _solve_pair(
        self,
        problem: ProblemData,
        solution: Solution,
        summaries: dict[str, _FlightSummary],
        pair: tuple[str, str],
    ) -> tuple[_PairCandidate | None, _PairSolveStats]:
        left = summaries[pair[0]]
        right = summaries[pair[1]]
        union = tuple(sorted(left.sea_facilities | right.sea_facilities))
        if len(union) > self.max_union_facilities:
            return None, _PairSolveStats(status="NOT_RUN", template_count=0, eligible=False)

        pooled_ids = tuple(sorted((*left.person_ids, *right.person_ids)))
        people_by_key: dict[RequestKey, list[str]] = {}
        for pid in pooled_ids:
            req = problem.requests[pid]
            people_by_key.setdefault((req.origin_id, req.destination_id), []).append(pid)
        group_keys = tuple(sorted(people_by_key))
        demand = tuple(len(people_by_key[key]) for key in group_keys)
        templates = tuple(
            template
            for template in self._route_templates(problem, group_keys)
            if template.route.trip_minutes < left.cost[0] + right.cost[0]
        )
        if not templates:
            return None, _PairSolveStats(status="INFEASIBLE", template_count=0, eligible=True)

        model = cp_model.CpModel()
        route_slots = 2
        z = {}
        x = {}
        used = []
        choices = []

        serviceable_by_group = [
            tuple(index for index, template in enumerate(templates) if group_index in template.serviceable_groups)
            for group_index in range(len(group_keys))
        ]
        if any(not indices for indices in serviceable_by_group):
            return None, _PairSolveStats(status="INFEASIBLE", template_count=len(templates), eligible=True)

        for slot in range(route_slots):
            empty = model.new_bool_var(f"empty_{slot}")
            template_vars = []
            for template_index in range(len(templates)):
                var = model.new_bool_var(f"z_{slot}_{template_index}")
                z[slot, template_index] = var
                template_vars.append(var)
            model.add_exactly_one([empty, *template_vars])
            u = model.new_bool_var(f"used_{slot}")
            model.add(u + empty == 1)
            used.append(u)

            choice = model.new_int_var(0, len(templates), f"choice_{slot}")
            model.add(
                choice
                == sum((template_index + 1) * z[slot, template_index] for template_index in range(len(templates)))
            )
            choices.append(choice)

            for group_index, count in enumerate(demand):
                var = model.new_int_var(0, count, f"x_{slot}_{group_index}")
                x[slot, group_index] = var
                model.add(
                    var
                    <= count
                    * sum(z[slot, template_index] for template_index in serviceable_by_group[group_index])
                )

            model.add(sum(x[slot, g] for g in range(len(group_keys))) >= u)

            for template_index, template in enumerate(templates):
                spec = problem.aircraft_types[template.route.aircraft_type]
                selected = z[slot, template_index]
                for leg_groups in template.leg_groups:
                    if not leg_groups:
                        continue
                    model.add(
                        sum(x[slot, group_index] for group_index in leg_groups) <= spec.seats
                    ).only_enforce_if(selected)

        model.add(used[0] >= used[1])
        model.add(sum(used) >= 1)
        model.add(choices[0] <= choices[1]).only_enforce_if(used[1])

        for group_index, count in enumerate(demand):
            model.add(sum(x[slot, group_index] for slot in range(route_slots)) == count)

        aircraft_expr = sum(
            template.route.trip_minutes * z[slot, template_index]
            for slot in range(route_slots)
            for template_index, template in enumerate(templates)
        )
        model.minimize(aircraft_expr)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.cp_time_limit_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        status_name = self._status_name(status)
        stats = _PairSolveStats(status=status_name, template_count=len(templates), eligible=True)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, stats

        old_aircraft = left.cost[0] + right.cost[0]
        cp_objective = int(round(solver.objective_value))
        if cp_objective >= old_aircraft:
            return None, stats

        cursors = {key: 0 for key in group_keys}
        passenger_groups: list[tuple[str, ...]] = []
        for slot in range(route_slots):
            if solver.value(used[slot]) == 0:
                continue
            ids: list[str] = []
            for group_index, key in enumerate(group_keys):
                count = solver.value(x[slot, group_index])
                start = cursors[key]
                ids.extend(people_by_key[key][start : start + count])
                cursors[key] += count
            if not ids:
                raise AssertionError("B3 selected an empty used route slot")
            passenger_groups.append(tuple(sorted(ids)))

        if any(cursors[key] != len(people_by_key[key]) for key in group_keys):
            raise AssertionError("B3 CP-SAT allocation did not cover pooled demand")

        optimized_routes: list[OptimizedPassengerRoute] = []
        for ids in passenger_groups:
            profile = self.route_optimizer.profile(problem, ids)
            optimized = self.route_optimizer.optimize(problem, profile)
            if optimized is None:
                return None, stats
            optimized_routes.append(optimized)

        candidate = _PairCandidate(
            source_uids=pair,
            passenger_groups=tuple(passenger_groups),
            optimized_routes=tuple(optimized_routes),
            old_cost=(
                old_aircraft,
                left.cost[1] + right.cost[1],
                left.cost[2] + right.cost[2],
            ),
            cp_status=status_name,
            cp_objective_aircraft_minutes=cp_objective,
            template_count=len(templates),
            pooled_group_count=len(group_keys),
            union_sea_facilities=union,
        )
        return (candidate if candidate.savings > 0 else None), stats

    def _route_templates(
        self,
        problem: ProblemData,
        group_keys: tuple[RequestKey, ...],
    ) -> tuple[_RouteTemplate, ...]:
        cached = self._template_cache.get(group_keys)
        if cached is not None:
            return cached

        service_sets: set[tuple[str, ...]] = set()
        for mask in range(1, 1 << len(group_keys)):
            facilities: set[str] = set()
            for group_index, (origin, destination) in enumerate(group_keys):
                if not (mask & (1 << group_index)):
                    continue
                if origin.startswith("F"):
                    facilities.add(origin)
                if destination.startswith("F"):
                    facilities.add(destination)
            if 1 <= len(facilities) <= 5:
                service_sets.add(tuple(sorted(facilities)))

        templates: list[_RouteTemplate] = []
        for service_set in sorted(service_sets, key=lambda item: (len(item), item)):
            for order in itertools.permutations(service_set):
                for base in sorted(AIRPORTS):
                    for aircraft_type in sorted(problem.aircraft_types):
                        route_key = (base, order, aircraft_type)
                        if route_key not in self._route_cache:
                            self._route_cache[route_key] = self.route_builder.build(
                                problem, base, order, aircraft_type
                            )
                        route = self._route_cache[route_key]
                        if route is None:
                            continue
                        template = self._make_template(group_keys, route)
                        if template is not None:
                            templates.append(template)

        result = tuple(templates)
        self._template_cache[group_keys] = result
        return result

    @staticmethod
    def _make_template(
        group_keys: tuple[RequestKey, ...],
        route: BuiltRoute,
    ) -> _RouteTemplate | None:
        leg_count = len(route.sea_stops) + 1
        serviceable: list[int] = []
        intervals: dict[int, tuple[int, int]] = {}

        for group_index, (origin, destination) in enumerate(group_keys):
            if origin == "LAND":
                pickup = 0
            elif origin in AIRPORTS:
                if origin != route.base_airport:
                    continue
                pickup = 0
            elif origin.startswith("F"):
                if origin not in route.service_order:
                    continue
                pickup = route.service_stop_index(origin)
            else:
                continue

            if destination == "LAND":
                delivery = leg_count
            elif destination in AIRPORTS:
                if destination != route.base_airport:
                    continue
                delivery = leg_count
            elif destination.startswith("F"):
                if destination not in route.service_order:
                    continue
                delivery = route.service_stop_index(destination)
            else:
                continue

            if pickup >= delivery:
                continue
            serviceable.append(group_index)
            intervals[group_index] = (pickup, delivery)

        if not serviceable:
            return None

        used_service_facilities = {
            location
            for group_index in serviceable
            for location in group_keys[group_index]
            if location.startswith("F")
        }
        if used_service_facilities != set(route.service_order):
            return None

        leg_groups: list[tuple[int, ...]] = []
        for leg in range(leg_count):
            groups = tuple(
                group_index
                for group_index in serviceable
                if intervals[group_index][0] <= leg < intervals[group_index][1]
            )
            leg_groups.append(groups)
        return _RouteTemplate(
            route=route,
            serviceable_groups=tuple(serviceable),
            leg_groups=tuple(leg_groups),
        )

    def _apply_candidate(
        self,
        problem: ProblemData,
        solution: Solution,
        candidate: _PairCandidate,
    ) -> None:
        groups = list(zip(candidate.passenger_groups, candidate.optimized_routes, strict=True))
        groups.sort(
            key=lambda item: (
                item[1].aircraft_minutes,
                item[1].route.aircraft_type,
                item[1].route.base_airport,
                item[1].route.service_order,
                item[0],
            )
        )
        for index, uid in enumerate(candidate.source_uids):
            if index < len(groups):
                ids, optimized = groups[index]
                self.route_optimizer.install(problem, solution, uid, ids, optimized)
            elif uid in solution.flights:
                del solution.flights[uid]

    @staticmethod
    def _candidate_key(candidate: _PairCandidate) -> tuple:
        return (
            -candidate.savings,
            candidate.passenger_delta,
            candidate.fuel_delta,
            len(candidate.optimized_routes),
            candidate.source_uids,
            tuple(route.route.service_order for route in candidate.optimized_routes),
        )

    @staticmethod
    def _person_ids(solution: Solution, flight_uid: str) -> list[str]:
        return sorted(
            pid
            for pid, assignment in solution.assignments.items()
            if assignment.flight_uid == flight_uid
        )

    @staticmethod
    def _flight_cost(problem: ProblemData, solution: Solution, flight_uid: str) -> tuple[int, int, float]:
        assignments = {
            pid: assignment
            for pid, assignment in solution.assignments.items()
            if assignment.flight_uid == flight_uid
        }
        metrics = evaluate_solution(
            problem,
            Solution(
                flights={flight_uid: solution.flights[flight_uid]},
                assignments=assignments,
            ),
        )
        return (
            metrics.total_aircraft_usage_minutes,
            metrics.total_passenger_travel_minutes,
            metrics.total_fuel_consumption_kg,
        )

    @staticmethod
    def _status_name(status: int) -> str:
        if status == cp_model.OPTIMAL:
            return "OPTIMAL"
        if status == cp_model.FEASIBLE:
            return "FEASIBLE"
        if status == cp_model.INFEASIBLE:
            return "INFEASIBLE"
        if status == cp_model.MODEL_INVALID:
            return "MODEL_INVALID"
        return "UNKNOWN"
