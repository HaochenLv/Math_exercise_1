from helicopter_planner.checking.checker import check_solution
from helicopter_planner.domain import (
    AIRCRAFT_TYPES,
    Assignment,
    FlightPlan,
    PersonRequest,
    ProblemData,
    SeaStop,
    Solution,
)
from helicopter_planner.solver.q1.tail_elimination import Q1TailEliminationSolver


class _StubStartingSolver:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    def solve(self, problem: ProblemData) -> Solution:
        return self.solution


def _toy_problem_and_solution() -> tuple[ProblemData, Solution]:
    locations = ("A01", "F001", "F002", "F003")
    raw = {
        ("A01", "F001"): 100.0,
        ("A01", "F002"): 105.0,
        ("A01", "F003"): 110.0,
        ("F001", "F002"): 20.0,
        ("F001", "F003"): 10.0,
        ("F002", "F003"): 10.0,
    }
    distances = {origin: {} for origin in locations}
    for origin in locations:
        for destination in locations:
            if origin == destination:
                distances[origin][destination] = 0.0
            elif (origin, destination) in raw:
                distances[origin][destination] = raw[(origin, destination)]
            elif (destination, origin) in raw:
                distances[origin][destination] = raw[(destination, origin)]
            else:
                raise AssertionError((origin, destination))

    requests: dict[str, PersonRequest] = {}
    solution = Solution()
    specs = {"T1": AIRCRAFT_TYPES["T1"]}

    def add_flight(uid: str, destination: str, count: int) -> None:
        solution.flights[uid] = FlightPlan(
            flight_uid=uid,
            base_airport="A01",
            aircraft_type="T1",
            sea_stops=[SeaStop(destination)],
        )
        for i in range(count):
            pid = f"{uid}_{i:02d}"
            requests[pid] = PersonRequest(pid, "A01", destination)
            solution.assignments[pid] = Assignment(pid, uid, 0, 1)

    add_flight("FLT000001", "F001", 9)  # spare 3
    add_flight("FLT000002", "F002", 9)  # spare 3
    add_flight("FLT000003", "F003", 6)  # donor, must split 3 + 3

    return ProblemData(distances=distances, requests=requests, aircraft_types=specs), solution


def test_tail_elimination_can_split_one_donor_across_two_recipients():
    problem, starting = _toy_problem_and_solution()
    result = Q1TailEliminationSolver(
        starting_solver=_StubStartingSolver(starting)
    ).solve_with_diagnostics(problem)

    assert len(result.solution.flights) == 2
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.donor_flight_uid == "FLT000003"
    assert decision.recipient_allocations == (
        ("FLT000001", 3),
        ("FLT000002", 3),
    )
    assert decision.aircraft_savings_minutes > 0

    check = check_solution(problem, result.solution, require_all_requests=True)
    assert check.ok, check.errors
