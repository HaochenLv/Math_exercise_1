from __future__ import annotations

from helicopter_planner.domain import ProblemData, Solution
from helicopter_planner.solver.q2.pair_recombination import (
    Q2ExactPairRecombinationSolver,
    _FlightSummary,
)


class Q2ResidualMainPairRecombinationSolver(Q2ExactPairRecombinationSolver):
    """Q2 B3B: exact residual x main two-sortie recombination.

    B3A only lets residual ``q2b0b-*`` flights interact with other residual
    flights. B3B exposes the much larger structural neighborhood in which one
    residual flight is pooled with one non-residual/main flight. The inherited
    exact pair model then repartitions every pooled OD count into one or two
    rebuilt sorties while re-optimizing route order, base, aircraft type and
    technical refuelling.

    Candidate generation keeps the neighborhood computationally bounded:
    for every residual flight we include its nearest ``neighbor_count`` main
    flights and every main flight sharing an offshore service facility when the
    pooled facility union is within the configured bound.
    """

    def _summaries(self, problem: ProblemData, solution: Solution) -> dict[str, _FlightSummary]:
        summaries: dict[str, _FlightSummary] = {}
        for uid in sorted(solution.flights):
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
        residual_uids = sorted(uid for uid in summaries if uid.startswith("q2b0b-"))
        main_uids = sorted(uid for uid in summaries if not uid.startswith("q2b0b-"))
        pairs: set[tuple[str, str]] = set()

        for residual_uid in residual_uids:
            residual = summaries[residual_uid]
            ranked: list[tuple[int, int, float, str]] = []
            for main_uid in main_uids:
                main = summaries[main_uid]
                union_size = len(residual.sea_facilities | main.sea_facilities)
                if union_size > self.max_union_facilities:
                    continue
                shared = bool(residual.sea_facilities & main.sea_facilities)
                distance = min(
                    problem.distance(a, b)
                    for a in residual.sea_facilities
                    for b in main.sea_facilities
                )
                ranked.append((0 if shared else 1, union_size, distance, main_uid))

            for _, _, _, main_uid in sorted(ranked)[: self.neighbor_count]:
                pairs.add(tuple(sorted((residual_uid, main_uid))))

            for main_uid in main_uids:
                main = summaries[main_uid]
                if not (residual.sea_facilities & main.sea_facilities):
                    continue
                if len(residual.sea_facilities | main.sea_facilities) > self.max_union_facilities:
                    continue
                pairs.add(tuple(sorted((residual_uid, main_uid))))

        return tuple(sorted(pairs))
