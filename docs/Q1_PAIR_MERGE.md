# Q1 Improvement V1 — Greedy Pairwise Savings Merge

## Purpose

This stage improves the verified Q1 single-facility baseline without changing the project architecture. The baseline remains the reference point. V1 adds only one new degree of freedom: two original baseline flights from the same airport may be replaced by one flight serving two distinct passenger destinations.

## Frozen decisions

- The baseline LAND-to-airport assignment is kept unchanged.
- Only two ORIGINAL baseline flights can be merged.
- A merged flight serves at most two passenger destinations.
- Merged flights are not merged again in V1.
- No passenger transfer is introduced.

These restrictions make the gain directly attributable to cross-facility joint transport rather than to a large simultaneous redesign.

## Candidate generation

For every pair of baseline flights with the same base airport and different passenger destinations:

1. collect both passenger sets;
2. enumerate T1/T2/T3 subject to total passenger capacity;
3. test both service orders `Fi -> Fj` and `Fj -> Fi`;
4. call `Q1RouteBuilder` to find the minimum-time fuel-feasible route for that fixed airport/type/order;
5. allow the route builder to insert only official technical refuel stops, with at most five sea landings total;
6. compute aircraft usage time, passenger travel time, and fuel;
7. retain the best candidate for this pair.

A pair is merge-eligible only when the new route STRICTLY reduces total aircraft usage time compared with the two original flights.

## Savings score

Primary savings:

`S_air = T_old_left + T_old_right - T_new`

Candidates are selected greedily by descending `S_air`. Equal primary savings prefer larger passenger-time savings, then larger fuel savings. Once a baseline flight is used in a selected pair, it cannot be used again.

This is deliberately a greedy savings heuristic, not yet a globally optimal matching algorithm.

## Q1RouteBuilder contract

Input:

- `ProblemData`
- base airport
- ordered tuple of service facilities
- aircraft type

Output:

- minimum-time feasible `BuiltRoute`, or `None`;
- ordered `SeaStop` list including technical refuel stops;
- service-stop indices for assignments;
- service arrival times;
- total route time and distance.

The route starts and ends at the same airport. Service facilities are never visited early as technical stops, preserving the official rule that a passenger must leave at the first visit to their destination after boarding.

## Acceptance pipeline

`Baseline -> Pair candidates -> Greedy disjoint selection -> Solution -> Fast Checker -> Evaluator -> official CSV -> independent Reference Validator`

A run is valid only when both checkers pass and the evaluator metrics exactly agree with the independent validator.

## V1 limitations / next candidates

V1 intentionally does NOT:

- re-optimize LAND airport allocation after merging;
- merge 3–5 passenger destinations;
- move a passenger between existing routes;
- swap passengers/routes;
- solve the disjoint-pair selection as an exact maximum-weight matching problem.

Those are possible next improvements after the V1 numerical result is inspected.
