# Q1 V5: Retype-Aware Generalized Tail Elimination

## Purpose

V4 is locally stable under whole-block relocate/swap and pairwise partial
passenger relocation, but those neighborhoods cannot remove a flight when its
passengers must be divided among several recipients whose **current** aircraft
types have insufficient spare seats.

V5 targets this gap.

## Starting point

The solver starts from the converged V4 solution.

Base-airport assignments remain fixed. Therefore:

- requests with explicit origins remain at their required airport;
- LAND passengers retain the airport selected by earlier stages;
- only flights with the same base airport may exchange passengers.

Cross-airport LAND reassignment is deliberately postponed to a later stage.

## Donor and recipients

Every single-destination flight is considered as a donor.

For a donor with `n` passengers, every other flight at the same base airport is
considered as a possible recipient. For each recipient and each feasible
inserted count `q = 1, ..., n`, the route is rebuilt from scratch.

The rebuild jointly chooses:

1. aircraft type T1/T2/T3;
2. service-facility order;
3. legal technical refueling stops;
4. the minimum-aircraft-time route under the five-sea-landing limit.

Thus a full T1 recipient may be upgraded to T2 or T3 to absorb donor
passengers when the added route time is smaller than the removed donor time.

## Dynamic program

For each donor, a small dynamic program chooses at most one insertion option
per recipient and requires the inserted counts to sum exactly to the donor
passenger count.

The lexicographic insertion cost is:

1. total recipient aircraft-time increase;
2. total passenger-time increase;
3. total fuel increase;
4. number of recipient flights;
5. number of recipient type changes;
6. deterministic recipient identifiers.

A donor is removable only when

\[
T_{\text{donor}} >
\sum_r \Delta T_r.
\]

Among all removable donors, V5 applies the one with the largest aircraft-time
saving, using passenger time and fuel as secondary tie-breakers.

## Polishing and termination

After every accepted donor elimination:

1. V3 whole-facility-block relocate/swap is run to convergence;
2. V4 partial passenger relocation is run to convergence;
3. V4 internally invokes V3 after its accepted moves.

V5 then searches again. It stops when no positive retype-aware donor
elimination remains.

## Validation

The official experiment must pass all of the following:

- the full test suite;
- the fast in-memory checker;
- export to official Q1 CSV format;
- the independent reference validator;
- exact metric agreement between evaluator and validator.

## Official Q1 result

The full 1,600-person instance was run in GitHub Actions at commit
`110ffa458fa81e87e6152e968ce2af0df6f011e6`.

| Metric | V4 start | V5 result | Change |
|---|---:|---:|---:|
| Total aircraft usage | 15,232 min | 14,963 min | -269 min (-1.766%) |
| Number of flights | 94 | 89 | -5 |
| Total passenger travel | 116,002 min | 121,120 min | +5,118 min |
| Total fuel consumption | 123,543.4 kg | 120,158.6 kg | -3,384.8 kg |
| Seat utilization | 48.2370% | 48.8883% | +0.6513 percentage points |

The solver accepted five generalized tail eliminations. Subsequent polishing
added five whole-block moves and two partial-relocation moves. The search
converged, all 12 tests passed, and both the fast checker and independent
reference validator accepted the exported solution.

Compared with the original direct-shuttle baseline, V5 reduces total aircraft
usage from 17,429 to 14,963 minutes and the flight count from 110 to 89.
