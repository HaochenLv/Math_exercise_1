# Q1 V6: LAND Cross-Airport Local Search

## Purpose

V1-V5 kept each LAND passenger at the airport selected by the earlier
single-facility construction. V6 opens that decision variable while preserving
all explicit A01/A02/A03 origins.

For every pair of flights based at different airports, V6 considers moving
`q=1,...,m` LAND passengers from one destination block to the other flight.
Both affected flights are then rebuilt independently at their own airports,
re-optimizing aircraft type, service order, and legal refueling.

Only strictly positive aircraft-time moves are accepted. After an accepted
cross-airport move, V3, V4, and V5 same-airport neighborhoods are polished to
convergence before the next cross-airport scan.

## Official Q1 result

The full 1,600-person instance was run in GitHub Actions at commit
`203d0bf92b45a9ceb0d813712a73aaaf9ff7f995`.

| Metric | V5 start | V6 result | Change |
|---|---:|---:|---:|
| Total aircraft usage | 14,963 min | 14,962 min | -1 min |
| Number of flights | 89 | 89 | 0 |
| Total passenger travel | 121,120 min | 121,056 min | -64 min |
| Total fuel consumption | 120,158.6 kg | 120,148.6 kg | -10.0 kg |
| Seat utilization | 48.8883% | 48.8712% | -0.0172 percentage points |

V6 accepted exactly one cross-airport LAND relocation. No V3, V4, or V5
polishing move became available afterward. The search converged, all 13 tests
passed, and both the fast checker and independent reference validator accepted
the exported solution.

## Interpretation

The extremely small primary improvement indicates that the earlier per-facility
LAND allocation, although frozen through V1-V5, already placed passengers close
to a strong pairwise local optimum across airports. The remaining meaningful
search gap is therefore not a one-way cross-airport passenger move, but a
larger coordinated recombination that can cross a temporarily non-improving
intermediate state.
