# Q2 B0A — Fixed-Q1-Route Zero-Cost Packing

## Purpose

B0A is deliberately a **diagnostic warm start**, not a complete Q2 solution.

We regenerate the mature Q1 V10 solution and freeze its:

- flight set;
- base airport of every flight;
- aircraft type;
- offshore stop order;
- refuel decisions;
- Q1 outbound passenger assignments.

The 2,400 Q2 requests not already served by Q1 are then tested against these
fixed routes. A return or offshore-shuttle passenger may be inserted only if
their required origin and destination already occur in the correct order on an
existing route. `LAND` resolves to that flight's base airport.

Therefore every B0A insertion has exactly zero incremental:

- aircraft usage time;
- flight count;
- route distance;
- fuel consumption;
- available seat-km.

## Dynamic onboard constraint

For a passenger assigned to pickup index `p` and delivery index `q`, the
passenger occupies every leg

\[
p,p+1,\ldots,q-1.
\]

For every flight leg `l`,

\[
L_l^{Q1}+\sum_{p,o:\;l\in o}x_{p,o}\le C,
\]

where `x[p,o]` selects one feasible route/index option for a Q2 passenger.

At a stop the diagnostic trace checks

\[
L^{after}=L^{before}-D+P,
\]

which explicitly implements the Q2 rule **drop off first, then pick up**.

## Optimization

A global CP-SAT packing model is solved lexicographically:

1. maximize the number of additional Q2 passengers packed into the frozen Q1
   skeleton;
2. with that maximum fixed, minimize their added passenger travel time.

No route change is allowed in B0A.

## Validation

The result is checked twice:

1. solver-side `check_solution(..., require_all_requests=False)`;
2. independent `reference_validator.validate_submission(...,
   require_all_requests=False)`.

The workflow also asserts that all fixed-route aircraft metrics are unchanged.

## Outputs

The GitHub Actions artifact `q2-b0a-warm-start` contains:

- `q2-routes.partial.csv`;
- `q2-assignments.partial.csv`;
- `metrics.json`;
- `packing_summary.json`;
- `validation.json`;
- `packed_extra.csv`;
- `remaining_requests.csv`;
- `onboard_trace.csv`.

The `.partial.csv` suffix is intentional: B0A does **not** claim to serve all
4,000 Q2 requests. The remaining-request structure is the input to the next Q2
construction stage.
