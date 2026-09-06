# Q1 V2 — Tail Elimination / Passenger Insertion

This stage starts from **Pair-Merge V1** and targets the remaining under-filled flights.

## Core idea

A single-destination donor flight may be removed if all of its passengers can be redistributed to spare seats on other flights from the **same base airport**, while the resulting extra route time is smaller than the donor flight time that is removed.

For donor flight `d`, if recipient-route modifications add total aircraft time `Delta`, the saving is

`S = T_d - Delta`.

The move is accepted only when `S > 0`.

## V2 restrictions

- The starting solution is Pair-Merge V1.
- Only a **single-destination** flight may be a donor.
- Donor passengers may be split across several recipients.
- Recipients must use the same base airport as the donor, so fixed-origin constraints and frozen LAND airport assignments remain valid.
- Only already-empty seats of the recipient's **current aircraft type** may be used; V2 does not change recipient aircraft type.
- A recipient may serve at most **three passenger destinations** after insertion.
- For a recipient whose route must change, every service-order permutation is evaluated by `Q1RouteBuilder`, which may add technical refuel stops and enforces the five-sea-landing cap.
- The independent Reference Validator remains authoritative for final legality and metric recomputation.

## Optimization inside one donor elimination

For each feasible recipient, insertion options `q = 1..spare_seats` are generated. A small dynamic program chooses how many donor passengers to place on each recipient so that exactly all donor passengers are reassigned.

Lexicographic insertion cost:

1. added aircraft usage time;
2. added passenger travel time;
3. added fuel consumption;
4. number of modified recipient flights.

Across donors, the algorithm greedily applies the positive move with the largest aircraft-time saving, then recomputes the current solution and repeats until no positive move remains.

This keeps V2 interpretable: it measures the value of **using fragmented spare seats to eliminate tail flights**, without yet introducing aircraft-type swaps, cross-airport LAND reassignment, or general ALNS moves.
