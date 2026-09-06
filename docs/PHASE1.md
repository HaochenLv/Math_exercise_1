# Phase 1 implementation contract

The optimization algorithm is intentionally absent in Phase 1. The stable plugin boundary is `helicopter_planner.solver.Solver`.

Data flow:

`ProblemData -> Solver (Phase 2) -> Solution -> Fast Checker -> Evaluator -> Exporter -> independent Reference Validator`

The final validator intentionally duplicates critical feasibility and metric logic rather than importing solver-side checking/evaluation code.
