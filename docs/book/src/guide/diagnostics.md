# Trust: audits, certificates, polishing

Ledge's policy is that **you should never have to trust the solver's own
claim of success**. Every mechanism below evaluates on the original,
unscaled problem data and is independently checkable.

## Independent KKT audits

`check_kkt` recomputes primal and dual residuals from the returned point
and multipliers — including the L1 subgradient conditions — without using
any solver state. The reported `primal_residual` / `dual_residual` on every
solution come from this audit, never from internal scaled-space estimates.
The comparison harness applies the same audit to OSQP and Clarabel results,
so published cross-solver numbers use one referee.

## Automatic scaling (default on)

Ten Ruiz equilibration passes balance the problem before iterating,
preserving factor structure (the scaling acts on rows of `F` and on `d`;
no dense matrix is formed). Scaling changes only the space ADMM iterates
in: termination and all reported residuals are evaluated on original data.
`scaling_iterations=0` disables it.

## Solution polishing (default on)

After convergence, the active set is guessed from the final iterate and one
direct KKT solve refines the solution — typically from ~1e-5 residuals to
**1e-11 or better**, for single-digit-percent extra time. The polished
candidate is adopted **only if the audited worst KKT residual improves**;
degenerate active sets are rejected and the ADMM iterate returned
unchanged, so polish never degrades a solution and never ships uncertified
multipliers. `SolveResult.polished` / `Solution::polished` records the
outcome.

## Infeasibility certificates

Contradictory constraints do not burn 10 000 iterations. The solver detects
divergence directions and returns `PrimalInfeasible` (with a normalized
Farkas certificate) or `DualInfeasible` (with an unbounded descent ray).
Certificates are attached to the solution, auditable with
`check_primal_certificate` / `check_dual_certificate`, and translated into
portfolio vocabulary — e.g. *"budget row conflicts with the sector caps"* —
as the leading hint.

The default `infeasibility_tolerance=1e-5` is deliberately stricter than
OSQP's, because a false "your portfolio is infeasible" is worse than a slow
`MaxIterations`: problems infeasible by a smaller margin fall back to
`MaxIterations` with hints.

In Python, infeasible statuses raise by default with the hint as the
message; pass `raise_on_failure=False` to inspect
`SolveResult.certificate`.

## Unconverged solves

`MaxIterations` results carry `convergence_hints`: which residual
dominates, whether the penalty was re-tuned repeatedly, and what to try
(more iterations, scaling, looser tolerance). The final iterate is still
returned with honestly reported residuals.
