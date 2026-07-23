# Tuning solver settings

Defaults are chosen so that the declared envelope converges without tuning
— **try defaults first**. Every knob below exists in Rust
(`SolverSettings`) and as a Python keyword argument on `solve()`,
`sequence()`, and `solve_batch()`.

| Setting (Python kwarg) | Default | Meaning |
|---|---|---|
| `max_iterations` | `10_000` | ADMM iteration cap |
| `absolute_tolerance` | `1e-6` | absolute stopping tolerance |
| `relative_tolerance` | `1e-5` | relative stopping tolerance |
| `rho` | `1.0` | initial augmented-Lagrangian penalty |
| `adaptive_rho` | `True` | residual-balancing penalty adaptation |
| `over_relaxation` | `1.6` | ADMM over-relaxation α ∈ (0, 2); `1.0` = plain ADMM |
| `scaling_iterations` | `10` | Ruiz equilibration passes; `0` disables |
| `infeasibility_tolerance` | `1e-5` | certificate detection threshold; `0` disables |
| `polish` | `True` | audit-gated active-set refinement after `Solved` |

Rust additionally exposes `sigma`, `check_termination_every`,
`adaptive_rho_interval` / `_tolerance` / `_multiplier`, `minimum_rho` /
`maximum_rho`, and `polish_regularization` /
`polish_refinement_iterations`; see the `SolverSettings` rustdoc.

## When something is slow or unconverged

Work through these in order:

1. **Read `convergence_hints`.** Unconverged results name the dominating
   residual and suggest the next step.
2. **Check your formulation, not the solver.** Explicit constraint rows
   grow the reduced factorization (`r = factors + rows`); prefer box
   constraints (`max_weight`, `max_short` templates add no rows) and keep
   `m` in the low hundreds.
3. **Warm-start rolling workloads** through a sequence instead of solving
   cold each date — measured ~2x fewer iterations and zero steady-state
   refactorizations.
4. **Badly scaled data**: leave `scaling_iterations` on (it is the
   difference between `Solved` and `MaxIterations` on ill-conditioned
   suites). If you disabled it for experiments, re-enable it.
5. **Looser tolerance**: for backtests, `absolute_tolerance=1e-5` often
   halves iterations; polishing usually recovers accuracy afterwards
   (verify `polished` and the audited residuals).
6. **More iterations**: large `n` with tight boxes can legitimately need
   more than 10k iterations cold; warm-started dates will not.

## What not to tune

- `over_relaxation`: 1.6 measured 1.7–2.9x iteration cuts across the smoke
  matrix; per-instance tuning overfits.
- `rho` / adaptation parameters: the residual-balancing policy replays the
  same penalty ladder every solve, which is what makes the factorization
  cache effective. Changing the ladder trades cache hits for guesses.
- `polish=False` buys single-digit percent time and costs six orders of
  magnitude of residual accuracy.
