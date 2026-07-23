# API surface

## Rust

Published API documentation:
[`ledge`](https://docs.rs/ledge-portfolio) and
[`ledge-core`](https://docs.rs/ledge-core).
To generate the Rust documentation locally:

```bash
cargo doc -p ledge-portfolio --no-deps --open
```

The `ledge` crate re-exports everything from `ledge-core`. Main types:

| Type / function | Role |
|---|---|
| `PortfolioProblem` | portfolio-vocabulary builder over the QP |
| `QpProblem`, `FactorQuad`, `L1Term` | the underlying factor-structured QP |
| `Solver`, `SolverSettings` | settings container and entry point |
| `Solution`, `SolveStatus`, `DualVariables` | results with audited residuals and duals |
| `Solver::workspace` → `Workspace` | equilibration + factorization cache across solves |
| `PortfolioSequence`, `RebalanceStep`, `solve_sequence` | rolling date-by-date API |
| `solve_batch`, `BatchAccount` (feature `rayon`) | parallel multi-account batch |
| `check_kkt`, `check_primal_certificate`, `check_dual_certificate` | independent audits |
| `generate_synthetic`, `SyntheticConfig` | deterministic test instances |

## Python

The package installs as `ledge-portfolio`, imports as `ledge`, and carries
docstrings on every public symbol:

```python
import ledge
help(ledge.PortfolioProblem)
help(ledge.solve_batch)
```

| Symbol | Role |
|---|---|
| `ledge.PortfolioProblem` | problem construction (NumPy arrays, keyword constraints/templates) |
| `.solve(**settings)` | one solve → `SolveResult` |
| `.sequence(**settings)` → `PortfolioSequence.solve_next(...)` | rolling API |
| `ledge.solve_batch(problems, steps, ...)` | parallel multi-account batch |
| `ledge.solve_mean_variance_factor(...)` | one-shot function form |
| `PortfolioProblem.to_json()` / `from_json()`, `SolveResult.to_json()` | reproduction dumps |
| `SolveResult` | weights, status, objective, audited residuals, diagnostics, certificate; `to_json()` includes the full dual blocks |

Python does not currently expose solution dual multipliers as direct
`SolveResult` attributes. Use `to_json()` when a bug report or audit needs
the complete serialized solver result.
