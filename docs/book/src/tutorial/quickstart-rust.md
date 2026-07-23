# First solve (Rust)

```rust,no_run
use ledge::{FactorCovariance, Matrix, PortfolioProblem, SolveStatus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3 assets, 1 factor.
    let factors = Matrix::new(3, 1, vec![1.0, -0.5, 0.25])?;
    let problem = PortfolioProblem::new(
        factors,
        FactorCovariance::Diagonal(vec![0.1]),
        vec![0.2, 0.3, 0.25],   // idiosyncratic variances d
        vec![0.08, 0.04, 0.06], // expected returns mu
    )?
    .with_risk_aversion(5.0)?
    .with_bounds(vec![0.0; 3], vec![0.6; 3])?;

    let solution = problem.solve(None)?;
    assert_eq!(solution.status, SolveStatus::Solved);
    println!("{:?}", solution.x);
    Ok(())
}
```

`PortfolioProblem` is a builder; each `with_*` method validates eagerly so
mistakes fail at build time, not as mysterious infeasible solves:

| Builder | Adds |
|---|---|
| `with_risk_aversion`, `with_budget` | objective scale and budget row |
| `with_bounds` | per-asset boxes |
| `with_equalities`, `with_inequalities` | explicit linear constraints (replace semantics — call before templates) |
| `with_turnover_penalty` | smooth L2 turnover around previous weights |
| `with_l1_turnover` | exact proportional transaction costs (prox block) |
| `with_tracking_benchmark` | tracking-error risk term |
| `with_industry_neutrality`, `with_group_targets`, `with_style_bounds`, `with_concentration_limit`, `with_short_limit` | [constraint templates](../guide/constraints.md) |

`problem.solve(None)` uses default settings; pass
`Some(&solver)` with a configured `Solver` for
[custom settings](../reference/tuning.md), and
`solution.warm_start()` to seed the next solve with the full primal/dual
state.

The lower-level `QpProblem` API remains available for custom continuous
convex QPs with the same factor-structured quadratic.
