# First solve (Python)

A minimal long-only mean-variance rebalance. Arrays must be NumPy `float64`.

```python
import numpy as np
from ledge import PortfolioProblem

rng = np.random.default_rng(7)
n, k = 100, 8
F = rng.normal(0.0, 0.2, size=(n, k))       # factor loadings
omega = np.diag(rng.uniform(0.04, 0.12, size=k))  # factor covariance
d = rng.uniform(0.05, 0.10, size=n)         # idiosyncratic variance
mu = rng.normal(0.08, 0.02, size=n)         # expected returns

problem = PortfolioProblem(
    F,
    omega,
    d,
    mu,
    risk_aversion=8.0,
    budget=1.0,
    lower_bounds=np.zeros(n),
    upper_bounds=np.full(n, 0.03),
)
result = problem.solve()
print(result.status, result.weights.sum(), result.primal_residual)
```

The objective minimized is

```text
risk_aversion / 2 * (w - b)' Sigma (w - b)      [b = 0 without a benchmark]
- expected_returns' w
+ turnover_penalty / 2 * ||w - previous_weights||^2
+ l1_turnover_costs' |w - previous_weights|
```

subject to `sum(w) = budget`, `lower <= w <= upper`, and any linear
constraints you pass (`equality_matrix @ w = equality_rhs`,
`inequality_matrix @ w <= inequality_rhs`).

## The next date

Pass the previous solution as a warm start and price turnover:

```python
next_problem = PortfolioProblem(
    F,
    omega,
    d,
    mu + rng.normal(0.0, 0.005, size=n),
    risk_aversion=8.0,
    lower_bounds=np.zeros(n),
    upper_bounds=np.full(n, 0.03),
    previous_weights=result.weights,
    l1_turnover_costs=0.001,   # 10 bps per unit traded; scalar broadcasts
)
second = next_problem.solve(warm_start=result.weights)
print(second)
```

For repeated dates prefer [a rolling sequence](rolling.md), which caches the
factorization and chains full primal/dual warm starts automatically.

## Reading the result

`SolveResult` carries `status`, `weights`, `objective`, `iterations`,
`solve_time`, independently evaluated `primal_residual` / `dual_residual`,
`polished`, duals, and — for infeasible problems — an auditable
`certificate`. Failures raise by default; pass `raise_on_failure=False` to
inspect the result instead. `convergence_hints` explains unconverged solves
in portfolio vocabulary.

One-shot function form: `ledge.solve_mean_variance_factor(...)` takes the
same keyword arguments and returns the same `SolveResult`.
