# Migrating from cvxpy

A mapping from the cvxpy patterns used in factor-model portfolio
rebalancing to the Ledge Python API. Every code pair in this guide is
executed and cross-checked against cvxpy + Clarabel by
[`python/tests/test_migration_guide.py`](https://github.com/Jiangki/ledge/blob/main/python/tests/test_migration_guide.py),
so the mappings stay correct as APIs evolve.

Ledge is **not** a cvxpy replacement. cvxpy is a modeling language over many
solvers; Ledge is one specialized solver for one problem family. Migrate the
rebalancing QP; keep cvxpy for everything else.

## 1. Should you migrate?

Migrate a workload when all of the following hold:

- The objective is mean-variance (optionally with tracking error, L2
  turnover, and/or proportional transaction costs) and the constraints are
  linear: budget, weight boxes, exposure equalities, upper inequalities.
- The covariance already has factor structure `Sigma = F @ Omega @ F.T +
  np.diag(d)` — Ledge takes `F`, `Omega`, `d` directly and never forms the
  `n x n` matrix.
- Instances live inside the declared envelope: roughly `n <= 5000` assets,
  `k <= 100` factors, `m <= 200` explicit linear constraint rows.
- The workload is repeated: rolling backtests and daily rebalances are where
  warm starts and factorization reuse pay (see the measured numbers in
  [`docs/examples/README.md`](https://github.com/Jiangki/ledge/blob/main/docs/examples/README.md)).

Stay in cvxpy when you need anything outside that set, for example:

- integer or cardinality constraints (`w != 0` counting, buy-in thresholds);
- second-order-cone or exponential-cone terms (robust epsilon-balls, CVaR
  with auxiliary formulations beyond a QP);
- objectives that are not a convex QP plus absolute-value terms;
- a *hard* turnover budget `cp.norm1(w - w_prev) <= tau`. Ledge prices
  turnover in the objective (`l1_turnover_costs`) instead of capping it; if
  your process needs the cap semantics, keep cvxpy for those dates or tune
  the cost until realized turnover sits where the cap did.

## 2. The core model

Ledge minimizes, over weights `w` with `sum(w) = budget` and
`lower <= w <= upper` plus optional linear constraints:

```text
risk_aversion / 2 * (w - b)' Sigma (w - b)   [b = 0 without a benchmark]
- expected_returns' w
+ turnover_penalty / 2 * ||w - previous_weights||^2
+ l1_turnover_costs' |w - previous_weights|
```

Two conventions to check before comparing numbers:

- **Factor of 1/2 on the risk term.** If your cvxpy model writes
  `gamma * cp.quad_form(w, Sigma)` without the half, pass
  `risk_aversion = 2 * gamma`.
- **Maximization.** `cp.Maximize(mu @ w - gamma/2 * cp.quad_form(...))` is
  the same problem; Ledge's minimized objective is its negation. Compare
  weights, or evaluate one objective function of your own on both weight
  vectors (that is how Ledge's own gold tests avoid convention traps).

The basic long-only rebalance, side by side:

```python
# cvxpy
w = cp.Variable(n)
Sigma = F @ Omega @ F.T + np.diag(d)          # dense n x n materialized
risk = 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(Sigma))
prob = cp.Problem(
    cp.Minimize(risk - mu @ w),
    [cp.sum(w) == 1.0, w >= lower, w <= upper],
)
prob.solve(solver=cp.CLARABEL)
weights = w.value
```

```python
# ledge
from ledge import solve_mean_variance_factor

result = solve_mean_variance_factor(
    F, Omega, d, mu,
    risk_aversion=gamma,
    budget=1.0,
    lower_bounds=lower,
    upper_bounds=upper,
)
weights = result.weights
```

Arrays are NumPy `float64`; `Omega` is passed as a dense PSD matrix (use
`np.diag(...)` for diagonal factor covariances). For repeated solves of one
structure, build a `PortfolioProblem` once instead of calling the one-shot
function (§6).

## 3. Constraint mapping

| cvxpy | Ledge |
|---|---|
| `cp.sum(w) == 1.0` | `budget=1.0` |
| `w >= lower`, `w <= upper` | `lower_bounds=lower, upper_bounds=upper` (pass together) |
| `A @ w == b` (exposures, neutrality) | `equality_matrix=A, equality_rhs=b` |
| `A @ w <= b` (caps) | `inequality_matrix=A, inequality_rhs=b` |
| `A @ w >= b` (floors) | negate: rows `-A`, right-hand side `-b` |
| `l <= A @ w <= u` | two stacked blocks: `A` with rhs `u`, `-A` with rhs `-l` |
| unconstrained budget | omit — but note Ledge always has a budget row; set it to the sum you want |

Floors and ranges, concretely:

```python
# cvxpy: sector floors and caps
constraints += [S @ w >= floor, S @ w <= cap]

# ledge: one upper-inequality block
inequality_matrix = np.vstack([S, -S])
inequality_rhs = np.concatenate([cap, -floor])
```

There is no dedicated "sector constraint" type on either side: an
industry-neutral book, style bounds, or a concentration limit are all rows
of `A`. For the common templates Ledge builds those rows for you
(roadmap 3.1):

```python
# cvxpy: industry weights pinned to the benchmark's, per-name cap
constraints += [S @ w == S @ b, w <= 0.06]

# ledge: template kwargs compile onto the same rows / boxes
result = solve_mean_variance_factor(
    F, Omega, d, mu,
    benchmark_weights=b,          # tracking objective ...
    industry_ids=industry_ids,    # ... and industry neutrality against b
    max_weight=0.06,              # box tightening, no extra rows
)
```

`industry_ids` is one integer per asset; add `industry_targets=` to pin
group weights explicitly (no benchmark needed). Style bands map through
`style_matrix=` / `style_lower=` / `style_upper=` (use `±np.inf` for
one-sided bands), and `max_short=` caps per-name shorts (`0.0` forces
long-only). Templates append to whatever `equality_matrix` /
`inequality_matrix` you passed, and their targets roll through
`solve_next(equality_rhs=..., inequality_rhs=...)` in sequences.

## 4. Turnover and transaction costs

```python
# cvxpy
objective += 0.5 * eta * cp.sum_squares(w - w_prev)   # smooth L2 preference
objective += kappa @ cp.abs(w - w_prev)               # proportional costs
```

```python
# ledge
result = solve_mean_variance_factor(
    F, Omega, d, mu,
    ...,
    previous_weights=w_prev,
    turnover_penalty=eta,        # L2; 0 to disable
    l1_turnover_costs=kappa,     # scalar broadcasts; per-asset array allowed
)
```

Both terms share the `previous_weights` anchor and can be combined. The L1
term is handled by a dedicated proximal block, not an epigraph
reformulation, so the no-trade region is machine-exact: assets whose
marginal utility change is below their cost stay *exactly* at
`w_prev[i]`, which is what makes realized turnover reports clean. The L1
multipliers are audited in the KKT check like every other dual.

## 5. Tracking error

```python
# cvxpy
active = w - benchmark
objective = 0.5 * gamma * cp.quad_form(active, cp.psd_wrap(Sigma)) - mu @ w
```

```python
# ledge
result = solve_mean_variance_factor(
    F, Omega, d, mu,
    ...,
    benchmark_weights=benchmark,
)
```

One convention: Ledge drops the constant `gamma/2 * b' Sigma b` from the
reported objective (standard QP-solver behavior). Weights are unaffected;
add the constant back if you compare objective values directly.

## 6. Rolling backtests: `cp.Parameter` → `PortfolioSequence`

The cvxpy pattern for a rolling backtest keeps the problem and swaps
parameter values so the solver can reuse its symbolic form:

```python
# cvxpy
mu_parameter = cp.Parameter(n)
prob = cp.Problem(cp.Minimize(risk - mu_parameter @ w), constraints)
for date in dates:
    mu_parameter.value = mu_by_date[date]
    prob.solve(solver=cp.CLARABEL, warm_start=True)
```

Ledge's equivalent is a first-class object; it also chains full
primal/dual warm starts and keeps the equilibration and reduced
factorizations cached across dates:

```python
# ledge
from ledge import PortfolioProblem

problem = PortfolioProblem(F, Omega, d, mu0, ..., previous_weights=w0,
                           l1_turnover_costs=0.001)
sequence = problem.sequence()
for date in dates:
    result = sequence.solve_next(
        expected_returns=mu_by_date[date],
        previous_weights=held_weights,      # rolls the turnover anchor
    )
    held_weights = result.weights
```

`solve_next` accepts only factorization-preserving updates: expected
returns, the turnover anchor, the tracking benchmark, the budget, and
equality / inequality right-hand sides. Changing the covariance, the
constraint matrices, bounds, or the penalty levels means the problem
structure changed — build a new `PortfolioProblem` and a new sequence
(cvxpy makes the same distinction: those are new `cp.Problem`s, not
parameter updates). A rejected step leaves the sequence unchanged, so a bad
date can be skipped without restarting the backtest.

Measured effect on the repository's momentum-backtest example (300 assets,
12 factors, 24 dates): warm dates take 37 versus 79 iterations and 0.91 ms
versus 1.71 ms against per-date cold solves, with 24 factorizations
replaced by 2
([`docs/examples/README.md`](https://github.com/Jiangki/ledge/blob/main/docs/examples/README.md)).

## 7. Statuses, failures, and duals

| cvxpy | Ledge |
|---|---|
| `prob.status == cp.OPTIMAL` | `result.status == "solved"` (default: anything else raises `RuntimeError`) |
| `cp.OPTIMAL_INACCURATE` | closest is `"max iterations"` with `convergence_hints` |
| `cp.INFEASIBLE` | `"primal infeasible"` + `result.certificate` (Farkas proof, checkable) |
| `cp.UNBOUNDED` | `"dual infeasible"` + `result.certificate.direction` (descent ray) |
| `prob.value` | `result.objective` (minimization convention, constants dropped per §5) |
| `constraint.dual_value` | not exposed in Python yet; the Rust API returns all multipliers, and `result` reports independently audited KKT residuals |

Differences worth knowing:

- Some contradictions never reach the solver: impossible inputs that are
  visible from the data alone (a budget outside what the boxes can sum to,
  mismatched dimensions) raise `ValueError` at build time, where cvxpy
  would return `INFEASIBLE` after a solve.
- Ledge **raises by default** on anything other than `"solved"`. Pass
  `raise_on_failure=False` to inspect the returned iterate,
  `convergence_hints`, and `certificate` instead — that is the mode to use
  inside backtest loops that skip bad dates.
- Infeasibility errors name the conflicting constraints in portfolio
  vocabulary (budget vs caps vs bounds), and the certificate is a
  machine-checkable proof, not just a status.
- `result.primal_residual` / `dual_residual` are always evaluated on your
  original data, never on internally scaled data, and
  `result.polished` reports whether the direct active-set refinement
  (residuals ~1e-11 instead of ~1e-5) was adopted.

## 8. Verify your migration

Do not trust either solver's reported objective when switching: evaluate
one NumPy objective of your own on both weight vectors, exactly like
[`python/tests/test_reference.py`](https://github.com/Jiangki/ledge/blob/main/python/tests/test_reference.py)
and
[`python/tests/test_migration_guide.py`](https://github.com/Jiangki/ledge/blob/main/python/tests/test_migration_guide.py)
do:

```python
def my_objective(weights):
    risk = 0.5 * gamma * weights @ (F @ (Omega @ (F.T @ weights))) \
         + 0.5 * gamma * weights @ (d * weights)
    return risk - mu @ weights   # + your turnover / tracking terms

assert my_objective(ledge_weights) <= my_objective(cvxpy_weights) + 1e-6
np.testing.assert_allclose(ledge_weights, cvxpy_weights, atol=1e-4)
```

Keep the cvxpy model in your test suite as an oracle for a few dates — that
is cheap insurance and exactly how this repository guards its own solver.
