# Ledge Python binding

Package name on install: **`ledge-portfolio`** (import name: `ledge`).

It exposes Ledge's factor mean-variance solver to NumPy (`float64` only) and
releases the GIL while solving. This is an **alpha** binding; the release
workflow builds abi3 wheels for the supported platforms.
Each distribution includes the Apache-2.0 project license and the generated
Rust dependency notices in `THIRD_PARTY_LICENSES.html`.

Install the published `0.2.0` release:

```bash
python -m pip install ledge-portfolio==0.2.0
```

Release `0.2.0` provides abi3 wheels for Linux x86-64/aarch64, macOS
universal2, and Windows x86-64. Other platforms build from the source
distribution and require a Rust toolchain.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install -e python/
python python/examples/rebalance.py
python -m pytest python/tests
```

For binding development:

```bash
cd python
maturin develop
python examples/rebalance.py
```

Primary API:

```python
from ledge import PortfolioProblem, solve_mean_variance_factor
```

`PortfolioProblem.solve(warm_start=previous_weights)` re-solves a single
problem. For rolling multi-date rebalances, prefer
`PortfolioProblem.sequence()`: the returned `PortfolioSequence` caches the
equilibration and reduced factorizations across dates and chains full
primal/dual warm starts automatically — each date is one
`sequence.solve_next(expected_returns=..., previous_weights=...,
benchmark_weights=..., budget=..., equality_rhs=..., inequality_rhs=...)`
call (all arguments optional; only factorization-preserving updates are
accepted). See [`examples/rolling.py`](examples/rolling.py) and the full
backtest in
[`../docs/examples/rolling_backtest.py`](../docs/examples/rolling_backtest.py).
For many accounts sharing one model, `ledge.solve_batch(problems, steps,
chain_previous_weights=..., **solver_kwargs)` runs one sequence per account
in parallel over the account axis (the GIL is released for the whole
batch); `steps` is one list of per-date dicts per account whose keys mirror
the `solve_next` keyword arguments, and `chain_previous_weights=True`
anchors each date's turnover at the previous solved date's weights, the
usual backtest convention.
`SolveResult` reports weights, status, objective, KKT residuals,
iterations, solve time, and adaptive-penalty diagnostics. Infeasible
problems stop early with status `'primal infeasible'` (or
`'dual infeasible'` for unbounded objectives): by default a `RuntimeError`
names the conflicting portfolio constraints; pass `raise_on_failure=False`
to inspect `SolveResult.certificate`, an independently checkable Farkas
combination (or descent direction).

Turnover control around `previous_weights`: `turnover_penalty` is a smooth
**L2** penalty; `l1_turnover_costs` (a scalar broadcast to all assets, or a
per-asset array) is **exact proportional transaction cost** with a genuine
no-trade region, handled by a dedicated proximal block. Both may be
combined. `benchmark_weights` switches the risk term to active risk
`(w - b)' Sigma (w - b)` against a tracking benchmark.

Migrating an existing cvxpy rebalance? See
[`../docs/cvxpy_migration.md`](../docs/cvxpy_migration.md) — every mapping
in it is executed against cvxpy + Clarabel by
[`tests/test_migration_guide.py`](tests/test_migration_guide.py).

See the root [README](../README.md) for scope, limitations, and smoke timings.