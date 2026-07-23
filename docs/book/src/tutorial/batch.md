# Many accounts: batch

`solve_batch` runs one rolling sequence per account, in parallel over the
account axis. Accounts share no state, so results are **bit-identical to a
serial loop regardless of thread count** — threading changes wall-clock,
never answers.

## Python

```python
import ledge

results = ledge.solve_batch(
    problems,                     # list[PortfolioProblem], one per account
    steps,                        # list[list[dict]]: per-account, per-date
    chain_previous_weights=True,  # backtest convention, see below
)
for account_result in results:    # input order preserved
    for solution in account_result:
        ...
```

Each step dict mirrors `solve_next` keyword arguments
(`{"expected_returns": ..., "budget": ...}`). The GIL is released for the
whole batch.

- `chain_previous_weights=True` implements the backtest convention: after a
  `Solved` date the turnover anchor moves to that date's solved weights;
  non-`Solved` dates leave the anchor unchanged (the account did not trade).
  An explicit `previous_weights` in a step wins. Requires a turnover term.
- Failures stay per account: one account's bad feed never discards the other
  accounts' finished results. Errors name the account (and step) index.

## Rust

```rust,ignore
use ledge::{solve_batch, BatchAccount};

let accounts: Vec<BatchAccount> = ...; // problem + ordered RebalanceSteps
let results = solve_batch(&accounts, &settings);
// Vec<Result<Vec<Solution>, PortfolioError>>, input order
```

Threading is behind the **non-default `rayon` cargo feature** (the Python
wheel enables it). Without the feature the same API runs serially with
identical results. `RAYON_NUM_THREADS` or a caller-installed pool controls
the width.

## Published throughput

1 model × 500 accounts × 250 dates (n=200, k=15, L2+L1 turnover, chained
anchors): **12.9 s wall on 4 vCPUs — 9.7k account-date solves per second,
4.0x over the serial build**, all 125k solves `Solved`. Raw samples and
methodology: `benchmarks/results/2026-07-batch/` in the repository.
