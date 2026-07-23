# Batch-over-accounts throughput — 2026-07

Published throughput for
[`roadmap 3.2`](../../../docs/ROADMAP.md#m3--04x-vertical-productization):
run a synthetic
**"1 model × 500 accounts × 250 trading days"** batch end-to-end and
publish the numbers. This is a self-timing measurement of Ledge's
`solve_batch` (no external solvers), so the cross-solver protocol rules do
not apply; machine, commit, command, and all per-account samples are
published for reproducibility.

## Workload

One shared factor model with per-account data: n=200 assets, k=15 factors,
budget 1, boxes
`[0, 10/n]`, an L2 turnover penalty (0.5) plus 10 bps proportional L1
costs, and a per-account expected-return tilt. Every account rolls through
250 dates of new expected returns with **backtest anchor chaining**
(`chain_previous_weights`): each date trades from the weights the previous
`Solved` date left behind. Default solver settings (polish on,
over-relaxation 1.6, scaling 10). 125,000 account-date solves in total.

## Results

| build | threads | solve wall time | throughput | speedup |
|---|---:|---:|---:|---:|
| `--features rayon` | 4 | 12.90 s | 9,693 account-dates/s | 3.96x |
| default (serial) | 1 | 51.08 s | 2,447 account-dates/s | 1.0x |

- All 125,000 solves reached `Solved`; total iterations 2,516,390
  (20.1 mean per solve, ~5,030 per account across its 250 dates) —
  **bit-identical between the parallel and serial runs** (per-account
  iteration counts and statuses match row for row across
  [`samples.csv`](samples.csv) and [`samples-serial.csv`](samples-serial.csv)),
  as the batch API guarantees: threading changes wall-clock, never answers.
- Speedup on 4 vCPUs is 3.96x — the account axis is embarrassingly
  parallel and the per-account sequences are compute-bound at this size.
- Summed iteration-only solver time is ~51 s in both runs (0.41 ms mean
  per solve): the parallel build does the same work on more cores.
- One-time account setup (building 500 problems and 125,000 steps,
  single-threaded) is 0.5 s and reported separately by the driver.

## Artifacts

| File | Content |
|---|---|
| [`samples.csv`](samples.csv) | Per-account rows from the 4-thread run: account, dates, solved count, total iterations, summed iteration-only solve time (ms) |
| [`samples-serial.csv`](samples-serial.csv) | Same rows from the serial control run |

## Environment

- Machine: cloud VM, Intel Xeon (4 vCPU), 15 GiB RAM, Linux 6.12.94+ x86_64
- Compiler: rustc 1.83.0; `--release` with thin LTO, `codegen-units = 1`
- Commit: `1ad4fa6` (branch `cursor/batch-rayon-c29a`)

## Reproduce

```bash
# 4-thread run (RAYON_NUM_THREADS to change the width)
cargo run -p ledge-portfolio --release --features rayon --example batch -- \
  --out samples.csv

# serial control
cargo run -p ledge-portfolio --release --example batch -- --out samples-serial.csv

# smaller smoke (CI runs this shape)
cargo run -p ledge-portfolio --release --features rayon --example batch -- \
  --accounts 24 --dates 12 --n 60 --k 6
```

Wall times on this shared 4-vCPU cloud VM are stable to roughly ±10%;
iteration counts are exactly reproducible (deterministic data, single
code path).
