# Smoke timings (not a competitive benchmark)

These numbers are **single-run, same-machine smoke timings** for Ledge's
synthetic factor QP generator. They exist so users can sanity-check install and
scale expectations.

They are **not**:

- a comparison against OSQP, Clarabel, MOSEK, or Gurobi;
- a claim of asymptotic performance;
- tuned or cherry-picked across seeds.

## How to reproduce

```bash
cargo run -p ledge-portfolio --release --example synthetic -- --n 500 --k 10 --seed 42
# control run with over-relaxation disabled:
cargo run -p ledge-portfolio --release --example synthetic -- --n 500 --k 10 --seed 42 --alpha 1.0
# control run with polishing disabled:
cargo run -p ledge-portfolio --release --example synthetic -- --n 500 --k 10 --seed 42 --polish false
```

Report the machine, `rustc --version`, commit SHA, and the full Markdown row
printed by the example. Cross-solver comparisons must follow
[`../benchmarks/README.md`](../benchmarks/README.md) and are published under
`../benchmarks/results/`.

## Recorded smoke table

Captured on 2026-07-21 in a cloud CI-like VM, after over-relaxation became
the default (`over_relaxation = 1.6`, on top of automatic Ruiz equilibration
with `scaling_iterations = 10`):

- OS: Linux 6.12.94+ x86_64
- CPU: Intel Xeon (4 vCPU)
- Compiler: rustc 1.83.0
- Build: `cargo run -p ledge-portfolio --release --example synthetic`
- Commit: see git history near this document's introduction

| instance | status | objective | primal residual | dual residual | iterations | time (ms) |
|---|---|---:|---:|---:|---:|---:|
| factor-n100-k5-s1 | Solved | -1.592e-2 | 9.4e-6 | 2.2e-6 | 30 | ~0.2 |
| factor-n500-k10-s42 | Solved | -1.750e-2 | 9.4e-6 | 7.0e-6 | 90 | ~2.8 |
| factor-n1000-k20-s7 | Solved | -1.836e-2 | 1.0e-5 | 5.5e-6 | 170 | ~15 |
| factor-n2000-k50-s3 | Solved | -1.839e-2 | 9.2e-6 | 6.6e-6 | 260 | ~90 |
| factor-n5000-k100-s11 | Solved | -1.875e-2 | 1.1e-5 | 6.0e-6 | 660 | ~1100 |

Same-machine control run with relaxation disabled (`--alpha 1.0`, otherwise
identical defaults) for the iteration-count comparison:

| instance | iterations (α=1.0) | iterations (α=1.6) | time α=1.0 (ms) | time α=1.6 (ms) |
|---|---:|---:|---:|---:|
| factor-n100-k5-s1 | 50 | 30 | ~0.3 | ~0.2 |
| factor-n500-k10-s42 | 170 | 90 | ~5 | ~2.8 |
| factor-n1000-k20-s7 | 310 | 170 | ~25 | ~15 |
| factor-n2000-k50-s3 | 750 | 260 | ~257 | ~90 |
| factor-n5000-k100-s11 | 1680 | 660 | ~2690 | ~1100 |

## Polishing update (2026-07-22)

Same machine class, after active-set polishing became the default
(`polish = true`, on top of scaling and over-relaxation). Iteration counts
are unchanged — polishing runs after termination — so only residuals and
wall time move. `--polish false` is the control:

| instance | worst residual (polish off) | worst residual (polish on) | time off (ms) | time on (ms) |
|---|---:|---:|---:|---:|
| factor-n100-k5-s1 | 9.4e-6 | 3.3e-15 | ~0.18 | ~0.28 |
| factor-n500-k10-s42 | 9.4e-6 | 3.9e-14 | ~2.9 | ~3.5 |
| factor-n1000-k20-s7 | 1.0e-5 | 1.2e-11 | ~16 | ~17 |
| factor-n2000-k50-s3 | 9.2e-6 | 1.6e-10 | ~99 | ~106 |
| factor-n5000-k100-s11 | 1.1e-5 | 4.1e-13 | ~1112 | ~1260 |

Polished objectives differ from raw ADMM objectives in the fourth digit or
so on these instances because the polished iterate is exactly feasible;
the raw objective was slightly flattered by ~1e-5-level constraint
violations.

Historical context: before automatic scaling (2026-07-14, same machine class)
the `n=2000, k=50` instance stopped at `MaxIterations` (10000 iterations,
primal residual 3.9e-4) and `n=1000, k=20` needed 3520 iterations. The
α=1.0 column above is the honest pre-relaxation baseline on today's code
(it already includes scaling and the continuous complementarity gate).
These are single-seed synthetic instances: they demonstrate that the
declared envelope (\(n \le 5000\), \(k \le 100\), few explicit constraints)
is reachable under default settings, not that every real problem of this
size converges.

The high-level rolling rebalance example (`cargo run -p ledge-portfolio --release
--example rebalance`) is a tiny problem and typically finishes in tens of
iterations; it is a usability demo, not a scale test.
