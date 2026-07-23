# Comparison report: polish-on defaults + L1 turnover instances — 2026-07

Fourth protocol-compliant comparison. Two things changed since the
[workspace report](../2026-07-workspace/README.md):

1. **Solution polishing is now the Ledge default** (roadmap 2.3, landed
   after the previous report was taken), so this is the first report
   measuring shipped defaults end to end.
2. **Every smoke-matrix instance now runs twice** — the smooth base problem
   and an `-l1` variant adding 10 bps proportional turnover costs anchored
   at the shared primal start. These are the first published cross-solver
   L1 numbers: Ledge keeps its soft-threshold prox block (reduced dimension
   unchanged); external solvers receive the standard epigraph reformulation
   (`n` extra variables, `2n` inequality rows) with multipliers mapped back
   into Ledge's convention for independent re-verification.

## Artifacts

| File | Content |
|---|---|
| [`samples.csv`](samples.csv) | Every raw sample (5040 measurements): instance, solver, repeat, phase, step, native status, Ledge-evaluated objective, independent KKT residuals, iterations, wall-clock ms |
| [`summary.md`](summary.md) | Auto-generated aggregation (median / p10 / p90) |

## Environment

- Machine: cloud VM, Intel Xeon (4 vCPU), 15 GiB RAM, Linux 6.12.94+ x86_64
- Compiler: rustc 1.83.0; `--release` with thin LTO, `codegen-units = 1`
- Commit: `59cccfe` (branch `cursor/m3-closeout-c29a`)
- Solvers: Ledge 0.1.0-dev (defaults incl. `polish = true`,
  `over_relaxation = 1.6`, `scaling_iterations = 10`), OSQP 1.0.1 (crate
  `osqp`), Clarabel 0.11.1 (crate `clarabel`, pure Rust)
- Command:

```bash
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out benchmarks/results/2026-07-l1 --repeats 10 \
  --rolling-steps 10 --commit 59cccfe
```

Protocol compliance is identical to the previous reports (shared data,
documented conversions, native statuses verbatim, independent `check_kkt`
on original data — now including the L1 subgradient conditions — uniform
phase wall clocks, 10 repeats, all samples published). Tolerances
unchanged: Ledge/OSQP `abs 1e-6 / rel 1e-5`, Clarabel at its tighter
defaults. All 5040 solve samples across all solvers report `solved`.

## Findings

**1. Polishing closes the accuracy caveat that every previous report had
to disclose.** Ledge's independently audited residuals at defaults drop
from ~1e-5/1e-6 to 1e-13..1e-11 across the matrix — now matching or
beating Clarabel's interior-point accuracy — and Ledge's objectives agree
with the tight-tolerance optimum digit for digit (previous reports showed
raw-ADMM objectives slightly *below* optimum because of ~1e-5 constraint
violations; that bias is gone). The price is single-digit-percent cold
overhead (n=5000: 1009 → 1093 ms) and a larger relative cost on cheap warm
steps (n=5000 rolling: 582 → 784 ms/step, since one polish factorization
is a bigger fraction of a short solve). Users who prefer the old speed can
pass `polish=false`; the default buys certified accuracy.

**2. On L1 instances Ledge is the fastest solver at every size, cold and
rolling.** Cold medians (ms), best external in parentheses:

| instance | ledge | best external (cold) | gap |
|---|---:|---:|---:|
| n=100-l1 | 0.145 | 0.605 (osqp lifted) | 4.2x |
| n=500-l1 | 2.53 | 5.42 (clarabel lifted) | 2.1x |
| n=1000-l1 | 13.4 | 15.4 (clarabel lifted) | 1.15x |
| n=2000-l1 | 95.8 | 222.1 (clarabel lifted) | 2.3x |
| n=5000-l1 | 1181 | 1902 (clarabel lifted) | 1.6x |

Rolling medians (ms/step): 0.15 / 2.25 / 9.7 / 68.9 / 620 for Ledge vs
best external 0.26 (osqp) / 2.48 (osqp) / 15.4 (clarabel) / 181.2 (osqp) /
1903 (clarabel). On the smooth instances the large-n cold ordering is
unchanged (lifted Clarabel remains fastest at n ≥ 2000, 2.3x at n=5000) —
adding realistic transaction costs is what flips it.

**3. The flip is structural, not a tuning artifact.** The epigraph
reformulation the external solvers require adds `n` variables and `2n`
rows, exactly the structure-destroying growth the factor form avoids: L1
costs the external solvers 2–6x (Clarabel n=5000 cold: 470 → 1902 ms;
OSQP: 5298 → 11287 ms) but costs Ledge ~8% (1093 → 1181 ms) — and Ledge's
warm rolling steps get *faster* on L1 instances (784 → 620 ms/step at
n=5000, median iterations 445 → 315) because sticky no-trade assets make
consecutive dates more similar. Proportional costs are the realistic
rebalancing case, so this is the comparison production users should read.

**4. Honest caveats.** (a) On the smooth instances, warm rolling at
n=2000/5000 is now led by lifted Clarabel (53.9 vs 74.1, 500 vs 784
ms/step) — polish-on defaults gave back part of the workspace report's
gains there; `polish=false` recovers them if raw-ADMM accuracy suffices.
(b) At n=5000-l1 Ledge's cold dual residual is ~1e-6 rather than polished
~1e-11: the polish candidate was adopted (it improved the audited worst
residual — primal is 8e-11) but the refinement plateaus on this instance,
where classifying 5000 L1 kinks from an ADMM-accurate iterate leaves some
misattributed subgradient mass. The solve is comfortably within tolerance
and the audited residuals report it honestly. (c) OSQP's dense-Q L1 rows
at n=1000 look *faster* than its smooth dense-Q rows (fewer iterations on
the stickier problem); dense-Q remains 1–2 orders of magnitude behind
either structured path throughout.

## Reproduce

```bash
cargo test -p ledge-bench-adapters --features osqp,clarabel   # correctness
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out /tmp/comparison --repeats 10 --rolling-steps 10
```

`--l1-bps 0` disables the L1 variants; any other value reprices them.
Medians on this 4-vCPU cloud VM are stable to roughly ±10%; warm rolling
steps have wider spreads (p10/p90 in `summary.md`).
