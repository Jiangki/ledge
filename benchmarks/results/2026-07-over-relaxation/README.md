# Comparison report after over-relaxation — 2026-07

Re-run of the [first protocol report](../2026-07/README.md) after ADMM
over-relaxation (`over_relaxation = 1.6`) became the Ledge default
(roadmap 1.5). Same machine class, same instances, same protocol; the only
solver change is Ledge's new default. External solver rows are re-measured,
not copied, and match the first report within noise.

## Artifacts

| File | Content |
|---|---|
| [`samples.csv`](samples.csv) | Every raw sample: instance, solver, repeat, phase, step, native status, Ledge-evaluated objective, independent KKT residuals, iterations, wall-clock ms |
| [`summary.md`](summary.md) | Auto-generated aggregation (median / p10 / p90) |

## Environment

- Machine: cloud VM, Intel Xeon (4 vCPU), Linux 6.12.94+ x86_64
- Compiler: rustc 1.83.0; `--release` with thin LTO, `codegen-units = 1`
- Commit: `5b33969` (branch `cursor/over-relaxation-c29a`)
- Solvers: Ledge 0.1.0-dev (this commit, `over_relaxation = 1.6`),
  OSQP 1.0.1 (crate `osqp`, its own default `alpha = 1.6`),
  Clarabel 0.11.1 (crate `clarabel`, pure Rust)
- Command:

```bash
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out benchmarks/results/2026-07-over-relaxation \
  --repeats 10 --rolling-steps 10 --commit 5b33969
```

Protocol compliance is identical to the [first report](../2026-07/README.md)
(same harness, same rules 1–7); tolerances unchanged (Ledge/OSQP
`abs 1e-6 / rel 1e-5`, Clarabel at its tighter defaults). Note that OSQP was
**already** running with its own over-relaxation default in both reports, so
this change removes an asymmetry rather than adding one.

## What changed (Ledge medians, first report → this report)

| instance | cold ms | cold iters | rolling ms/step | rolling iters |
|---|---:|---:|---:|---:|
| n=100 | 0.224 → 0.146 | 50 → 30 | 0.219 → 0.146 | 45 → 30 |
| n=500 | 4.56 → 2.57 | 170 → 90 | 4.26 → 2.19 | 155 → 75 |
| n=1000 | 25.0 → 14.1 | 310 → 170 | 18.2 → 9.95 | 215 → 120 |
| n=2000 | 256 → 91.6 | 750 → 260 | 129 → 77.7 | 330 → 215 |
| n=5000 | 2628 → 1085 | 1680 → 660 | 930 → 706 | 460 → 400 |

## Honest findings

**1. The cold-start gap to lifted Clarabel narrowed 2–3x but did not
close.** n=2000: Ledge 91.6 ms vs Clarabel 53.8 ms (was 4.8x, now 1.7x);
n=5000: 1085 ms vs 481 ms (was 5.9x, now 2.3x). Ledge now also beats
Clarabel cold at n=500 (2.57 ms vs 3.43 ms) in addition to n=100.

**2. Rolling re-solves at n ≤ 1000 are now led or shared by Ledge.**
n=500: Ledge 2.19 ms/step vs OSQP 2.35 and Clarabel 3.36; n=1000: Ledge
9.95 ms/step vs Clarabel 10.6 and OSQP 15.3. OSQP still owns the tiny-n
high-frequency point (0.081 ms/step at n=100). At n ≥ 2000 Clarabel's fixed
~9 lifted interior-point iterations still win (n=5000: 517 ms/step vs Ledge
706 ms/step).

**3. The remaining large-n deficit is no longer mainly iteration count.**
Ledge's 400 rolling iterations at n=5000 cost ~706 ms — roughly 1.8 ms per
iteration on a reduced system that is refactored from scratch every step
(setup is ~1 ms, but the per-solve `FactorizedSystem` rebuild is O(nr²)).
Factorization reuse across solves (roadmap 2.4) is now the binding lever,
ahead of further iteration-count work (vector ρ deferred accordingly).

**4. Accuracy caveat unchanged.** At default tolerances Ledge returns ~1e-5
independent residuals vs Clarabel's ~1e-11; objectives are only comparable
at comparable residuals (see the first report's finding 4). Polishing
(roadmap 2.3) remains the planned answer.

## Reproduce

```bash
cargo test -p ledge-bench-adapters --features osqp,clarabel   # correctness
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out /tmp/comparison --repeats 10 --rolling-steps 10
```

Medians on this 4-vCPU cloud VM are stable to roughly ±10% (see p10/p90
spreads in `summary.md`).
