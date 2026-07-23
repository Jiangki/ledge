# Comparison re-run after workspace factorization reuse — 2026-07

Third protocol-compliant comparison, re-run after roadmap 2.4 landed:
`Solver::workspace` caches the Ruiz equilibration and keeps a penalty-keyed
cache of SMW-reduced factorizations across solves. The Ledge adapter now
builds a workspace in the setup phase and re-solves through it, matching the
phase semantics OSQP always had (its factorization also persists across
`update_lin_cost` re-solves). Baseline for comparison:
[`../2026-07-over-relaxation/`](../2026-07-over-relaxation/README.md), taken
on the same machine class with the same instance set, seeds, tolerances, and
protocol.

## Artifacts

| File | Content |
|---|---|
| [`samples.csv`](samples.csv) | Every raw sample (2520 measurements): instance, solver, repeat, phase, step, native status, Ledge-evaluated objective, independent KKT residuals, iterations, wall-clock ms |
| [`summary.md`](summary.md) | Auto-generated aggregation (median / p10 / p90) |

## Environment

- Machine: cloud VM, Intel Xeon (4 vCPU), 15 GiB RAM, Linux 6.12.94+ x86_64
- Compiler: rustc 1.83.0; `--release` with thin LTO, `codegen-units = 1`
- Commit: `890fcd7` (branch `cursor/factorization-workspace-c29a`)
- Solvers: Ledge 0.1.0-dev (this commit, defaults incl. `over_relaxation
  1.6`, `scaling_iterations 10`), OSQP 1.0.1 (crate `osqp`), Clarabel 0.11.1
  (crate `clarabel`, pure Rust)
- Command:

```bash
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out benchmarks/results/2026-07-workspace --repeats 10 \
  --rolling-steps 10 --commit 890fcd7
```

## Protocol compliance

Identical to the previous reports (same data, documented conversions, native
statuses verbatim, independent `check_kkt` on original data, phase-split
uniform wall clocks, 10 repeats with all samples published). One semantic
change is disclosed explicitly:

**Ledge setup now includes equilibration + first factorization.** Previous
reports charged Ledge's whole setup to the cold/rolling phases because no
persistent state existed. With the workspace, setup does the work OSQP's
setup always did (symbolic + numeric factorization), so the phases are now
*more* symmetric across solvers, but Ledge's setup/cold split is not directly
comparable with the pre-2.4 reports. Total time is: at n=5000, old cold
1084.9 ms ≈ new setup 71.0 ms + new cold 1009.4 ms.

Workspace solves replay the one-shot penalty policy, so iterate paths — and
therefore iteration counts and residuals — are identical to `Solver::solve`
of the same data; only setup cost moves. Tolerances unchanged: Ledge defaults
(`abs 1e-6`, `rel 1e-5`); OSQP matched; Clarabel at its own tighter defaults
(≈1e-8, to its disadvantage in time and advantage in accuracy).

## Findings (deltas vs the over-relaxation report)

**1. Rolling re-solves got 10–29% faster per step; Ledge now leads or ties
rolling up to n = 1000 and narrows the large-n gap.** Median ms/step, Ledge:

| instance | before | after | change | best external (rolling) |
|---|---:|---:|---:|---:|
| n=100 | 0.146 | 0.104 | −29% | osqp (lifted) 0.079 |
| n=500 | 2.187 | 1.871 | −14% | osqp (lifted) 2.347 |
| n=1000 | 9.951 | 8.937 | −10% | clarabel (lifted) 10.610 |
| n=2000 | 77.742 | 67.511 | −13% | clarabel (lifted) 54.120 |
| n=5000 | 706.227 | 581.878 | −18% | clarabel (lifted) 518.161 |

At n=1000 Ledge's warm rolling step now beats Clarabel's fixed ~9
interior-point iterations (8.9 vs 10.6 ms). At n=5000 the gap shrank from
1.37x (706 vs 517) to 1.12x (582 vs 518). Iteration counts are unchanged by
design — the entire saving is the removed per-step equilibration, `O(nr²)`
setup, and adaptive-ρ ladder refactorizations.

**2. Cold solves improved only by the setup they no longer contain.**
n=5000: 1084.9 → 1009.4 ms cold with setup measured separately at 71.0 ms.
Clarabel (lifted) remains the strongest cold baseline at n ≥ 2000 (54 ms at
n=2000, 479 ms at n=5000). The comparative story is unchanged: cold-start
parity at large n is not claimed.

**3. Setup is no longer effectively zero for Ledge, and that is honest.**
Ledge setup at n=5000 is 71 ms vs 144 ms OSQP (lifted) and 68 ms Clarabel
(lifted). Previous reports showed Ledge setup ≤1 ms because the real setup
work was hidden inside every solve; it is now paid once, where a rolling
user actually pays it.

**4. Remaining large-n lever is iteration count, not setup.** With setup
amortized, a warm n=5000 step is ~400 iterations × ~1.4 ms. Clarabel spends
~10 iterations. Closing the residual 1.12x (and the accuracy gap, ~1e-5 vs
~1e-11 independent residuals) points at polishing (roadmap 2.3) and
vector ρ (deferred from 1.5), exactly as recorded in the roadmap.

**Accuracy caveat (unchanged).** At default tolerances Ledge and OSQP return
~1e-5 residual points whose objectives can appear below the tight-tolerance
optimum; objectives are only comparable at comparable independent residuals.

## Reproduce

```bash
cargo test -p ledge-bench-adapters --features osqp,clarabel   # correctness
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out /tmp/comparison --repeats 10 --rolling-steps 10
```

Medians on this 4-vCPU cloud VM are stable to roughly ±10%; warm rolling
steps have wider spreads (p10/p90 in `summary.md`) because warm-start
quality varies per perturbation step.
