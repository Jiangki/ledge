# First protocol comparison report — 2026-07

First cross-solver comparison published for Ledge, satisfying every rule in
[`../../README.md`](../../README.md). Read the honest-findings section
before quoting any single number.

## Artifacts

| File | Content |
|---|---|
| [`samples.csv`](samples.csv) | Every raw sample (2520 measurements): instance, solver, repeat, phase, step, native status, Ledge-evaluated objective, independent KKT residuals, iterations, wall-clock ms |
| [`summary.md`](summary.md) | Auto-generated aggregation (median / p10 / p90) |

## Environment

- Machine: cloud VM, Intel Xeon (4 vCPU), 15 GiB RAM, Linux 6.12.94+ x86_64
- Compiler: rustc 1.83.0; `--release` with thin LTO, `codegen-units = 1`
- Commit: `6831e5b` (branch `cursor/comparison-adapters-c29a`)
- Solvers: Ledge 0.1.0-dev (this commit), OSQP 1.0.1 (crate `osqp`),
  Clarabel 0.11.1 (crate `clarabel`, pure Rust)
- Command:

```bash
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out benchmarks/results/2026-07 --repeats 10 \
  --rolling-steps 10 --commit 6831e5b
```

## Protocol compliance

1. **Same data** — all solvers consume the identical `QpProblem` from the
   deterministic generator (smoke matrix: n=100..5000, k=5..100, one budget
   equality, 4 random inequality rows, long-only boxes).
2. **Convention conversion** — all three solvers natively minimize
   `0.5 x'Qx + q'x`; no objective rescaling. External solvers receive two
   documented encodings: `dense-Q` (materialized covariance, capped at
   n≤1000) and `lifted` (k auxiliary variables, sparse objective — the
   strongest reasonable baseline). See `benchmarks/adapters/src/convert.rs`.
3. **Warm starts** — cold solves share one primal start (the feasible
   uniform portfolio). Rolling re-solves warm-start from each solver's own
   previous solution: Ledge and OSQP primal + dual; Clarabel does not accept
   warm starts (interior point) and this is recorded, not worked around.
4. **Native statuses** — recorded verbatim in `samples.csv`; every solve in
   this report terminated `solved` by its own criterion.
5. **Independent verification** — every returned point was re-checked with
   Ledge's `check_kkt` against the original (never scaled, never lifted)
   data; residual columns in `summary.md` come from that checker, not from
   solver self-reports.
6. **Phase split** — setup (conversion + solver construction), cold solve,
   and 10 rolling re-solves with deterministically perturbed expected
   returns are timed separately with uniform wall clocks.
7. **Repeats** — 10 repeats per (instance, solver); all samples published;
   summary reports median and quantiles.

Tolerances: Ledge defaults (`abs 1e-6`, `rel 1e-5`); OSQP set to match
(`eps_abs 1e-6`, `eps_rel 1e-5`, `max_iter 10000`); Clarabel left at its own
defaults (≈1e-8 feasibility/gap — **tighter** than the ADMM solvers, to its
disadvantage in time and advantage in accuracy). The independent residual
columns make the achieved-accuracy differences visible.

## Honest findings

**1. Exploiting factor structure is worth 1–2 orders of magnitude — but a
lifted formulation captures most of it.** On `dense-Q` (what a general QP
solver receives when nobody exploits structure), both OSQP and Clarabel
degrade sharply: at n=1000 Clarabel needs ~1.2 s per solve dense vs ~11 ms
lifted, OSQP ~890 ms dense vs ~82 ms lifted. The lifted reformulation —
which any capable user can write by hand — recovers the structure without
Ledge. Ledge's value is doing this automatically, not exclusively.

**2. Ledge does not win cold starts against a well-formulated Clarabel.**
Cold, lifted formulation, medians: Ledge is fastest at n=100 (0.22 ms) and
beats OSQP everywhere (about 2x at every size), but from n=500 upward
Clarabel (lifted) is the strongest baseline — n=2000: 54 ms vs Ledge 256 ms;
n=5000: 449 ms vs Ledge 2628 ms — while also reaching ~1e-11 residuals vs
Ledge's ~1e-5.

**3. Rolling re-solves narrow but do not close the gap at large n.** Warm
starts cut Ledge's per-step cost about 3x (n=5000: 930 ms/step vs 2628 ms
cold). OSQP benefits similarly at small n (0.08 ms/step at n=100, the best
rolling number in the table). But Clarabel's fixed ~9 interior-point
iterations on the lifted system still win at n≥1000 (n=5000: 484 ms/step)
even with zero warm-start capability, because Ledge's ADMM iteration count
grows with n (50 at n=100 → 1680 cold / ~460 warm at n=5000) and every
Ledge solve currently rebuilds its reduced factorization.

**4. Accuracy caveat when reading the objective column.** At default
tolerances Ledge (and OSQP at matched tolerances) return points with ~1e-5
primal residuals; tiny box/budget violations can make their objectives
appear *below* the true optimum (Ledge n=5000 cold: -1.8810e-2 vs the
tight-tolerance optimum ≈ -1.8631e-2). Objectives across solvers are only
comparable at comparable independent residuals.

**Implications recorded for the roadmap.** The honest competitive story
today is (a) structure exploitation without manual reformulation, (b) pure
Rust embedding with zero native dependencies and near-zero setup cost
(Ledge setup ≤1 ms at n=5000 vs 68–136 ms for the lifted externals), and
(c) winning the small-n / high-frequency regime. It is *not* raw large-n
speed: closing that gap needs fewer ADMM iterations (over-relaxation,
better ρ policies — roadmap 1.5), factorization reuse across solves
(roadmap 2.4), and polishing for accuracy parity (roadmap 2.3). Per
`docs/PLAN.md` §6, rolling sequence performance is the primary comparative
story; cold-start results remain visible and are not overstated.

## Reproduce

```bash
cargo test -p ledge-bench-adapters --features osqp,clarabel   # correctness
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out /tmp/comparison --repeats 10 --rolling-steps 10
```

Numbers move with hardware; medians on this 4-vCPU cloud VM are stable to
roughly ±10% (see p10/p90 spreads in `summary.md`).
