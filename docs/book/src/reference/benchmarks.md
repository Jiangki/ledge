# Benchmarks and evidence

Ledge publishes **measured, reproducible, protocol-compliant** numbers and
nothing else. The protocol (`benchmarks/README.md` in the repository)
requires shared instance data, documented conversions, native statuses
verbatim, independent KKT re-verification of every returned point,
phase-split timing, and ≥10 repeats with all raw samples published.

## The honest headline

- **Dense-`Q` usage of a general QP solver is 1–2 orders of magnitude
  slower than factor-aware solving.** That gap is the cost of ignoring
  factor structure, measurable within the *same* external solver (dense vs
  lifted formulation). See the technical note
  [`docs/factor_structure_note.md`](https://github.com/Jiangki/ledge/blob/main/docs/factor_structure_note.md).
- **Ledge's comparative strength is rolling and turnover-aware workloads**,
  not universal cold-start speed. On instances with proportional (L1)
  transaction costs — the realistic rebalancing case — Ledge is the
  fastest solver at every size, cold and rolling, because the epigraph
  reformulation external solvers need costs them 2–6x while Ledge's prox
  block costs ~8%. On smooth instances a hand-lifted Clarabel formulation
  remains the strongest cold baseline at `n >= 2000`.
- Ledge's audited residuals at defaults are `~1e-11` or better since
  polishing; objectives across solvers agree once compared at comparable
  residuals.

![Published L1 rolling median solve times on a logarithmic scale](../assets/l1-rolling-comparison.svg)

This chart is generated from the committed `2026-07-l1` report. It is a
visual index, not a substitute for setup/cold/residual details in the report.

## Published reports

All under `benchmarks/results/` in the repository, each with raw samples:

| Report | What it measures |
|---|---|
| `2026-07/` | first protocol report (pre over-relaxation) |
| `2026-07-over-relaxation/` | re-run after α=1.6 became the default |
| `2026-07-workspace/` | re-run after factorization reuse across solves |
| `2026-07-l1/` | re-run with polish defaults + L1 turnover instances (prox block vs epigraph) |
| `2026-07-batch/` | 1 model × 500 accounts × 250 dates batch throughput |

## Reproduce

```bash
# Self-timing smoke matrix (no external solvers):
cargo run -p ledge-portfolio --release --example synthetic -- --n 500 --k 10 --seed 42

# Full protocol comparison (OSQP + Clarabel behind non-default features):
cargo run --release -p ledge-bench-adapters --features osqp,clarabel \
  --bin compare -- --out /tmp/comparison --repeats 10 --rolling-steps 10

# Batch throughput:
cargo run -p ledge-portfolio --release --features rayon --example batch
```

Read each report's findings before quoting any single number; the reports
state where external solvers win.
