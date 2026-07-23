# Ledge project plan

This is the public technical plan for Ledge: a factor-model portfolio
rebalancing engine written in Rust with first-class Python bindings.
Milestone execution details live in [`ROADMAP.md`](ROADMAP.md), technical
decisions in [`DECISIONS.md`](DECISIONS.md), and solver mathematics in
[`algorithm.md`](algorithm.md).

> **License:** Apache-2.0. Source visibility, clean-history publication, and
> registry releases are controlled by the
> [public-release runbook](OPEN_CORE.md#5-public-release-runbook-roadmap-16-gate).

Commercial pricing, customer, revenue, and pilot strategy is intentionally
not maintained in this repository. The repository-level open-core boundary
and the rules for any future private extension are public and operationally
defined in [`OPEN_CORE.md`](OPEN_CORE.md).

---

## 1. Positioning and scope

**Ledge is a rebalancing execution engine for factor risk models.**

Inputs are factor loadings \(F\), factor covariance \(\Omega\),
idiosyncratic variances \(d\), expected returns \(\mu\), constraints, and
previous weights. Outputs are auditable weights, duals, independent KKT
residuals, diagnostics, and reproducible rolling solves.

### Goals, in priority order

1. **Trustworthy.** Inside the declared envelope (continuous convex factor
   QPs; roughly \(n \le 5000\), \(k \le 100\), explicit linear rows
   \(m \le 200\)), defaults should converge reliably. Successes have
   independently checked KKT residuals; failures have useful diagnostics or
   checkable infeasibility certificates.
2. **Usable.** Rust and NumPy APIs should cover budget, boxes,
   industry/style exposures, smooth L2 turnover, exact L1 transaction costs,
   tracking error, rolling sequences, and account batches.
3. **Embeddable.** The core stays pure Rust, deterministic, free of required
   BLAS/LAPACK or commercial runtimes, and suitable for backtests and
   services.
4. **Evidenced.** Performance claims must follow the written
   [`benchmark protocol`](../benchmarks/README.md), publish raw samples, and
   say where external solvers win.

### Non-goals

- MIP/MIQP, general nonlinear programming, or a general modeling language.
- GPU or distributed solving.
- Alpha generation or risk-model estimation; \(F\), \(\Omega\), and \(d\)
  are inputs.
- Unverified “faster than X” marketing.
- Withholding numerical correctness or intentionally slowing the public
  edition.
- SOCP before 1.0.

---

## 2. Why factor-model rebalancing

Ledge targets users who already have \(F/\Omega/d\) and repeatedly solve
daily, weekly, or monthly portfolio rebalances.

1. **The structure fits.** The SMW reduction scales with
   \(r = k + m\), not the number of entries in a dense \(n \times n\)
   covariance matrix.
2. **Repeated solves matter.** Rolling dates mostly change \(\mu\), the
   turnover anchor, a benchmark, or right-hand sides. That makes warm starts
   and factorization reuse part of the API rather than user boilerplate.
3. **The scope is auditable.** A narrow problem class supports a clear test
   matrix, honest limitations, and independent residual checks.
4. **General solvers remain valid baselines.** cvxpy, OSQP, and Clarabel are
   broader tools. Ledge should be chosen only when its native factor and
   rebalance semantics are useful.

---

## 3. Technical acceptance gates

| Gate | Acceptance evidence | Status |
|---|---|---|
| Trust | `n=2000, k=50` and the declared smoke matrix solve under defaults; comparison protocol and raw data published | Done |
| Rebalancing | Exact L1 turnover, certificates, polishing, tracking error, and measured rolling warm-start effect | Done |
| Workflows | Constraint templates, serialization, and 1 model × 500 accounts × 250 dates batch evidence | Done |
| Distribution | Public OSS license, crates.io packages, PyPI wheels, public docs | Done — `0.2.0` published |
| Stability | 1.0 compatibility, semver/MSRV, and deprecation review | Pending |

The current measured evidence is indexed in
[`docs/book/src/reference/benchmarks.md`](book/src/reference/benchmarks.md).
Synthetic workloads are useful regression evidence, not a guarantee for
every real portfolio.

---

## 4. Architecture and dependency policy

```text
Python / NumPy ─┐
                ├─> ledge (portfolio semantics)
Rust callers ───┘        │
                         └─> ledge-core
                             ├─ scaling + ADMM/SMW
                             ├─ L1 prox + polishing
                             ├─ certificates + KKT audit
                             └─ workspace + sequence + batch

benchmarks/adapters ──> OSQP / Clarabel (non-default comparison features)
```

| Choice | Reason |
|---|---|
| Rust core with `forbid(unsafe_code)` | Embeddable and auditable; reduced dense systems stay small in the intended envelope |
| PyO3 + maturin; NumPy-only runtime | Direct access for quantitative Python users with a small runtime surface |
| ADMM + SMW reduction | Preserves native factor structure and supports separable prox blocks |
| `serde` and `rayon` as opt-in Rust features | Default builds remain small; Python wheels enable the workflow features |
| `criterion`, `proptest`, cvxpy+Clarabel tests | Performance regression, invariant testing, and an independent correctness oracle |
| mdBook + GitHub Pages | One lightweight documentation stack |

Sparse \(F\), a C API, and WASM remain demand-triggered. Async/service
frameworks do not belong in the solver core.

---

## 5. Open-core boundary

This entire repository is intended to become the Apache-2.0 open core. There
is no second “open-source folder” and no private implementation code in this
tree. The complete path inventory and release procedure are in
[`OPEN_CORE.md`](OPEN_CORE.md).

The public core includes:

- the full numerical solver, scaling, certificates, polishing, exact L1 prox,
  and independent KKT checks;
- portfolio constraints, turnover and tracking semantics;
- workspaces, rolling sequences, and in-process multi-thread account batch;
- Rust/Python APIs, serialization, examples, docs, benchmark adapters, and
  raw reports.

Any future proprietary product must live in a **separate private
repository**, consume only published public APIs, and must not withhold
numerical fixes or functionality required for a complete single-machine
solver. Persistent orchestration, deployment, customer-specific integration,
and service operations are examples of work that may live outside the core;
the authoritative rule is [`OPEN_CORE.md` §2](OPEN_CORE.md#2-open-core-inventory-everything-here-is-open).

---

## 6. Validation and benchmark discipline

1. Unit tests and `proptest` invariants.
2. cvxpy+Clarabel gold checks with independently recomputed residuals.
3. Regression instances for fixed bugs.
4. Ill-conditioned and infeasible suites.
5. Criterion baselines and release-build smoke tests.
6. Protocol comparisons with shared instances, declared formulations,
   phase-split timings, at least ten repeats, and all raw samples committed.

Cloud timing is noisy. Public numbers must record machine, compiler, commit,
command, and caveats. Rolling sequence and transaction-cost workloads are
the primary comparative story; cold-start results must still be shown.

---

## 7. Release policy

- Every release follows the controls in
  [`OPEN_CORE.md` §5](OPEN_CORE.md#5-public-release-runbook-roadmap-16-gate)
  and must make `./scripts/check_open_core.sh --release` pass.
- Each release follows: changelog → tests → package checks → tag → crates.io
  (`ledge-core`, then `ledge-portfolio`) → PyPI wheels → GitHub Release
  linking the evidence used for claims.
- Pre-1.0 breaking changes require migration notes.
- At 1.0, semver promises begin and the README must state the MSRV and
  deprecation policy.
- Apache-2.0 inbound=outbound is the contributor model; no CLA is planned.

---

## 8. Current priorities

1. Collect redacted real workloads for the regression set.
2. Evaluate sparse factor storage only if those workloads demonstrate need.
3. Review the API, MSRV, semver, and deprecation policy before 1.0.

---

## 9. Document map

| Path | Role |
|---|---|
| [`PLAN.md`](PLAN.md) | Public technical scope, architecture, quality, and release policy |
| [`ROADMAP.md`](ROADMAP.md) | Executable milestones, task status, and technical notes |
| [`DECISIONS.md`](DECISIONS.md) | Public engineering/open-core decision log |
| [`OPEN_CORE.md`](OPEN_CORE.md) | Authoritative path boundary and public-release runbook |
| [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) | Copyable maintainer sign-off for irreversible release steps |
| [`algorithm.md`](algorithm.md) | Solver mathematics |
| [`factor_structure_note.md`](factor_structure_note.md) | Measured explanation of factor-structure exploitation |
| [`cvxpy_migration.md`](cvxpy_migration.md) | Executed cvxpy-to-Ledge mappings |
| [`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md) | Self-timing smoke evidence |
| [`examples/`](examples/) | Reproducible end-to-end workflows |
| [`book/`](book/) | Reader-facing mdBook source |
| [`assets/`](assets/) | Visual assets and reproducibility notes |
| [`../benchmarks/`](../benchmarks/) | Comparison protocol, adapters, raw samples, and reports |

Review this plan at milestone boundaries, log public technical decisions in
`DECISIONS.md`, and keep commercial strategy outside this repository.
