# Roadmap

Executable public engineering roadmap after **0.1.0**. Scope, architecture,
quality policy, and release policy live in [`PLAN.md`](PLAN.md); the
repository boundary and public-release procedure live in
[`OPEN_CORE.md`](OPEN_CORE.md).

Milestones are roughly 6–10 weeks each. **Do not enter the next milestone
until the previous exit criteria pass.** Dates are guidance, not promises.

A short summary also appears in the root
[README](../README.md#project-status-and-roadmap).

---

## Milestone overview

| Milestone | Delivery | Theme |
|---|---|---|
| **M0** | 0.1.x | Make “it runs” into “others can start without traps” |
| **M1** | 0.2.0 | Trust — scaling, PyPI, first honest comparison |
| **M2** | Shipped early in 0.2.0 | Usability — L1, certificates, polishing, rolling cache |
| **M3** | Shipped early in 0.2.0 | Vertical productization — batch & workflows |
| **M4** | 0.5 → 1.0 | API freeze, compatibility policy, ecosystem evidence |

---

## M0 — Immediate (0.1.x patches)

**Goal:** strangers can install, test, and report useful issues.

| # | Task | Location | Status |
|---|---|---|---|
| 0.1 | Land [`PLAN.md`](PLAN.md), rewrite this roadmap, open [`DECISIONS.md`](DECISIONS.md) | `docs/` | Done |
| 0.2 | `criterion` bench skeleton: x-update, `FactorizedSystem::new`, end-to-end synthetic | `crates/ledge-core/benches/` | Done — `cargo bench -p ledge-core --features bench-internals` |
| 0.3 | `proptest` tests: random feasible convex QP → Solved with KKT ≤ tol; warm vs cold agree | `crates/ledge-core/tests/` | Done — `tests/proptest_kkt.rs` |
| 0.4 | Python gold tests: cvxpy+Clarabel oracle on ~20 random instances (test extra) | `python/tests/test_reference.py` | Done — `pip install -e "python/[test]"` |
| 0.5 | Richer `MaxIterations` diagnostics (final residuals + heuristic hints) | `crates/ledge-core/src/solver.rs` | Done — `Solution.diagnostics` / Python `convergence_hints` |
| 0.6 | Issue templates: bug / performance / problem-instance | `.github/ISSUE_TEMPLATE/` | Done |

**Exit criteria:** CI green including new tests; benches reproducible locally
with one command.

---

## M1 — 0.2.0 “Trust”

**Goal:** fix known default convergence failures; ship wheels; ship first
honest comparison.

| # | Task | Notes |
|---|---|---|
| 1.1 | **Ruiz equilibration** | Done — `scaling.rs`, default `scaling_iterations: 10`; see Technical notes §1 |
| 1.2 | Objective cost scaling (scalar \(c\), OSQP-style) | Done — shipped with 1.1 |
| 1.3 | Unscaled residual reporting: `check_kkt` always on original data | Done — termination and all reported residuals use original data |
| 1.4 | Expand smoke matrix to \(n=5000\), \(k=100\); update [`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md) and README support statement | Done — full matrix Solved under defaults (2026-07-21) |
| 1.5 | Over-relaxation (\(\alpha \approx 1.6\)) and optional vector ρ | Over-relaxation done — default `over_relaxation: 1.6`, smoke-matrix iterations cut 1.7–2.9x (n=5000: 1680 → 660); see Technical notes §1a. Vector ρ re-evaluated 2026-07-22 with a measured prototype and **stays deferred**: no static per-block factor wins across the smoke matrix (1.6x gains at n=1000/2000 with explicit rows, 1.6x losses at n=5000); see [`DECISIONS.md`](DECISIONS.md) |
| 1.6 | **Public-release gate** — Apache-2.0, resolve the occupied crates.io `ledge` package name, enable publishing, publish a clean-root GitHub repository | Done — clean-root public repository and `v0.2.0` tag published |
| 1.7 | `maturin-action` wheels + PyPI Trusted Publishing → `ledge-portfolio 0.2.0` | Done — abi3 wheels and sdist published to PyPI |
| 1.8 | OSQP / Clarabel adapters (non-default features) + first protocol report | Done — `benchmarks/adapters`, report in `benchmarks/results/2026-07/` |
| 1.9 | Short technical note: “why factor structure is worth exploiting” using 1.8 data | Done — [`factor_structure_note.md`](factor_structure_note.md): measured dense-Q vs lifted gap (1–2 orders of magnitude inside the same solver), SMW cost model, what native factor form adds beyond lifting, honest large-n limits; every number cited from published reports |

**Exit criteria:**

- `n=2000, k=50` Solved under defaults.
- Repository licensed Apache-2.0 and published from a clean root — done;
  `pip install ledge-portfolio==0.2.0` verified against PyPI.
- Comparison report published and satisfies all rules in
  [`../benchmarks/README.md`](../benchmarks/README.md).

---

## M2 — “Usable” (shipped early in 0.2.0)

**Goal:** cover real rebalancing needs so factor-model users can move repeated
portfolio QPs out of general modeling boilerplate when this niche fits.

| # | Task | Notes |
|---|---|---|
| 2.1 | **Exact L1 turnover prox** | Done — dedicated soft-threshold consensus block (`L1Term` on `QpProblem`, `PortfolioProblem::with_l1_turnover`, Python `l1_turnover_costs`); reduced dimension \(r\) unchanged; KKT audit scores the L1 subgradient, polishing pins no-trade assets, sequences move the anchor in place; validated against epigraph reformulations (Rust + proptest) and cvxpy+Clarabel (Python); see Technical notes §3 |
| 2.2 | **Infeasibility certificates** | Done — `PrimalInfeasible` / `DualInfeasible` statuses with normalized, independently auditable certificates (`certificate.rs`, `check_primal_certificate` / `check_dual_certificate`) and portfolio-vocabulary hints; see Technical notes §2 |
| 2.3 | **Polishing** | Done — active-set refinement after `Solved` (`polish.rs`, default `polish: true`); smoke-matrix residuals drop from ~1e-5/1e-6 to 1e-11..1e-15 for ~7-13% extra wall time; adopted only when the audited worst KKT residual improves; see Technical notes §2a |
| 2.4 | **Factorization reuse across solves** | Done — `Solver::workspace` → `workspace.rs`; scaling built once, penalty-keyed factorization cache, linear/rhs updates in place, iterate paths identical to one-shot solves; see Technical notes §4 and `benchmarks/results/2026-07-workspace/` |
| 2.5 | `solve_sequence` API (Rust + Python) | Done — `PortfolioProblem::sequence()` → `PortfolioSequence::solve_next(RebalanceStep)` plus one-call `solve_sequence` (`sequence.rs`); Python `PortfolioProblem.sequence()` / `PortfolioSequence.solve_next(...)`. Warm starts chained and workspace factorization cache inside; see Technical notes §4a |
| 2.6 | Tracking-error sugar: minimize \((w-b)^\mathsf{T}\Sigma(w-b)\) | Done — `PortfolioProblem::with_tracking_benchmark` / Python `benchmark_weights`; pure linear-cost shift \(-\lambda\Sigma b\) through the factor structure, same QP underneath; `RebalanceStep.benchmark_weights` rolls the benchmark without refactorizing; see Technical notes §3a |
| 2.7 | Public-data-style rolling example + published warm-start numbers | Done — `docs/examples/rolling_backtest.py`: momentum backtest (300 assets, 12 factors, 24 dates) with 10 bps L1 costs and a tracking benchmark; warm dates 79 → 37 iterations (2.1x) and 1.71 → 0.91 ms (1.9x) vs cold, factorizations 24 → 2; numbers published in `docs/examples/README.md` |
| 2.8 | Collect maintainer interviews and redacted real workloads from issues / downloads | Open — external adoption needed |

**Exit criteria:**

- cvxpy migration guide usable — done:
  [`cvxpy_migration.md`](cvxpy_migration.md); every mapping in it is
  executed against cvxpy+Clarabel in CI
  (`python/tests/test_migration_guide.py`).
- Rolling benchmark numbers published — done (2.7 and
  `benchmarks/results/2026-07-workspace/`).
- ≥2 external (redacted) real workloads in the regression set — open;
  depends on 2.8 outreach.

---

## M3 — “Vertical productization” (shipped early in 0.2.0)

**Goal:** grow from “solver” into a documented rebalancing engine while
keeping the repository boundary mechanically checkable.

| # | Task | Notes |
|---|---|---|
| 3.1 | Constraint template builders: industry-neutral, style bounds, concentration, short limits → existing linear constraints | Done — `with_industry_neutrality` / `with_group_targets` (equality rows), `with_style_bounds` (inequality/equality rows), `with_concentration_limit` / `with_short_limit` (box tightening, no new rows); templates append to user constraint blocks so sequences roll their targets via `RebalanceStep` RHS updates; Python kwargs `industry_ids=` / `industry_targets=` / `style_matrix=` / `style_lower=` / `style_upper=` / `max_weight=` / `max_short=`; see Technical notes §6 |
| 3.2 | Multi-thread batch over the account axis (`rayon`, feature-gated); publish “1×500×250” throughput | Done — `solve_batch(&[BatchAccount], settings)` runs one `PortfolioSequence` per account, in parallel over accounts behind the non-default `rayon` feature (serial with identical results without it); optional backtest anchor chaining (`chain_previous_weights`); per-account error isolation; Python `ledge.solve_batch(problems, steps, ...)`. Published: 1 model × 500 accounts × 250 dates (n=200, k=15) in 12.9 s on 4 vCPUs = 9.7k solves/s, 4.0x over serial ([`benchmarks/results/2026-07-batch/`](../benchmarks/results/2026-07-batch/README.md)); see Technical notes §8 |
| 3.3 | Problem / solution serialization (serde + JSON/binary) for bug reproduction | Done — non-default `serde` feature on `ledge-core` / `ledge-portfolio`: `QpProblem`, `PortfolioProblem`, `SolverSettings`, `WarmStart`, `Solution` (duals, residuals, certificates included) work with any serde format; deserialization re-runs construction validation; Python `PortfolioProblem.to_json()` / `from_json()` and `SolveResult.to_json()`; see Technical notes §7 |
| 3.4 | Evaluate sparse `F`; implement CSR path only if real workloads need it | |
| 3.5 | mdBook + GitHub Pages docs site (tutorial, migration, API, tuning) | Done — `docs/book/` with tutorial / guide / reference chapters; CI builds it on every push, and `docs-deploy.yml` publishes the public Pages site manually |
| 3.6 | **Open-core boundary review** — verify new workflows remain in-core and no private implementation entered this tree | Done — [`OPEN_CORE.md`](OPEN_CORE.md) + `scripts/check_open_core.sh` |

**Exit criteria:** batch throughput published — done
([`benchmarks/results/2026-07-batch/`](../benchmarks/results/2026-07-batch/README.md));
docs site built in CI and deployed to GitHub Pages — done; repository boundary recorded in
[`OPEN_CORE.md`](OPEN_CORE.md) — done.

---

## M4 — 0.5 → 1.0 stability

- API freeze review → **1.0** with semver, MSRV policy, and a documented
  deprecation policy.
- Validate the API against external, redacted real workloads before freezing
  it; synthetic evidence alone is not enough for 1.0.
- Keep all solver correctness and single-machine workflow improvements in
  this repository. Any future private integration must live in a separate
  repository and consume only published APIs, as required by
  [`OPEN_CORE.md`](OPEN_CORE.md).

---

## Technical notes (implementation detail)

### 1. Ruiz equilibration (M1) — implemented

- Iterated row/column ∞-norm balancing: variable scale \(E=\mathrm{diag}(e)\),
  per-block constraint scales, objective scalar \(c\).
- **Factor-structure key:** never form \(Q\) explicitly.
  \(E Q E = (E G)(E G)^\mathsf{T} + E\,\mathrm{diag}(d)\,E\) with
  \(G G^\mathsf{T} = F\Omega F^\mathsf{T}\) — scaling touches rows of \(G\)
  and \(d\); SMW reduction is unchanged.
- Column norms of \(Q\) estimated from the exact diagonal plus a
  Cauchy-Schwarz bound; estimate quality affects conditioning only.
- Module: `scaling.rs` → `ScaledProblem`. The solver iterates in scaled
  space; warm starts are scaled in, and termination, `check_kkt`, and the
  returned `Solution` are always evaluated on the original data.
- Setting: `SolverSettings { scaling_iterations: 10 }` (`0` disables);
  exposed in Python as `scaling_iterations`.
- Acceptance held: ill-conditioned suite (`tests/scaling.rs`) where scaling
  off → `MaxIterations`, scaling on → `Solved`; details in
  [`algorithm.md`](algorithm.md) §3.

### 1a. Over-relaxation (M1) — implemented

- Every consensus block (equality, inequality, box) sees the blend
  \(\alpha Ax^{k+1} + (1-\alpha)z^k\) in its slack and multiplier updates;
  \(\alpha = 1\) recovers plain ADMM (Boyd et al. §3.4.3, OSQP default).
- Setting: `SolverSettings { over_relaxation: 1.6 }`, validated to lie
  strictly inside \((0, 2)\); exposed in Python as `over_relaxation` and on
  the synthetic example as `--alpha`.
- Termination, adaptive-ρ residual balancing, and every reported residual
  use the true iterates, never the relaxed blend, so `Solved` does not
  depend on \(\alpha\).
- Measured on the smoke matrix (same machine, α=1.0 control run):
  iterations 50→30 (n=100), 170→90 (n=500), 310→170 (n=1000),
  750→260 (n=2000), 1680→660 (n=5000); wall time at n=5000 2.7s → 1.1s.
  Details in [`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md).
- Vector ρ (per-constraint penalties) re-evaluated 2026-07-22 after 2.3/2.4
  landed, with an env-hook prototype scaling one consensus block's penalty
  consistently: equality-row boosts ×3–×1000 change nothing measurable
  (identical smoke-matrix iteration counts), and inequality-row factors are
  instance-dependent with a sign flip (n=1000/2000 improve ~1.6x at
  ×0.1–×0.3; n=5000 worsens 1.6x at the same factors). Stays deferred;
  decision and reopening condition in [`DECISIONS.md`](DECISIONS.md).

### 2. Infeasibility certificates (M2) — implemented

- OSQP-style detection: on infeasible problems the ADMM iterate differences
  converge to certificate directions — dual differences \(\delta y\) to a
  Farkas certificate of primal infeasibility, primal differences
  \(\delta x\) to an unbounded descent ray proving dual infeasibility.
  Detection runs on the termination-check cadence (module
  `certificate.rs`), comparing original-space iterates between consecutive
  checks, so certificates never depend on the equilibration.
- New statuses `PrimalInfeasible` / `DualInfeasible`; the normalized
  (unit \(\infty\)-norm) certificate is attached to
  `Solution.certificate` and is independently auditable with
  `check_primal_certificate` / `check_dual_certificate`, exactly as
  `check_kkt` audits solutions.
- Setting: `SolverSettings { infeasibility_tolerance: 1e-5 }` (`0`
  disables; exposed in Python). Stricter than OSQP's `1e-4` because false
  infeasibility reports would burn trust; problems infeasible by a smaller
  margin fall back to `MaxIterations` with hints. Random feasible-QP
  proptests and a zero-slack boundary-feasible test guard against false
  positives.
- Portfolio layer translates certificates into portfolio vocabulary as the
  leading hint (budget vs sector caps vs bounds; riskless rewarded
  direction for dual certificates) — the niche differentiator general QPs
  skip. `PortfolioSequence` restarts cold after an infeasible date so the
  diverged duals never poison the next warm start; Python raises with the
  same message unless `raise_on_failure=False`, and `SolveResult.certificate`
  carries the arrays.

### 2a. Solution polishing (M2) — implemented

- After a `Solved` termination, the active set is guessed from the final
  iterate (constraint tight within a scaled multiple of the stopping
  tolerance *and* multiplier not pointing the other way), and the
  equality-constrained KKT system of that guess is solved directly through
  the same SMW reduction the iterations use: reduced dimension
  \(r' = k + m_{eq} + m_{active}\); active bounds fold into the base
  diagonal like the ADMM box block, so long-only portfolios with many
  pinned weights stay cheap. Module: `polish.rs`.
- The saddle system is regularized by `polish_regularization` (default
  `1e-6`) and the regularization error removed by
  `polish_refinement_iterations` (default `3`) rounds of iterative
  refinement against the unregularized KKT matrix (OSQP-style). Up to four
  classic active-set passes drop rows with wrong-sign multipliers and add
  rows the candidate violates — needed because an ADMM-accurate iterate
  misclassifies marginal constraints.
- Acceptance is audit-based: every candidate is re-checked with `check_kkt`
  on the original data and adopted only when the **worst** KKT residual
  improves; `Solution::polished` records the outcome. Wrong-sign duals from
  genuinely degenerate active sets (e.g. every variable on a bound making
  the budget row redundant) are therefore rejected, and the ADMM iterate is
  returned unchanged — Ledge never ships uncertified multipliers.
- Certificates are untouched: only `Solved` iterates are polished
  (decision recorded 2026-07-22 with 2.2).
- Measured on the smoke matrix: worst residual ~1e-5/1e-6 → 1e-11..1e-15
  at +7-13% wall time (n=5000: 1.11 s → 1.26 s); polished objectives are
  exactly feasible, removing the ~1e-5-violation bias of raw ADMM
  objectives. Settings exposed in Python as `polish=True` and
  `SolveResult.polished`.

### 3. Exact L1 turnover prox (M2) — implemented

- Extra consensus block \(x - w_0 = z_t\) with a soft-threshold update
  (`L1Term` on `QpProblem`, engine in `workspace.rs`). Adds one more
  \(\rho I\) on the diagonal of the x-system, exactly like the box block;
  **reduced dimension \(r\) unchanged**. An epigraph reformulation would
  add \(2n\) inequality rows instead — that form is used only as the test
  oracle (`tests/l1.rs`, including proptest; Python
  `tests/test_l1.py` against cvxpy+Clarabel).
- The L1 multiplier joins `DualVariables::l1`; `check_kkt` scores it in
  stationarity plus the subgradient conditions (\(|\lambda_i| \le c_i\),
  and \(\lambda_i = \pm c_i\) on the trade sign when trading), so audited
  optimality covers the nonsmooth term. Warm starts carry `l1_dual`.
- Polishing understands the kink: assets inside the no-trade region are
  pinned at the anchor (`Side::Anchor`); trading assets carry the signed
  cost \(\pm c_i\) in the linear term; up to the usual four refinement
  passes flip misclassified trade signs. Recovered multipliers are audited
  before adoption like every polish candidate.
- Ruiz scaling transforms costs and anchor with the variable scales and the
  cost scalar; certificates gain the L1 recession slope
  \(\sum_i c_i |d_i|\) in the dual-certificate objective gap, so a descent
  direction only proves unboundedness after paying the L1 growth.
- Sequences: `Workspace::update_l1_anchor` moves \(w_0\) in scaled space in
  \(O(n)\) without touching the factorization cache;
  `RebalanceStep.previous_weights` drives both the L2 fold-in and the L1
  anchor. Python: `l1_turnover_costs=` (scalar broadcast or per-asset
  array) on all entry points.
- Comparison adapters (1.8 harness) feed external solvers the standard
  epigraph reformulation — \(n\) extra `t` variables with linear cost
  \(c\), \(2n\) inequality rows — in both the dense-Q and lifted
  formulations, mapping the epigraph multipliers back onto the L1
  subgradient dual (\(\lambda_i = y^{up}_i - y^{lo}_i\)) so protocol
  re-verification via `check_kkt` covers the nonsmooth term. Ledge itself
  keeps the prox block; each solver gets the best encoding available to it.
  Measured 2026-07 ([`2026-07-l1`](../benchmarks/results/2026-07-l1/README.md)):
  the epigraph costs external solvers 2–6x while the prox block costs ~8%,
  making Ledge the fastest solver at every smoke-matrix size, cold and
  rolling, on L1 instances.

### 3a. Tracking-error sugar (M2) — implemented

- `PortfolioProblem::with_tracking_benchmark(b)` /
  Python `benchmark_weights=`: the risk term becomes
  \(\tfrac{\lambda}{2}(w-b)^\mathsf{T}\Sigma(w-b)\). Expanding the square
  only shifts the linear cost by \(-\lambda\Sigma b\), computed through the
  factor structure (never forming \(\Sigma\)); the constant
  \(\tfrac{\lambda}{2}b^\mathsf{T}\Sigma b\) is dropped from reported
  objectives, as is conventional.
- Because it is a pure linear-cost shift, `RebalanceStep.benchmark_weights`
  rolls the benchmark date-by-date with cached factorizations intact; the
  sequence keeps the raw covariance to recompute the shift.
- Tests: shift equivalence against manually adjusted returns, exact
  benchmark reproduction under pure tracking, combined
  tracking + L2 + L1 audits (`tests/tracking.rs`,
  `python/tests/test_tracking.py` incl. cvxpy+Clarabel oracle).

### 4. Workspace / factorization cache (M2) — implemented

- `Solver::workspace(&QpProblem) -> Workspace` (module `workspace.rs`) owns
  the Ruiz-equilibrated problem copy and a penalty-keyed LRU cache of
  SMW-reduced factorizations; `Solver::solve` is now a single-use workspace,
  so there is exactly one ADMM engine.
- `workspace.solve(warm_start)` reuses all of it. `update_linear` (new μ /
  \(w_0\)) and `update_equality_rhs` / `update_inequality_rhs` (new budget /
  caps) reapply the frozen scaling as exact transforms in \(O(n)\) without
  invalidating the cache; termination stays on original data.
- Every solve replays the one-shot penalty policy (start at `settings.rho`,
  adapt as usual), so a workspace changes cost, never iterates: the iterate
  path is identical to a fresh `Solver::solve` of the same data. Penalties
  already visited hit the cache, so warm rolling sequences reach a steady
  state of zero refactorizations per step (asserted in
  `tests/workspace.rs`; visible via `Workspace::factorizations`). Carrying
  the previous solve's final ρ over instead was measured and rejected — see
  `DECISIONS.md` (2026-07-21).
- Measured (criterion `rolling_resolve`, one warm rolling step, same
  machine): n=500 1.73 → 1.47 ms, n=1000 13.1 → 12.1 ms, n=2000 54.8 →
  48.3 ms; protocol rerun in `benchmarks/results/2026-07-workspace/`.
  The saved work is the per-step equilibration + \(O(nr^2)\) setup and
  ladder refactorizations; iteration count remains the dominant cost at
  large n (polishing landed with 2.3; vector ρ remains the next
  convergence lever).
- A cache miss still costs a full rebuild: \(B^{-1}\) reweights every entry
  of the reduced Gram matrix when ρ changes, so no low-rank update applies.
- This object is also the public in-process primitive that any external
  service integration should wrap through published APIs.

### 4a. `solve_sequence` / `PortfolioSequence` (M2) — implemented

- `PortfolioProblem::sequence()` (or `sequence_with(&Solver)`) returns a
  `PortfolioSequence` (module `sequence.rs`) owning a `Workspace`;
  `solve_next(&RebalanceStep)` applies one date's data changes and solves,
  warm-started from the previous date's full primal/dual solution
  automatically. `solve_sequence(problem, settings, &[RebalanceStep])` is the
  one-call batch form; Python mirrors it as
  `PortfolioProblem.sequence(**solver_kwargs)` →
  `PortfolioSequence.solve_next(expected_returns=…, previous_weights=…,
  budget=…, equality_rhs=…, inequality_rhs=…)`.
- `RebalanceStep` can only express factorization-preserving updates —
  expected returns, turnover anchor (`previous_weights`; the L2 penalty and
  L1 costs are part of the problem structure and stay fixed), tracking
  benchmark (`benchmark_weights`), budget, and equality / inequality
  right-hand sides. Structural changes (covariance, constraint matrices,
  bounds, turnover penalty) are rejected with an explanatory error: build a
  new problem and a new sequence.
- Steps are atomic: every field is validated before any state changes, so a
  rejected date leaves the sequence exactly as it was and the caller can
  skip it and keep rolling. An unconverged solve (`MaxIterations`) does not
  abort the sequence; its iterate still seeds the next warm start
  (`NumericalFailure` restarts cold).
- The portfolio-level `q = -(μ + penalty·w₀)` fold-in is recomputed by the
  sequence, so users update *returns*, not solver vectors; termination
  stays on the updated original data (workspace rules, Technical notes §4).
- Rolling behavior on the deterministic examples (`examples/sequence.rs`,
  `python/examples/rolling.py`): after the first date, every warm date
  reuses the cached factorization (count stays at 1) and converges in ~10
  iterations vs ~20 cold.

### 5. Sparse `F` (M3, demand-triggered)

- Trigger: country / industry dummy factors with ~2–5 nonzeros per row.
- Abstract `MatVecOp` (dense / CSR). **Do not build before real demand.**

### 6. Constraint template builders (M3) — implemented

- Pure sugar over existing machinery — no solver changes. Group templates
  (`with_industry_neutrality`, deriving targets from the tracking
  benchmark; `with_group_targets` with explicit targets) emit one
  indicator equality row per group. `with_style_bounds` emits the finite
  sides of each band as inequality rows (upper first, then negated lower),
  collapsing an exact band (`lower == upper`) to a single equality row.
  `with_concentration_limit` (`|w_i| <= cap`) and `with_short_limit`
  (`w_i >= -limit`; `0` = long-only) only tighten the boxes, so the
  reduced dimension `r = k + m` never grows for them.
- Templates **append** to the user constraint blocks (`with_equalities` /
  `with_inequalities` replace them, so call those first). Appended rows are
  ordinary user rows afterwards: sequences roll industry/style targets
  through `RebalanceStep::equality_rhs` / `inequality_rhs` with cached
  factorizations intact (asserted in `tests/templates.rs`).
- Templates validate eagerly (empty groups, out-of-range ids, crossing or
  doubly-infinite bands, caps contradicting existing bounds) with the new
  `PortfolioError::Template` so mistakes fail at build time, not as
  infeasible solves. A total short-budget cap
  (`sum_i max(-w_i, 0) <= S`) is deliberately excluded: it needs a
  long/short variable split the QP form does not model.
- Python: constructor kwargs on `PortfolioProblem` and
  `solve_mean_variance_factor` (`industry_ids`, `industry_targets`,
  `style_matrix` / `style_lower` / `style_upper`, `max_weight`,
  `max_short`); equivalence against hand-built matrices is tested in
  `python/tests/test_templates.py`.

### 7. Problem / solution serialization (M3) — implemented

- Non-default `serde` cargo feature on `ledge-core` (re-exported by
  `ledge`); the default build keeps zero serialization dependencies.
  Serializable: `QpProblem`, `PortfolioProblem`, `SolverSettings`,
  `WarmStart`, and `Solution` with every dual block, residuals,
  diagnostics, and infeasibility certificates — a bug report can carry
  exactly what the solver saw and returned.
- Deserialization cannot bypass validation: `Matrix` rebuilds through
  `Matrix::new`, `PortfolioProblem` replays its builder methods, and
  `QpProblem` is re-validated at solve entry. Bound vectors travel as
  `Option<f64>` per entry (`null` = unbounded side) because JSON has no
  representation for infinities; any self-describing binary serde format
  works unchanged (`postcard` is the tested stand-in).
- Round-trips are bit-exact — a JSON-replayed problem produces the
  identical iterate path (`tests/serde.rs`; JSON parsing needs
  `serde_json`'s `float_roundtrip` feature, documented in the guides).
  Solutions from `NumericalFailure` contain non-finite iterates and only
  round-trip through binary formats.
- Python: `PortfolioProblem.to_json()` / `PortfolioProblem.from_json()`
  and `SolveResult.to_json()` ship in the default wheel, so issue
  reporters attach two JSON strings without installing anything extra.

### 8. Multi-thread batch over the account axis (M3) — implemented

- `solve_batch(accounts, settings) -> Vec<AccountResult>` (module
  `batch.rs`): each [`BatchAccount`] is a `PortfolioProblem` plus its
  ordered `RebalanceStep`s, solved as its own `PortfolioSequence` —
  per-account equilibration, factorization cache, and chained warm starts,
  exactly the numerical work of a `solve_sequence` loop. Accounts share no
  state, so results are **bit-identical to the serial loop regardless of
  feature or thread count** (asserted in `tests/batch.rs`, which CI runs
  with and without the feature).
- Threading is the non-default `rayon` cargo feature on `ledge-core` /
  `ledge` (`rayon::par_iter` over accounts; `RAYON_NUM_THREADS` or a
  caller-installed pool control the width). The default build keeps zero
  new dependencies and runs the same API serially.
- `chain_previous_weights: bool` implements the backtest convention: after
  a `Solved` date the turnover anchor moves to that date's solved weights
  (what the account actually holds); non-`Solved` dates leave the anchor
  (the account did not trade); an explicit `previous_weights` in a step
  wins. Requires a turnover term, checked before any solve.
- Failures stay per-account: the return is one
  `Result<Vec<Solution>, PortfolioError>` per account in input order, so
  one account's bad feed never discards the other 499 accounts' results.
- Python: `ledge.solve_batch(problems, steps, chain_previous_weights=...,
  **solver_kwargs)` with per-date step dicts mirroring `solve_next`
  kwargs; the GIL is released for the whole batch; errors name the
  account (and step) index. The wheel enables the feature.
- Published throughput (`cargo run -p ledge-portfolio --release --features rayon
  --example batch`): 1 model × 500 accounts × 250 dates, n=200, k=15,
  L2+L1 turnover with chained anchors — 12.9 s wall on 4 vCPUs
  (9.7k account-dates/s), 3.96x over the serial build, all 125k solves
  `Solved`; artifacts in
  [`benchmarks/results/2026-07-batch/`](../benchmarks/results/2026-07-batch/README.md).

---

## Explicitly not on the roadmap

- Mixed-integer programming (MIP / MIQP)
- GPU or distributed solvers
- Unverified cross-solver marketing claims
- Becoming a general-purpose LP/QP modeling stack
- Deliberately slowing the open-source edition

---

## Related docs

- Product plan: [`PLAN.md`](PLAN.md)
- Open-core boundary and release runbook: [`OPEN_CORE.md`](OPEN_CORE.md)
- Decisions: [`DECISIONS.md`](DECISIONS.md)
- Smoke timings: [`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md)
- Algorithm notes: [`algorithm.md`](algorithm.md)
- Technical note (1.9): [`factor_structure_note.md`](factor_structure_note.md)
- Docs site source (3.5): `docs/book/` (`mdbook build docs/book`)
- Benchmark protocol: [`../benchmarks/README.md`](../benchmarks/README.md)
