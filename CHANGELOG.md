# Changelog

All notable changes to Ledge are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use
[Semantic Versioning](https://semver.org/) once the public API stabilizes.

Version links below point to the corresponding GitHub tags or comparisons.

## [Unreleased]

### Fixed

- Replaced unsupported operator-name math macros with GitHub/KaTeX-compatible
  roman notation across the README and mathematical documentation.
- Updated package, documentation-site, and roadmap copy to reflect the
  published `0.2.0` artifacts.

### Security

- Pinned third-party GitHub Actions to immutable commits, restricted default
  workflow permissions to read-only, verified the downloaded mdBook archive,
  and moved workflow-dispatch input handling out of interpolated shell source.
- Added Dependabot coverage for Cargo, Python, and GitHub Actions plus a
  repository security-audit report.

### Documentation

- Reworked the README quick start, navigation, feature overview, documentation
  map, Rust example, platform guidance, and current roadmap.

## [0.2.0] - 2026-07-22

### Fixed

- Split OSQP/Clarabel comparison-adapter tests into an independent CI job so
  adapter compile/test failures remain visible when fmt/clippy fail first on
  the main Rust job.
- Removed three unused legacy PNG duplicates from `docs/assets`; the linked
  SVG/GIF assets remain the maintained, reproducible versions.
- Corrected release packaging commands: `maturin sdist` does not accept
  `--locked`, and Cargo cannot assemble the dependent portfolio `.crate`
  before the exact `ledge-core` version reaches the crates.io index.
- Corrected the clean-root runbook to compare Git tree IDs and tag the new
  public root SHA; a clean-root commit never shares the private gate SHA.

### Added

- Apache-2.0 release gate for a clean-root public history: the public
  technical plan/roadmap/decision log no longer carries commercial strategy;
  `OPEN_CORE.md` records the selected clean-history and package-name
  decisions; `check_open_core.sh --release` is a strict gate for license,
  package metadata, synchronized release versions, the occupied Rust package
  name, stale private/pre-release wording, crate publish flags,
  packaged license/notice files, private-surface paths, strategy details,
  release automation, and README assets. `PUBLIC_RELEASE_CHECKLIST.md` adds
  copyable sign-offs for rights, history scanning, package inspection,
  visibility, registries, and post-release controls. A manual `release.yml`
  builds abi3-py39 wheels for Linux x86-64/aarch64, macOS universal2, and
  Windows x86-64 and can publish through PyPI Trusted Publishing after
  maintainer configuration and approval.
- Reproducible `cargo-about` attribution report for Rust dependencies compiled
  into Python wheels; project and third-party legal files are included in
  wheel/sdist metadata and checked by the strict release gate.
- Reproducible visual tour: a real seeded terminal-capture GIF, rolling
  rebalance/cache diagram, and an L1 comparison chart parsed from the
  committed report. `scripts/generate_demo_assets.py` regenerates/checks the
  assets used by the README and mdBook.
- mdBook documentation site skeleton (roadmap 3.5) under `docs/book/`:
  tutorial (install, first solve in Python and Rust, rolling, batch),
  guides (constraints/templates, turnover and tracking, trust machinery,
  serialization, the cvxpy migration guide included verbatim), and
  reference (tuning, API surface, benchmark evidence, design map). CI
  builds the book on every push (`docs` job, artifact only); public GitHub
  Pages deployment is a separate manual-only workflow
  (`docs-deploy.yml`) gated on the public-release decision (roadmap 1.6).

- Technical note "Why factor structure is worth exploiting"
  (`docs/factor_structure_note.md`, roadmap 1.9): the measured dense-Q vs
  lifted gap (1–2 orders of magnitude inside the same external solver),
  what the SMW reduction pays per iteration, what a native factor engine
  adds beyond the lifted reformulation (warm starts, factorization reuse,
  prox-block L1, portfolio-vocabulary duals), and the honest large-n
  cold-start limits — every number cited from the published protocol
  reports.

- Comparison harness: `-l1` instance variants (`--l1-bps`, default 10 bps,
  `0` disables) add proportional turnover costs anchored at the shared
  primal start to every smoke-matrix instance. Ledge keeps its prox block;
  external solvers receive the documented epigraph reformulation from
  `convert.rs`. Fourth protocol report published in
  `benchmarks/results/2026-07-l1/`: first re-run with polish-on defaults
  and the first cross-solver L1 turnover measurements.

- Multi-thread batch over the account axis (roadmap 3.2): new
  `solve_batch(&[BatchAccount], settings)` runs one rolling
  `PortfolioSequence` per account — per-account equilibration,
  factorization cache, and chained warm starts — and distributes accounts
  over rayon's thread pool behind a new non-default `rayon` cargo feature
  on `ledge-core` / `ledge-portfolio` (without the feature the same function runs
  serially with bit-identical results; the default build gains no
  dependencies). `BatchAccount.chain_previous_weights` implements the
  backtest convention: each date after a `Solved` one anchors the turnover
  terms at that date's solved weights unless the step provides an anchor
  explicitly. Failures stay per account (`Vec<Result<Vec<Solution>,
  PortfolioError>>` in input order). Python: `ledge.solve_batch(problems,
  steps, chain_previous_weights=..., **solver_kwargs)` with per-date step
  dicts mirroring `PortfolioSequence.solve_next` kwargs, GIL released for
  the whole batch, errors naming the account/step index; the wheel enables
  the feature. Published throughput (`--example batch`, defaults): 1 model
  x 500 accounts x 250 dates (n=200, k=15, L2+L1 turnover) in 12.9 s on 4
  vCPUs — 9.7k account-dates/s, 3.96x over the serial build, all 125k
  solves `Solved` (`benchmarks/results/2026-07-batch/`).

- Constraint template builders (roadmap 3.1): portfolio vocabulary compiled
  onto the existing constraint machinery, with eager validation and no
  solver changes. `PortfolioProblem::with_industry_neutrality(&ids)`
  derives per-industry targets from the tracking benchmark and
  `with_group_targets(&ids, &targets)` takes them explicitly (one equality
  row per group); `with_style_bounds(&exposures, &lower, &upper)` appends
  the finite sides of each style band as inequality rows (exact bands
  become equality rows); `with_concentration_limit(cap)` and
  `with_short_limit(limit)` tighten the box bounds without adding rows.
  Templates append to the user constraint blocks, so they stack with
  hand-built rows and with each other, and rolling sequences move template
  targets through the existing `RebalanceStep` RHS updates with cached
  factorizations intact. New `PortfolioError::Template` reports empty
  groups, out-of-range ids, crossing or doubly-infinite bands, and caps
  that contradict existing bounds at build time. Python: `industry_ids=`,
  `industry_targets=`, `style_matrix=` / `style_lower=` / `style_upper=`,
  `max_weight=`, and `max_short=` kwargs on `PortfolioProblem` and
  `solve_mean_variance_factor`.

- Problem / solution serialization for bug reproduction (roadmap 3.3):
  new non-default `serde` cargo feature on `ledge-core` / `ledge-portfolio` deriving
  `Serialize` / `Deserialize` for `QpProblem`, `PortfolioProblem`,
  `SolverSettings`, `WarmStart`, and `Solution` — including every dual
  block, residuals, diagnostics, and infeasibility certificates — with the
  format left to the caller (JSON and `postcard` are exercised in tests).
  Deserialization re-runs construction validation (`Matrix` rebuilds
  through `Matrix::new`; `PortfolioProblem` replays its builder methods),
  unbounded box sides serialize as `null` because JSON cannot represent
  infinities, and round-trips replay the identical iterate path (use
  `serde_json` with its `float_roundtrip` feature). Python (always
  enabled in the wheel): `PortfolioProblem.to_json()` /
  `PortfolioProblem.from_json()` and `SolveResult.to_json()`, so an issue
  can attach the exact problem and the exact solver output as two JSON
  strings.

- cvxpy migration guide (`docs/cvxpy_migration.md`, M2 exit criterion):
  maps the common cvxpy portfolio patterns onto the Ledge API — objective
  conventions (the 1/2 factor, maximize vs minimize), a constraint table
  (floors via negation, ranges via stacking), L2 + exact L1 turnover,
  tracking benchmarks, `cp.Parameter` loops → `PortfolioSequence`, and
  status/failure semantics including certificates. Every code pair in the
  guide is executed and cross-checked against cvxpy+Clarabel in CI by the
  new `python/tests/test_migration_guide.py` (independent NumPy objective,
  gold-test tolerances), so the guide cannot silently drift from the API.
- L1 epigraph conversion in the comparison adapters: `ConvertedQp` now
  encodes problems with an `L1Term` for external solvers via the standard
  epigraph reformulation (`n` extra `t` variables carrying the linear
  costs, `2n` inequality rows) in both dense-Q and lifted formulations,
  instead of rejecting them. Epigraph multipliers map back onto Ledge's L1
  subgradient dual (`lambda_i = y_upper_i - y_lower_i`) so protocol
  re-verification with `check_kkt` audits the nonsmooth term; rolling
  linear-cost updates preserve the `t`-block costs and lifted primal
  starts set `t_i = |x_i - a_i|`. New feature-gated cross-check tests run
  every adapter/formulation pair on L1 instances.

- Exact L1 turnover / proportional transaction costs (roadmap 2.1):
  `QpProblem` gains an optional `L1Term { costs, anchor }` adding
  `sum_i costs[i] * |x[i] - anchor[i]|` to the objective, handled by a
  dedicated ADMM soft-threshold consensus block — the SMW-reduced
  dimension is unchanged (an epigraph reformulation would add `2n`
  constraint rows; it survives only as the test oracle). The L1
  multiplier is a first-class dual (`DualVariables::l1`): `check_kkt`
  audits the subgradient interval and signed-cost pinning, warm starts
  carry `WarmStart::l1_dual`, Ruiz scaling transforms costs and anchor,
  dual-infeasibility certificates account for the L1 recession slope, and
  polishing pins no-trade assets at the anchor (with sign-flip refinement
  and audited multiplier recovery). High-level API:
  `PortfolioProblem::with_l1_turnover(previous_weights, costs)` — may be
  combined with the quadratic penalty (shared anchor) — and
  `Workspace::update_l1_anchor` / `RebalanceStep.previous_weights` roll
  the anchor through sequences without refactorizing. Python:
  `l1_turnover_costs=` (scalar broadcast or per-asset array) on
  `PortfolioProblem`, `solve_mean_variance_factor`, and sequences. The
  no-trade region is machine-exact; validated against epigraph
  reformulations (including proptest) in Rust and cvxpy+Clarabel gold
  tests in Python.

- Tracking-error objective (roadmap 2.6):
  `PortfolioProblem::with_tracking_benchmark(benchmark_weights)` switches
  the risk term to active risk `risk_aversion / 2 * (w-b)' Sigma (w-b)`.
  Implemented as a pure linear-cost shift `-risk_aversion * Sigma @ b`
  computed through the factor structure — same QP underneath, no solver
  changes; the constant `b' Sigma b` term is dropped from reported
  objectives as is conventional. `RebalanceStep.benchmark_weights` (Python
  `solve_next(benchmark_weights=...)`) rolls the benchmark date-by-date
  with cached factorizations intact. Python: `benchmark_weights=` on all
  entry points.

- Rolling backtest example with published warm-start numbers (roadmap
  2.7): `docs/examples/rolling_backtest.py` runs a seeded
  public-data-style momentum backtest (300 assets, market + sector +
  style factors, 24 monthly rebalances, 10 bps proportional costs,
  tracking benchmark) twice — fresh solves per date vs one
  `PortfolioProblem.sequence()` — and prints the comparison.
  Measured (release build): warm dates 79 -> 37 iterations (2.1x) and
  1.71 -> 0.91 ms (1.9x) per date, reduced factorizations 24 -> 2;
  published in `docs/examples/README.md`.

- Solution polishing, on by default (roadmap 2.3): after a `Solved`
  termination the solver guesses the active set from the final iterate
  (tight within a scaled tolerance *and* multiplier not pointing the other
  way), solves the equality-constrained KKT system of that guess directly
  through the same SMW reduction the iterations use (active bounds fold
  into the diagonal, so reduced dimension stays
  `factors + equalities + active inequalities`), regularizes with
  `polish_regularization = 1e-6`, removes the regularization error with
  `polish_refinement_iterations = 3` refinement rounds, and runs up to four
  classic active-set passes (drop wrong-sign rows, add violated rows). The
  candidate is adopted **only** when its independently audited
  (`check_kkt`) worst KKT residual improves, so enabling polish never
  degrades a solution; degenerate active sets with non-unique multiplier
  splits fall back to the ADMM iterate instead of shipping wrong-sign
  duals. Smoke-matrix worst residuals drop from ~1e-5/1e-6 to
  1e-11..1e-15 for ~7-13% extra wall time, and polished objectives are
  exactly feasible. New `SolverSettings.polish` /
  `polish_regularization` / `polish_refinement_iterations`,
  `Solution.polished`, Python `polish=True` kwarg (solve, sequence, and
  the free function), `SolveResult.polished`, and `--polish` on the
  synthetic example. Certificates and failure diagnostics are never
  polished (only `Solved` iterates are).

- Infeasibility certificates (roadmap 2.2): solves on problems with
  contradictory constraints now stop with the new
  `SolveStatus::PrimalInfeasible` (Farkas certificate) and unbounded
  problems with `SolveStatus::DualInfeasible` (descent-ray certificate)
  instead of burning the full iteration budget. Detection is OSQP-style —
  ADMM iterate differences converge to certificate directions — but runs on
  **original-space** iterates at the termination-check cadence, so reported
  infeasibility never depends on the equilibration. Certificates are
  normalized to unit infinity norm, attached to `Solution.certificate`, and
  independently auditable via the new `check_primal_certificate` /
  `check_dual_certificate` (the `check_kkt` policy extended to failures).
  New setting `SolverSettings.infeasibility_tolerance = 1e-5` (`0`
  disables; Python `infeasibility_tolerance=...`). The portfolio layer
  prepends a hint naming the conflicting constraints in portfolio
  vocabulary (budget, inequality caps, bounds; riskless rewarded direction
  for unbounded objectives), and `PortfolioSequence` restarts cold after an
  infeasible date so its diverged duals never seed the next warm start.
  Python exposes `SolveResult.certificate`
  (`InfeasibilityCertificate` with `kind`, Farkas weight arrays, or the
  descent `direction`); `raise_on_failure=True` raises with the semantic
  hints included.

- Rolling sequence API (roadmap 2.5): `PortfolioProblem::sequence()` (or
  `sequence_with(&Solver)`) returns a `PortfolioSequence` whose
  `solve_next(&RebalanceStep)` applies one date's data changes — new
  expected returns, turnover anchor, budget, or constraint right-hand
  sides — and solves warm-started from the previous date's full
  primal/dual solution, on top of the workspace factorization cache.
  `solve_sequence(problem, settings, &[RebalanceStep])` is the one-call
  batch form. Steps are atomic (a rejected date leaves the sequence
  unchanged) and can only express factorization-preserving updates;
  structural changes are rejected with an explanatory error. Python
  mirrors the API as `PortfolioProblem.sequence(**solver_kwargs)` /
  `PortfolioSequence.solve_next(expected_returns=..., previous_weights=...,
  budget=..., equality_rhs=..., inequality_rhs=...)`, releasing the GIL
  while solving. New rolling examples: `cargo run -p ledge-portfolio --release
  --example sequence` and `python python/examples/rolling.py` (ten dates,
  one reduced factorization total, warm dates converge in about half the
  cold iteration count).

- ADMM over-relaxation, on by default: every consensus block (equality,
  inequality, box) sees the blend `alpha * Ax + (1 - alpha) * z_prev` in
  its slack and multiplier updates. New
  `SolverSettings.over_relaxation = 1.6` (validated to lie strictly inside
  `(0, 2)`; `1.0` recovers plain ADMM), exposed in Python as
  `solve(..., over_relaxation=...)` and on the synthetic example as
  `--alpha`. Termination and every reported residual still use the true
  iterates on the original data, so `Solved` does not depend on the
  relaxation. Smoke-matrix iteration counts drop 1.7-2.9x
  (`n=5000, k=100`: 1680 -> 660 iterations, ~2.7s -> ~1.1s); see
  `docs/SMOKE_TIMINGS.md`. A protocol re-run of the cross-solver
  comparison is published in `benchmarks/results/2026-07-over-relaxation/`:
  the cold-start gap to lifted Clarabel at large n narrowed 2-3x (n=5000:
  5.9x -> 2.3x) and Ledge now leads or ties rolling re-solves up to
  n=1000; factorization reuse (roadmap 2.4) is now the binding lever at
  large n.
- Cross-solver comparison adapters (`benchmarks/adapters`, workspace crate
  `ledge-bench-adapters`): OSQP and Clarabel behind the non-default `osqp` /
  `clarabel` cargo features, so the default build has no extra native
  dependency. Both a naive dense-`Q` conversion and a factor-lifted sparse
  reformulation are provided and documented; adapters keep native
  termination statuses verbatim, map duals back into Ledge's convention,
  and every returned point is re-verified with the independent `check_kkt`.
  Feature-gated tests cross-check both solvers against Ledge (objective
  within `1e-6`).
- Protocol-compliant comparison harness (`compare` binary): phase-split
  setup / cold / rolling-re-solve timing with uniform wall clocks, shared
  instances and primal starts, deterministic perturbed expected-return
  sequences, ten-plus repeats, and full raw-sample CSV plus aggregated
  Markdown output. First published report:
  `benchmarks/results/2026-07/`.
- Automatic data scaling: Ruiz equilibration plus OSQP-style cost
  normalization, on by default via `SolverSettings.scaling_iterations = 10`
  (`0` disables; Python `solve(..., scaling_iterations=...)`). Scaling
  preserves the factor structure (`Q` is never formed) and only changes the
  space ADMM iterates in: warm starts are scaled on entry, and termination,
  `check_kkt`, and every reported residual, objective, and iterate remain in
  terms of the original data. With this change the full synthetic smoke
  matrix through `n=5000, k=100` reaches `Solved` under default settings
  (previously `n=2000, k=50` stopped at `MaxIterations`); see
  `docs/SMOKE_TIMINGS.md`.
- Ill-conditioned integration suite (`crates/ledge-core/tests/scaling.rs`):
  skewed variable-unit instances where scaling off fails and scaling on
  solves, warm-start round-trips through scaling, and solution agreement
  with the unscaled path on well-conditioned data.
- Failure diagnostics: solves that stop at `MaxIterations` or
  `NumericalFailure` now attach `Solution.diagnostics`
  (`ConvergenceDiagnostics`) with the effective stopping tolerances,
  coefficient-magnitude spread, penalty-limit state, and ordered heuristic
  hints. The Python `SolveResult` exposes them as `convergence_hints`, and
  the `raise_on_failure` error message includes them.
- Property-based tests (`proptest`): random feasible factor QPs must return
  `Solved` with independently checked KKT residuals, and warm-started
  re-solves must agree with cold solves without extra iterations.
- Criterion microbenchmarks for the reduced factorization, one x-update, and
  end-to-end synthetic solves, behind the new non-default `bench-internals`
  feature (`cargo bench -p ledge-core --features bench-internals`).
- Python gold-standard tests against cvxpy + Clarabel on 20 random
  instances, installed via the new `test` extra
  (`pip install -e "python/[test]"`); CI runs them.

### Fixed

- The feature-gated OSQP/Clarabel adapters failed to compile (missing
  `DualVariables::l1` field) after the roadmap 2.1 merge, and a
  `clippy 1.97` lint (`manual_assert_eq`) broke the CI lint step before
  the adapter tests could report it. Both fixed; CI is green again on the
  full `--all-features` build.

### Changed

- `check_kkt` now scores box-bound multipliers continuously: the positive
  part of a multiplier is charged `|multiplier| * distance-to-upper-bound`
  as complementarity (symmetrically for the negative part and the lower
  bound) instead of counting fully against the dual residual whenever the
  variable sat outside a fixed `1e-7` activity window. This makes the
  checker fair to interior-point solutions, whose multipliers decay
  smoothly near bounds. To keep `Solved` as strong as before, the solver's
  termination test now also requires the complementarity residual to meet
  `max(primal_tolerance, dual_tolerance)`.
- Convergence hints for badly scaled data now point at the
  `scaling_iterations` setting instead of announcing planned equilibration.
### Documentation

- Recorded the measured vector-ρ re-evaluation in `docs/DECISIONS.md`
  (2026-07-22): no static per-block penalty factor wins across the smoke
  matrix, so vector ρ stays deferred with an explicit reopening condition;
  roadmap 1.5 is closed for M2.
- Added `docs/PLAN.md` — vertical product plan (niche, expectations, stack,
  open-core boundary, and technical acceptance gates).
- Rewrote `docs/ROADMAP.md` as executable milestones M0–M4 with exit criteria
  and technical notes (scaling, certificates, L1 prox, workspace).
- Added `docs/DECISIONS.md` ADR log for plan adoption.
- Added GitHub issue templates for bugs, performance/convergence, and
  redacted problem instances.
- Standardized documentation in English (README, `docs/algorithm.md`,
  contributor and security guides).
- Added `CODE_OF_CONDUCT.md` and `NOTICE`.
- Fixed README diagram SVG encoding (Sigma / Omega / arrows).
- Removed pre-publish checklist docs; kept product roadmap and algorithm notes.
- Clarified scope and non-goals without historical repository-name framing.
- Added `SECURITY.md`, `docs/SMOKE_TIMINGS.md`, README diagrams, and CI.

## [0.1.0] - 2026-07-14

### Added

- Factor-structured convex QP kernel using an SMW reduced system.
- Equality, upper-inequality, and box constraints.
- Independent primal, dual, and complementarity KKT residual checks.
- Residual-balancing adaptive ADMM penalty.
- High-level mean-variance `PortfolioProblem` Rust API.
- L2 turnover control and primal/dual warm starts for rebalancing.
- Installable PyO3/maturin Python package with NumPy inputs.
- Deterministic single-solve and rolling-rebalance examples.
- Reproducible synthetic problem generator and solver-neutral benchmark hook.

### Known limitations

- No infeasibility certificates, polishing, automatic scaling, or exact L1
  turnover prox.
- Python distributions currently build from source; no wheels are published.
- Large synthetic instances (e.g. n=2000, k=50 under defaults) may hit
  `MaxIterations`.
- Public APIs are alpha and may change before 1.0.

[Unreleased]: https://github.com/Jiangki/ledge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jiangki/ledge/releases/tag/v0.2.0
[0.1.0]: https://github.com/Jiangki/ledge/commits/main
