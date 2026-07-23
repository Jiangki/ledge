# Decisions

Public architecture, numerical, API, evidence, and open-core decision log.
Commercial pricing, customer, revenue, and pilot strategy is maintained
outside this repository. Keep each entry concise: date, decision, why, and
alternatives considered.

---

## 2026-07-20 — Focus on factor-model portfolio rebalancing

**Decision:** Ledge targets a **factor-model portfolio rebalancing engine**,
not a general-purpose QP modeling stack. At the time of this decision the
repository remained private while the whole working tree was prepared as the
public core; the release-gate decision is recorded below.

**Why:**

- SMW reduction and rolling warm starts match “many assets, few factors,
  repeated rebalances.”
- Trust (scaling, certificates, polishing, packaging, fair comparisons) must
  precede broad distribution.
- A narrow scope makes the test matrix, documentation, and support envelope
  honest and maintainable.

**Alternatives considered:**

1. General QP performance race against OSQP / Clarabel — rejected (poor
   solo ROI; strong academic competitors).
2. End-to-end wealth-management platform — rejected (data, compliance, and
   operational scope are outside the solver).
3. A closed correctness-only binary — rejected; adoption and trust require an
   installable core others can audit.

**Consequences:**

- [`PLAN.md`](PLAN.md) is the product north star.
- [`ROADMAP.md`](ROADMAP.md) follows milestones M0–M4 with hard exit criteria.
- When public, the open edition must remain a complete, correct
  single-machine solver; never intentionally slowed.

---

## 2026-07-20 — Keep complete single-machine workflows in the public core

**Decision:** The public repository owns everything needed for trustworthy
single solves **and** in-process workflows: scaling, certificates, polishing,
exact L1 prox, factorization reuse, rolling sequences, and multi-thread
account batch. Persistent deployment/orchestration and customer-specific
operations may be built only in a separate private repository that consumes
published APIs.

**Why:** Numerical correctness and ordinary in-process composition must not
be withheld. A separate repository makes the boundary reviewable and forces
any extension to use the same API available to everyone.

**Alternatives considered:** Put scaling/L1 or the small rayon account loop
outside the core — rejected. Deliberately slow the public solver — rejected.

---

## 2026-07-21 — Comparison adapters: separate crate, dual formulations

**Decision:** OSQP/Clarabel comparison adapters live in a dedicated
workspace crate (`benchmarks/adapters`, `ledge-bench-adapters`) behind
non-default `osqp` / `clarabel` features, and every published comparison
feeds external solvers **two** encodings of the same instance: naive
dense-`Q` and a factor-lifted sparse reformulation.

**Why:**

- Keeping native solver dependencies out of `ledge-core` (even optional)
  preserves the zero-native-deps default build and simple wheels.
- Only comparing against dense-`Q` would flatter Ledge dishonestly: a
  competent user hands a general sparse solver the lifted form. Publishing
  both makes the "factor structure is worth exploiting" claim measurable
  (dense vs lifted gap) without hiding the strongest baseline.
- Protocol conformance (native statuses verbatim, independent `check_kkt`
  re-verification, phase-split timing, ≥10 repeats with raw samples) is
  implemented once in the harness rather than promised in prose.

**Alternatives considered:** optional features on `ledge-core` (rejected:
pollutes the core dependency tree); benchmarking only the lifted form
(rejected: loses the evidence for the structure-exploitation narrative);
scripting comparisons via cvxpy in Python (rejected: measures Python
overhead, not solver cores).

**Consequences:** cross-solver claims in README/docs must cite
`benchmarks/results/YYYY-MM/`; the smoke matrix stays the shared instance
set; Clarabel's lack of warm starts is recorded, not worked around.

---

## 2026-07-21 — Over-relaxation default on; vector ρ deferred

**Decision:** Ship ADMM over-relaxation enabled by default
(`over_relaxation = 1.6`, valid range the open interval `(0, 2)`), applied
uniformly to all three consensus blocks. Defer vector ρ (per-constraint
penalties) until polishing (2.3) and factorization reuse (2.4) show whether
iteration count is still the binding constraint at large n.

**Why:**

- The 1.8 comparison identified iteration growth with n (50 → 1680) as the
  main gap vs lifted Clarabel at large n. Over-relaxation is the cheapest
  standard remedy: no extra factorization, no new state, one blend per
  update.
- Measured on the smoke matrix it cuts iterations 1.7–2.9x and n=5000 wall
  time from ~2.7s to ~1.1s, with unchanged termination checks on original
  data — a pure win under the default-settings trust policy.
- α=1.6 follows the OSQP default and Boyd's recommended [1.5, 1.8] range;
  no per-instance tuning was done, keeping the honest-benchmark story.

**Alternatives considered:** off-by-default (rejected: trust milestone wants
best defaults, and termination is α-independent); tuning α per instance
(rejected: overfits the smoke matrix); implementing vector ρ in the same
change (rejected: refactors the SMW column scaling and adaptive-ρ policy —
measure whether it is still needed after M2 items land).

**Consequences:** default iteration counts and timings change everywhere;
`docs/SMOKE_TIMINGS.md` records the α=1.0 control run. Published comparisons
must state the α in force (the 2026-07 report predates this change and used
α=1.0).

---

## 2026-07-21 — Workspace: frozen scaling, ρ-keyed factorization cache

**Decision:** Roadmap 2.4 ships as `Solver::workspace(&QpProblem) ->
Workspace` (`workspace.rs`): Ruiz equilibration is computed once;
`update_linear` / `update_equality_rhs` / `update_inequality_rhs` reapply
the *frozen* scalings `E, D, c` to new data as exact transforms; SMW-reduced
factorizations live in a small LRU cache keyed by the exact penalty. Every
solve replays the one-shot penalty policy — start at `settings.rho`, adapt
as usual — so a workspace changes **cost, never iterates**. The one-shot
`Solver::solve` now runs through a fresh single-use workspace, so exactly
one ADMM engine exists.

**Why:**

- The 2026-07 comparisons flagged per-solve setup (equilibration plus the
  `O(nr²)` factorization) as pure overhead on rolling workloads — the
  workload class `PLAN.md` prioritizes.
- Freezing the scaling keeps updates `O(n)` and preserves the auditability
  rule unchanged: the scaled problem stays an exact image of the updated
  original data; only normalization quality could drift, and only if the new
  cost differs wildly from the base.
- The adaptive ladder is multiplicative from a fixed start, so revisited
  penalties are bit-identical and the cache absorbs every refactorization
  after the first solve (asserted in `tests/workspace.rs`).

**Alternatives considered:**

1. Persist the previous solve's final ρ as the next start — implemented
   first, then rejected on measurement: on the smoke matrix rolling
   workloads it *increased* warm-step iteration counts (n=500: 90–120 vs
   50–90; n=5000: up to +25% per step), because the balanced end-of-solve ρ
   is not the best start for the next warm-started solve under this
   residual-balancing rule. Replay + cache keeps the exact one-shot path at
   the same amortized factorization cost, and keeps workspace vs one-shot
   results bit-comparable.
2. Rank updates to survive ρ changes — rejected for now: `B⁻¹` reweights
   every entry of the reduced Gram matrix, so no low-rank shortcut exists in
   this formulation; noted in `algorithm.md` §6.
3. Recomputing the cost scalar per update — rejected: `c` multiplies the
   scaled covariance columns, so it would invalidate the factorizations it
   exists to protect.
4. Bounds / constraint-matrix updates — deferred: matrix changes genuinely
   change structure; bounds updates are cheap but wait for a concrete use
   case (2.5 `solve_sequence` or user demand).

**Consequences:** the Ledge comparison adapter now charges equilibration +
first factorization to the *setup* phase (same semantics OSQP already had),
so setup/cold splits are not directly comparable with pre-2.4 reports;
`Solution.solve_time` on workspace solves covers iteration only;
`solve_sequence` (2.5) and any future Pro persistent service wrap this
object rather than growing a second engine.

---

## 2026-07-22 — Sequence API: portfolio-level steps, atomic updates

**Decision:** Roadmap 2.5 ships as a portfolio-level rolling API:
`PortfolioProblem::sequence()` → `PortfolioSequence::solve_next(&RebalanceStep)`
plus the one-call `solve_sequence` (module `sequence.rs`), mirrored in Python
as `PortfolioProblem.sequence()` / `PortfolioSequence.solve_next(...)`. A
`RebalanceStep` can only express factorization-preserving updates (expected
returns, turnover anchor, budget, equality/inequality right-hand sides); every
field is validated before any state changes, so rejected steps are atomic; the
sequence chains full primal/dual warm starts internally and an unconverged
(`MaxIterations`) date does not abort the sequence.

**Why:**

- Users think in μ / w₀ / budget, not in the folded QP vector
  `q = -(μ + penalty·w₀)`. Folding belongs in one place; exposing
  `Workspace::update_linear` directly to Python would push that arithmetic
  onto every caller.
- The turnover penalty sits on the diagonal of `Q`, so changing it would
  invalidate every cached factorization — rejecting it in the step type makes
  the cost model visible in the API instead of silently rebuilding.
- Atomic steps let production loops drop one bad date (bad feed, impossible
  budget) and keep rolling — the failure mode interviews keep mentioning.
- Warm-start chaining inside the object removes the most error-prone
  boilerplate (forgetting duals, warm-starting after a numerical failure);
  `NumericalFailure` restarts cold because its iterate is non-finite.

**Alternatives considered:** exposing `Workspace` to Python 1:1 (rejected:
q-space, not user-space; keeps GIL-friendly ownership harder); a
`solve_sequence` free function only, no stateful object (rejected: streaming
workloads get dates one at a time); allowing bounds/penalty changes with
automatic re-setup (rejected: hides a full rebuild behind an "update";
build a new sequence instead); aborting the whole sequence on the first
`MaxIterations` (rejected: statuses are per-date data, the caller decides).

**Consequences:** `solve_next` reports iteration-only `solve_time` (workspace
semantics); the batch form returns one `Solution` per step and stops only on
invalid steps; 2.6 tracking-error sugar and any future Pro batch engine
should build on `PortfolioSequence` rather than reimplementing warm-start
plumbing.

---

## 2026-07-22 — Infeasibility certificates: original-space detection, auditable output

**Decision:** Roadmap 2.2 ships OSQP-style infeasibility detection with two
Ledge-specific choices. First, candidate directions are the differences of
**original-space** iterates between consecutive termination checks — never
scaled-space quantities — so a reported `PrimalInfeasible` / `DualInfeasible`
is as equilibration-independent as `Solved`. Second, every certificate is
normalized to unit infinity norm, attached to `Solution.certificate`, and
auditable by standalone checkers (`check_primal_certificate` /
`check_dual_certificate`) that recompute the Farkas / descent-ray residuals
independently, mirroring `check_kkt`. Default
`infeasibility_tolerance = 1e-5` (stricter than OSQP's `1e-4`; `0`
disables). The portfolio layer prepends a hint naming the participating
constraints in user vocabulary; `PortfolioSequence` restarts the next date
cold after an infeasible one.

**Why:**

- A false "your portfolio is infeasible" would burn the trust the milestone
  exists to build; scaled-space detection is the known false-positive source
  in OSQP issue history, and the checks are cheap at the existing
  termination cadence (`check_kkt` already runs there).
- Feasible problems admit no exact Farkas certificate at all, so with a
  strict tolerance the failure mode collapses to "missed detection", which
  falls back to the previous behavior (`MaxIterations` plus hints).
- Auditable certificates extend the project's "independent KKT check"
  policy to failures: users can verify the proof, not trust the status.
- An infeasible date's duals diverge along the certificate ray, so chaining
  them as the next warm start would poison recovery — measured in
  `tests/certificate.rs` (`sequence_recovers_cold_after_an_infeasible_date`).

**Alternatives considered:** per-iteration deltas as in OSQP (rejected:
checking on the termination cadence costs nothing extra and window
differences average out transient noise); projecting cone-violating
directions instead of rejecting them (rejected: hides real violations —
noise up to the tolerance is clamped, anything larger disqualifies);
OSQP's `1e-4` default (rejected: bias strict, false negatives are benign
here); keeping warm starts across infeasible dates (rejected on the test
above).

**Consequences:** `SolveStatus` gains two variants and `Solution` a
`certificate` field (pre-1.0 breaking change); Python raises on infeasible
statuses unless `raise_on_failure=False` and exposes
`SolveResult.certificate` (`InfeasibilityCertificate`); the `MaxIterations`
"primal residual dominates" hint now states that no certificate was found
within tolerance; 2.3 polishing must leave certificates untouched (polish
only `Solved` iterates).

---

## 2026-07-22 — Polishing: audit-gated active-set refinement, on by default

**Decision:** Roadmap 2.3 ships as an OSQP-style polishing step
(`polish.rs`) enabled by default (`polish = true`). After `Solved`, the
active set is guessed from the final iterate — a constraint counts as
active only when it is *tight within a scaled multiple of the stopping
tolerance* **and** its multiplier does not point the other way — and the
resulting equality-constrained KKT system is solved directly through the
same SMW reduction the iterations use (active bounds fold into the base
diagonal, so the reduced dimension stays `k + m_eq + m_active`). The
saddle system is regularized by `polish_regularization = 1e-6` and
corrected by `polish_refinement_iterations = 3` refinement rounds against
the unregularized matrix; up to four classic active-set passes drop
wrong-sign rows and add violated rows. The candidate is adopted **only**
when its `check_kkt`-audited worst residual improves on the ADMM
iterate's; `Solution.polished` records the outcome.

**Why:**

- Smoke-matrix residuals drop from ~1e-5/1e-6 to 1e-11..1e-15 for ~7-13%
  extra wall time — one reduced factorization per pass, no extra
  iterations. Polished objectives are exactly feasible, removing the
  small violation bias raw ADMM objectives carry.
- The audit gate makes polish-on a no-regression default (trust policy:
  best defaults, same as over-relaxation): a failed or non-improving
  polish returns the ADMM iterate untouched.
- Dual-sign-only active-set guesses (plain OSQP) misclassify interior
  variables carrying ~1e-19 multiplier sign noise; tightness-only guesses
  over-pin near degeneracy. Requiring both plus bounded drop/add passes
  fixed every observed misclassification on the test matrix.

**Alternatives considered:** adopting OSQP's acceptance rule (primal/dual
residual only, no sign audit) — rejected: degenerate active sets (e.g.
bang-bang portfolios where pins make the budget row redundant) yield
wrong-sign multipliers that stationarity alone cannot see, and Ledge
would ship uncertified duals; polishing `MaxIterations` iterates —
rejected: 2.2 decision requires certificates and failure diagnostics to
stay untouched; a dedicated dense KKT factorization for polishing —
rejected: the SMW shape already fits and keeps the factor-structure cost
model.

**Consequences:** `Solution` gains a `polished` field and settings gain
`polish` / `polish_regularization` / `polish_refinement_iterations`
(pre-1.0 breaking change); Python exposes `polish=True` and
`SolveResult.polished`; reported default residuals change everywhere
(docs/SMOKE_TIMINGS.md records a polish-off control run); vector ρ's
trigger question (is iteration count still the binding lever at large n?)
should now be re-measured against polish-on defaults.

---

## 2026-07-22 — Exact L1 turnover: dedicated prox block, audited subgradients

**Decision:** Roadmap 2.1 ships proportional transaction costs
\(\sum_i c_i|x_i-a_i|\) as an optional `L1Term` on `QpProblem`, handled by a
third ADMM consensus block (\(x=z_t\)) with an elementwise soft-threshold
update. The block adds one \(\rho I\) to the x-system diagonal, so the
SMW-reduced dimension is unchanged. The L1 multiplier becomes a first-class
dual (`DualVariables::l1`): `check_kkt` scores the subgradient interval and
signed-cost pinning, polishing pins no-trade assets at the anchor and folds
signed costs into trading assets' linear term, warm starts carry `l1_dual`,
Ruiz scaling transforms costs/anchor, and dual certificates add the L1
recession slope to the descent test. Sequences move the anchor in scaled
space (`Workspace::update_l1_anchor`) without touching the factorization
cache. Python exposes `l1_turnover_costs` (scalar broadcast or per-asset
array), requiring `previous_weights`.

**Why:**

- Proportional costs with a genuine no-trade region are the whole point of
  "exact L1" over the L2 approximation; the prox block gets stickiness
  machine-exact (asserted in `tests/l1.rs`) rather than approximately.
- The epigraph route adds \(2n\) constraint rows, growing the reduced
  dimension \(O(n)\) and destroying the factor-structure advantage — it
  survives only as the test oracle (Rust epigraph equivalence + proptest;
  Python cvxpy+Clarabel gold tests).
- The project's audit policy ("never ship uncertified multipliers") forced
  the full dual treatment: without `l1` duals, stationarity would carry an
  unattributed \(\pm c_i\) residual and every audit would fail.

**Alternatives considered:** epigraph reformulation (rejected above);
handling L1 only in the portfolio layer by iterating smooth solves
(rejected: no convergence story, no duals); leaving polishing incompatible
with L1 (rejected: no-trade pins are exactly active-set structure, and
kinked solutions are the common case at realistic cost levels).

**Consequences:** `QpProblem` gains `l1` and `DualVariables` gains `l1`
(pre-1.0 breaking change for struct literals); external comparison
adapters reject L1 problems until the harness grows an epigraph converter;
`RebalanceStep.previous_weights` now drives both the L2 fold-in and the L1
anchor; infeasibility Farkas rays exclude L1 duals (a bounded convex term
never causes primal infeasibility).

---

## 2026-07-22 — Tracking error as linear-cost sugar; rolling example published

**Decision:** Roadmap 2.6 ships `PortfolioProblem::with_tracking_benchmark`
(Python `benchmark_weights=`) implemented as a pure linear-cost shift
\(-\lambda\Sigma b\) computed through the factor structure; no solver
changes. The constant \(\tfrac{\lambda}{2}b^\mathsf{T}\Sigma b\) is dropped
from reported objectives. `RebalanceStep.benchmark_weights` rolls the
benchmark inside sequences (requires the base problem to have one). Roadmap
2.7 ships `docs/examples/rolling_backtest.py` — a seeded momentum backtest
(300 assets, 12 factors, 24 monthly dates, 10 bps L1 costs, tracking
benchmark) — with warm-start numbers published in
`docs/examples/README.md`: warm dates 79 → 37 iterations (2.1x), 1.71 →
0.91 ms (1.9x), factorizations 24 → 2 vs per-date cold solves.

**Why:** expanding \((w-b)^\mathsf{T}\Sigma(w-b)\) shows tracking is the
same QP; anything more (a benchmark type, a second quadratic) would
duplicate machinery for zero numerical gain. Requiring the base problem to
have a benchmark before steps may move it keeps "sequence = fixed
structure, moving data" uniform with the turnover-anchor rule. The example
uses generated data styled on public equity data so the repo stays
self-contained and the numbers reproducible from one seed.

**Alternatives considered:** storing the benchmark in the QP layer
(rejected: `QpProblem` already expresses it through `linear`); allowing
steps to introduce a benchmark mid-sequence (rejected: silently changes
the reported-objective convention mid-stream); shipping real market data
in the repo (rejected: licensing and size for no additional evidence).

**Consequences:** reported objectives on tracking problems omit the
benchmark-variance constant (documented on the method); the sequence keeps
a copy of the raw covariance to recompute the shift, which is \(O(nk)\)
per benchmark update; the example doubles as the measured-workflow
reference for 2.8 user interviews.

---

## 2026-07-22 — Vector ρ re-evaluated: stays deferred (measured)

**Decision:** Keep the scalar adaptive penalty. Vector ρ (per-block or
per-constraint penalties) remains deferred, now on measurement rather than
suspicion. Reopen only when a real workload fails under defaults with
diagnostics implicating penalty imbalance between blocks (e.g. the
"rho re-tuned >10 times" hint with the residual gap concentrated in one
block) — and then evaluate an *adaptive* per-block scheme against its
factorization-cache cost, not a static factor.

**Why:** A throwaway prototype (env-hook scaling one consensus block's
penalty consistently through the x-system columns and dual updates;
release build, defaults otherwise) measured on the smoke matrix:

- Equality-block boosts ×3–×1000 (OSQP's own heuristic is ×1e3 on
  equality rows) change nothing measurable: identical iteration counts
  across the whole matrix (30/90/170/260/660), polish-off residuals
  differing in the third digit. After Ruiz equilibration the budget /
  exposure rows are simply not the binding block.
- Inequality-block factors on instances with explicit rows are
  instance-dependent with a sign flip: n=1000, m=20 improves 230 → 140
  iterations at ×0.1, n=2000, m=50 improves 500 → 310 at ×0.3, but
  n=500, m=10 is best at ×1 and n=5000, m=100 *worsens* 1270 → 2030 at
  ×0.3. No static factor wins across the declared envelope.
- Iteration count is still the dominant cold cost at large n (n=5000:
  ~1.2 s in iterations vs ~tens of ms setup), so the lever question was
  worth re-asking — but a static per-block ρ is not that lever.

**Alternatives considered:** OSQP-style per-constraint ρ with an
active/inactive split — their motivation is boxes encoded as rows, while
Ledge's box block already has its own consensus and the adaptive scalar ρ
tunes it; shipping the block factors as settings — rejected, no
configuration wins across the smoke matrix, it would only push tuning onto
users and every distinct penalty combination is another factorization
cache key.

**Consequences:** Roadmap 1.5's open question is closed for M2; the
prototype is not committed (scalar ρ code path unchanged). Any future
adaptive per-block scheme must account for the workspace design: penalties
that rarely repeat would forfeit the zero-refactorization steady state
rolling sequences currently reach.

---

## 2026-07-22 — Constraint templates: append-only sugar, boxes before rows

**Decision:** Roadmap 3.1 ships as `PortfolioProblem` builder methods that
compile portfolio vocabulary onto the existing constraint machinery:
`with_industry_neutrality` (targets derived from the tracking benchmark) /
`with_group_targets` (explicit targets) emit one indicator equality row per
group; `with_style_bounds` emits the finite sides of each exposure band as
inequality rows and collapses exact bands to equality rows;
`with_concentration_limit` and `with_short_limit` only tighten the box
bounds. Templates **append** to the user constraint blocks and their rows
are indistinguishable from user rows afterwards; `with_equalities` /
`with_inequalities` keep replace semantics and must be called first.
Invalid template data (empty groups, out-of-range ids, crossing or
doubly-infinite bands, caps contradicting existing bounds) fails at build
time with the new `PortfolioError::Template`.

**Why:**

- Interviews and the migration guide both show users hand-rolling exactly
  these matrices; the errors they make (sign of the lower-band row, missing
  groups) are compile-time-checkable, so check them.
- Appending (not replacing) lets templates stack with each other and with
  hand-built rows, and makes sequence semantics free: template targets are
  ordinary user RHS entries, so `RebalanceStep::equality_rhs` /
  `inequality_rhs` roll them with cached factorizations intact.
- Box templates deliberately produce no rows: every appended row grows the
  reduced dimension `r = k + m`, while bound tightening is free. A
  concentration cap as `2n` inequality rows would be strictly worse.
- A total short-budget cap was excluded: `sum_i max(-w_i, 0) <= S` needs a
  long/short variable split (structural change); a per-asset short cap is
  the box-compatible form.

**Alternatives considered:** a separate `ConstraintTemplateBuilder` object
compiling to `(matrix, rhs)` pairs (rejected: two ways to say the same
thing, and users would still stack matrices by hand); storing template rows
separately from user rows (rejected: sequences would need new step fields
per template and the row order would be invisible); silently skipping
doubly-infinite style bands (rejected: a no-op constraint is a bug in the
caller's data).

**Consequences:** `PortfolioError` gains the `Template` variant (pre-1.0
change); documented row order (groups in id order; per style upper row then
negated lower row) is API; Python constructor kwargs `industry_ids` /
`industry_targets` / `style_matrix` / `style_lower` / `style_upper` /
`max_weight` / `max_short` mirror the builders on `PortfolioProblem` and
`solve_mean_variance_factor`.

---

## 2026-07-22 — Serialization: non-default serde feature, validated deserialization

**Decision:** Roadmap 3.3 ships as a non-default `serde` cargo feature on
`ledge-core` (re-exported by `ledge`) deriving `Serialize` / `Deserialize`
for `QpProblem`, `PortfolioProblem`, `SolverSettings`, `WarmStart`, and
`Solution` (duals, residuals, diagnostics, certificates included), format
left to the caller. Three wire-format choices: `Matrix` travels as
`{rows, cols, data}` and rebuilds through `Matrix::new`;
`PortfolioProblem` travels as its builder inputs and replays the builder
methods on deserialization; bound vectors travel as `Option<f64>` entries
(`null` = unbounded side). The Python wheel enables the feature and adds
`PortfolioProblem.to_json()` / `from_json()` and `SolveResult.to_json()`.

**Why:**

- The M2 exit criterion "external real workloads in the regression set" and
  the problem-instance issue template both need a lossless, replayable dump
  format users can produce without writing conversion code.
- Deserialization must not become a validation bypass: `Matrix` invariants
  protect indexing, and replaying `PortfolioProblem` builders means a
  tampered dump fails with the same errors as wrong constructor input
  (tested). `QpProblem` needs no shadow — every solve entry already
  validates it.
- JSON cannot represent infinities (`serde_json` writes `null` and refuses
  to read it back), and unbounded box sides are common. `Option<f64>` is
  self-describing in every serde format and reads naturally in JSON;
  format-specific string hacks (`"inf"`) would break non-self-describing
  binary formats like `postcard`.
- Bit-exact replay is the whole point of a reproduction dump, so the tests
  assert the replayed iterate path is identical and the docs require
  `serde_json`'s `float_roundtrip` feature (default `serde_json` parsing
  may be off by 1 ULP).

**Alternatives considered:** a bespoke text format (rejected: serde gets
JSON + binary + every other format for one derive); serializing
`PortfolioProblem` field-for-field (rejected: deserialization would skip
builder validation and panics become reachable); making `serde`
a default feature (rejected: default build stays dependency-free);
shipping save/load helpers (rejected: `serde_json::to_string` is already
one line, and format choice belongs to the caller).

**Consequences:** the `serde` feature becomes part of the public API
surface (wire compatibility matters from now on; field renames are
breaking); solutions from `NumericalFailure` round-trip only through
binary formats (non-finite iterates); CI runs the round-trip suite via
`cargo test -p ledge-core --features serde`; the bug-report flow can now
ask for `problem.to_json()` + `result.to_json()` attachments.

---

## 2026-07-22 — Batch over accounts: keep the parallel loop public

**Decision:** Roadmap 3.2 ships as `solve_batch(&[BatchAccount], settings)
-> Vec<Result<Vec<Solution>, PortfolioError>>` (module `batch.rs`): one
`PortfolioSequence` per account, distributed over rayon's thread pool behind
a **non-default `rayon` feature** (the same function runs serially, with
bit-identical results, without it). `BatchAccount` adds one semantic knob,
`chain_previous_weights`: after a `Solved` date the turnover anchor moves to
that date's solved weights; non-`Solved` dates leave it (the account did not
trade); an explicit step anchor wins. Failures are isolated per account.
Python gets `ledge.solve_batch(problems, steps, ...)` with per-date step
dicts, GIL released for the whole batch, and the feature enabled in the
wheel. The published "1 model × 500 accounts × 250 dates" number lives in
`benchmarks/results/2026-07-batch/`.

**Why:**

- Accounts share no state, so the account axis is embarrassingly parallel;
  one `par_iter` plus the existing sequence machinery is the whole engine.
  Anything a user could reproduce with rayon in ten lines belongs in the
  public core; the external boundary starts at persistent deployment and
  orchestration, not ordinary in-process composition.
- Parallel == serial bit-for-bit keeps the trust policy intact: the feature
  changes wall-clock, never answers, and CI asserts it by running the same
  test file both ways.
- Anchor chaining is the standard backtest convention and cannot be
  expressed by precomputed steps (each date's anchor is the previous date's
  *solution*); without it the batch API could not run the workload the
  throughput gate describes. Chaining only from `Solved` dates matches
  "previous weights are what the account actually holds".
- Per-account error isolation follows the sequence philosophy one level up:
  one account's bad feed must not discard 499 finished accounts.

**Alternatives considered:** rayon as a default dependency (rejected: the
default build stays dependency-free, same policy as `serde`); a
`solve_batch` that returns the first error (rejected: isolation above); a
scheduling/checkpoint "batch engine" object (rejected for this repository:
persistent service, checkpoint/resume, and result storage are external
operations under `OPEN_CORE.md` §2); parallelism inside a single solve
(rejected: the per-solve
kernels are memory-bound at these sizes and the account axis already
saturates cores).

**Consequences:** `ledge-core` / `ledge-portfolio` gain the `rayon` feature and
`solve_batch` / `BatchAccount` / `AccountResult` exports; the Python wheel
depends on rayon from now on; the M3 throughput exit criterion is met
(12.9 s for 125k account-dates on 4 vCPUs, 4.0x over serial, samples
published); any external batch orchestration should wrap the public
`BatchAccount` sequences rather than growing a second numerical engine.

---

## 2026-07-22 — Docs site: mdBook built in CI, public deployment gated

**Decision:** Roadmap 3.5 ships as an mdBook under `docs/book/` (tutorial /
guide / reference chapters; the cvxpy migration guide is `{{#include}}`d
verbatim so it keeps a single source). CI builds the site on every push and
uploads it as an artifact (`docs` job). **Public GitHub Pages deployment is
a separate, manual-only workflow** (`docs-deploy.yml`,
`workflow_dispatch`), because Pages would make the site public while the
repository is private — deployment is therefore gated on the 1.6
public-release decision, not on this task.

**Why:** the roadmap wants the docs-site skeleton and its content review to
be finished engineering before the public gate, so flipping the switch
later is a one-click action, not a writing project. Engineering docs
(PLAN / ROADMAP / DECISIONS / algorithm) stay out of the book — they are
maintainer-facing and would drift; the book links to them instead.

**Alternatives considered:** deploying to Pages now (rejected: exposes a
private project); Sphinx/mkdocs (rejected in `PLAN.md` §4.2 — one Rust
toolchain, no second docs stack); writing all chapters as `{{#include}}`s
of existing docs (rejected: repository docs are written for the repo tree,
the tutorial/tuning content needed writing for a reader-facing voice).

**Consequences:** the book must build for CI to pass (`mdbook build
docs/book`); after the 1.6 gate the maintainer enables Pages
(Settings → Pages → GitHub Actions) and runs the deploy workflow, or adds a
push trigger; `docs/book/book/` is gitignored build output.

---

## 2026-07-22 — Comparison harness gains L1 variants; fourth report published

**Decision:** the `compare` harness runs every smoke-matrix instance twice:
the smooth base problem and an `-l1` variant adding proportional turnover
costs (`--l1-bps`, default 10 bps per asset) anchored at the shared primal
start. Ledge keeps its prox block; external solvers receive the documented
epigraph reformulation (`n` extra variables, `2n` inequality rows) from
`convert.rs`, with epigraph multipliers mapped back for independent
`check_kkt` re-verification. The fourth protocol report
(`benchmarks/results/2026-07-l1/`) is the first with polish-on defaults and
the first with cross-solver L1 measurements.

**Why:** the adapters gained the epigraph conversion when L1 landed
(2026-07-22 L1 decision), but no published number exercised it — the
"prox block vs epigraph" claim in README and the 1.9 note was implemented
yet unmeasured. Polishing also changed default residuals (~1e-5 → ~1e-11),
so all previously published Ledge accuracy caveats were stale.

**Alternatives considered:** a separate L1-only report (rejected: the
side-by-side smooth vs `-l1` rows on identical instances *are* the
evidence); benchmarking L1 at multiple cost levels (rejected for now: one
realistic level, 10 bps, keeps the report readable; the flag exists for
reruns).

**Consequences:** future report reruns include L1 rows by default;
`README`/note claims about L1 cost cite `2026-07-l1`; the accuracy-caveat
paragraphs in earlier reports remain as history (each report is
self-describing).

---

## 2026-07-22 — Open-core boundary: repo-level, no extraction folder

**Decision:** Do **not** create a dedicated folder that copies the
open-source parts of the repository out as a separate "public" source of
truth. The boundary stays **repository-level**: this repository *is* the
open core (Apache-2.0) and its working tree at the 1.6 gate, scoped by
the new [`OPEN_CORE.md`](OPEN_CORE.md) manifest, is the single source of
truth. Any future proprietary extension lives in a **separate private
repository** that depends on the published `ledge-core` / `ledge-portfolio`
crate APIs.
Shipped alongside: `docs/OPEN_CORE.md` (path-level inventory of what is open,
what must never enter this tree, the strategy-doc decision, and the
public-release runbook) and `scripts/check_open_core.sh` (secrets /
external-surface / license / publish gate check).

**Why:**

- There is currently **no proprietary code in the tree** — everything here is
  the intended open core. A folder that "extracts the open part" would copy
  the whole repository into a subdirectory.
- A copy is a second source of truth that drifts on the first edit, and it
  breaks the Cargo workspace, CI, Python packaging, and every relative path
  (`readme = "../../README.md"`, doc links) that assumes the current layout.
- The boundary the maintainer actually wants — one authoritative definition
  of the open surface — is delivered by a committed manifest + the repo
  itself, with zero duplication, and a script makes it enforceable.
- Repo-level separation also enforces boundary discipline: an external
  extension can only consume *published* public APIs, which keeps the public
  API honest.

**Alternatives considered:** extraction subfolder (rejected above); monorepo
with a private extension folder that is git-ignored or private-submoduled
(rejected: mixes closed code into the public tree and its history, and the
boundary would depend on ignore rules); rewriting history now to strip
old internal strategy docs (needed only if the maintainer wants them absent
from the public record — handled as a history choice in the runbook, not a
prerequisite of this boundary decision).

**Consequences:** option 1 from the manifest is selected: the current
`PLAN` / `ROADMAP` / `DECISIONS` files contain only public technical and
boundary material; pricing/customer/revenue/pilot strategy is not maintained
here. Older internal content still exists in git history, so publication must
either knowingly accept that history or start from a clean/squashed public
history. `scripts/check_open_core.sh --release` is the strict pre-publish
gate; roadmap 1.6 points at the `OPEN_CORE.md` runbook.

---

## 2026-07-22 — Approve Apache-2.0, clean public history, and release names

**Decision:** Apply Apache-2.0 to the complete open-core tree; publish a new
clean-root repository while retaining this historical repository as a private
archive; release version `0.2.0`; publish Rust packages `ledge-core` and
`ledge-portfolio` with library name `ledge`; publish Python distribution
`ledge-portfolio` with import name `ledge`.

**Why:** Apache-2.0 provides an explicit permissive copyright and patent
grant. A clean root keeps removed commercial planning and the former license
era out of the public Git record without rewriting the internal archive. The
registry package `ledge` belongs to an unrelated project, while
`ledge-portfolio` preserves the project identity and `[lib] name = "ledge"`
keeps Rust source imports stable.

**Alternatives considered:** publish the existing history (rejected because
the old planning text would become permanent public history); force-rewrite
the archive (rejected as disruptive); retain Rust package `ledge` (rejected
because the crates.io name is occupied).

**Consequences:** the release-gate tree carries Apache metadata and legal
files in each distributable archive, all Cargo commands address package
`ledge-portfolio`, and the strict gate must pass before export. Repository
creation/visibility, trusted-publisher setup, tags, and registry publication
remain explicit maintainer operations.

---

## Template for future entries

```text
## YYYY-MM-DD — Short title

**Decision:** ...
**Why:** ...
**Alternatives considered:** ...
**Consequences:** ...
```
