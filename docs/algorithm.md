# Algorithm notes

This document records Ledge's mathematical conventions, what is implemented
today, and what is not. The authoritative code entry point is
`crates/ledge-core/src/solver.rs`.

## 1. QP formulation and notation

The solver accepts

\[
\begin{aligned}
\min_x\quad & \frac12x^\mathsf{T}Qx+q^\mathsf{T}x
+\sum_i c_i\,|x_i-a_i|,\\
\text{s.t.}\quad
&A_e x=b_e,\quad A_i x\le b_i,\quad \ell\le x\le u,\\
&Q=F\Omega F^\mathsf{T}+\operatorname{diag}(d),
\end{aligned}
\]

where \(\Omega\succeq0\), \(d\ge0\), and \(c\ge0\). The optional weighted-L1
term (`QpProblem.l1`, an `L1Term { costs, anchor }`) models proportional
transaction costs around a previous portfolio \(a\); \(c=0\) or a missing
term recovers the plain QP. \(\Omega\) may be diagonal or dense.
Let \(\Omega=LL^\mathsf{T}\) and \(G=FL\); then
\(F\Omega F^\mathsf{T}=GG^\mathsf{T}\). The implementation allows
semidefinite (not strictly positive definite) \(\Omega\), but rejects obvious
negative pivots.

Stack equalities and inequalities as \(A=[A_e;A_i]\) and define the sets

\[
\mathcal C=\{z:z_e=b_e,\ z_i\le b_i\},\qquad
\mathcal B=[\ell,u].
\]

The ADMM split is \(Ax=z_c,\ x=z_b\), plus \(x=z_t\) when the L1 term is
present. The nonsmooth part of the objective is
\(\delta_\mathcal C(z_c)+\delta_\mathcal B(z_b)
+\sum_i c_i\,|(z_t)_i-a_i|\).

## 2. ADMM and adaptive penalty

The solve starts from \(\rho>0\) and uses a small proximal regularizer
\(\sigma>0\). Given unscaled duals \(y_c,y_b\), the x-update is

\[
\begin{aligned}
x^{k+1}=\arg\min_x\;&
\frac12x^\mathsf{T}Qx+q^\mathsf{T}x
+\frac{\sigma}{2}\|x-x^k\|_2^2\\
&+\frac{\rho}{2}\|Ax-z_c^k+y_c^k/\rho\|_2^2
+\frac{\rho}{2}\|x-z_b^k+y_b^k/\rho\|_2^2.
\end{aligned}
\]

The resulting linear system is

\[
\left(Q+\sigma I+\rho A^\mathsf{T}A+\rho I\right)x^{k+1}
=\sigma x^k-q+A^\mathsf{T}(\rho z_c^k-y_c^k)
+\rho z_b^k-y_b^k.
\]

The z and dual updates apply over-relaxation with coefficient
\(\alpha\in(0,2)\) (`over_relaxation`, default `1.6`; `1.0` recovers plain
ADMM). With the relaxed quantities

\[
\hat z_c^{k+1}=\alpha Ax^{k+1}+(1-\alpha)z_c^k,\qquad
\hat z_b^{k+1}=\alpha x^{k+1}+(1-\alpha)z_b^k,
\]

the updates are

\[
\begin{aligned}
z_c^{k+1}&=\Pi_\mathcal C(\hat z_c^{k+1}+y_c^k/\rho),\\
z_b^{k+1}&=\Pi_\mathcal B(\hat z_b^{k+1}+y_b^k/\rho),\\
y_c^{k+1}&=y_c^k+\rho(\hat z_c^{k+1}-z_c^{k+1}),\\
y_b^{k+1}&=y_b^k+\rho(\hat z_b^{k+1}-z_b^{k+1}).
\end{aligned}
\]

Equality projection takes \(b_e\) directly; upper-bound inequalities use
elementwise \(\min(v,b_i)\); box projection is elementwise clamping.
Termination and reported residuals always use the true iterates
(\(Ax^{k+1}\), never the relaxed blend), so `Solved` does not depend on
\(\alpha\). On the synthetic smoke matrix the default \(\alpha=1.6\) cuts
iteration counts roughly 1.7–2.9x versus \(\alpha=1.0\) (see
[`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md)).

When the L1 term is present, a third consensus block \(x=z_t\) with dual
\(y_t\) joins the same pattern. Its z-update is the proximal operator of
the weighted L1 distance to the anchor — elementwise soft-thresholding:

\[
(z_t^{k+1})_i=a_i+S_{c_i/\rho}\!\left((\hat z_t^{k+1}+y_t^k/\rho)_i-a_i\right),
\qquad
S_\kappa(v)=\operatorname{sign}(v)\max(|v|-\kappa,0),
\]

with the same over-relaxed blend and dual update as the other blocks. The
block contributes another \(\rho I\) to the x-system diagonal and one more
\(\rho z_t^k-y_t^k\) term to its right-hand side, so the reduced dimension
of §4 is unchanged. Consensus residuals for termination and adaptive-ρ
balancing include \(x-z_t\) and \(\Delta z_t\).

By default, every `adaptive_rho_interval` iterations the solver compares ADMM
consensus residuals:

\[
r=\max(\|Ax-z_c\|_\infty,\|x-z_b\|_\infty),\qquad
s=\rho\|A^\mathsf{T}\Delta z_c+\Delta z_b\|_\infty.
\]

If \(r>\tau s\), \(\rho\) is multiplied by the configured factor; if
\(s>\tau r\), it is divided, clamped between `minimum_rho` and `maximum_rho`.
Each penalty change rebuilds the SMW reduced factorization. Duals are stored
in unscaled form, so stored multipliers are not rescaled when \(\rho\) changes.
Set `adaptive_rho=false` for a fixed penalty. The consensus residuals used
for penalty balancing are evaluated on the true iterates, not the relaxed
blends.

With a [`Workspace`](#4a-workspace-factorization-reuse) the factorizations
additionally persist *across* solves in a penalty-keyed cache: the policy is
unchanged, but a revisited \(\rho\) reuses its factorization instead of
rebuilding it.

## 3. Ruiz equilibration and cost scaling

With default settings (`scaling_iterations = 10`; `0` disables), the solver
iterates on an equilibrated copy of the data. Each Ruiz pass computes a
variable scaling \(E=\operatorname{diag}(e)\), row scalings \(D_e, D_i\) for
the two constraint blocks, and a cost scalar \(c\), accumulated over passes:

\[
Q\mapsto cEQE,\quad q\mapsto cEq,\quad
A\mapsto DAE,\quad b\mapsto Db,\quad
[\ell,u]\mapsto E^{-1}[\ell,u].
\]

The factor structure is preserved: with \(F\Omega F^\mathsf{T}=GG^\mathsf{T}\),
the scaled quadratic is represented by row-scaled columns \(\sqrt{c}\,EG\) and
diagonal \(cE d E\) — \(Q\) is never formed. Column norms of \(Q\) are
estimated per pass from the exact diagonal
\(Q_{jj}=\lVert G_j\rVert^2+d_j\) and the Cauchy–Schwarz bound
\(|Q_{ij}|\le\lVert G_i\rVert\,\lVert G_j\rVert\); estimate quality affects
conditioning only, never correctness, because the scaled problem is built from
exactly scaled data. Per-pass factors are clamped to \([10^{-6},10^{6}]\).
After the Ruiz pass, an OSQP-style cost normalization brings the mean
estimated quadratic column norm and \(\lVert q\rVert_\infty\) toward one.

Auditability rule: **scaling only changes the space ADMM iterates in.**
Warm starts are scaled on entry; iterates are unscaled
(\(x=E\,\bar x\), \(y=D\bar y/c\), \(y_b=E^{-1}\bar y_b/c\)) before every
termination check, and `check_kkt`, all reported residuals, objective values,
and returned iterates are always evaluated on the original data. A reported
`Solved` therefore never depends on the equilibration.

## 4. SMW reduction

Let

\[
B=\operatorname{diag}(d)+(\sigma+\rho)I,\qquad
U=[G,\sqrt{\rho}A^\mathsf{T}].
\]

The x-update left-hand side is \(H=B+UU^\mathsf{T}\). Ledge does not form
\(Q\) or \(H\) explicitly; it applies

\[
H^{-1}=B^{-1}
-B^{-1}U(I+U^\mathsf{T}B^{-1}U)^{-1}U^\mathsf{T}B^{-1}.
\]

Only one Cholesky factorization of

\[
S=I+U^\mathsf{T}B^{-1}U
\]

is required. With \(n\) assets, \(k\) factors, and \(m\) explicit linear
constraints, the reduced dimension is \(r=k+m\):

- setup time is roughly \(O(nr^2+r^3)\);
- each iteration is roughly \(O(nr+r^2)\), plus constraint matrix products;
- storage is roughly \(O(nr+r^2)\).

The box split \(x=z_b\) contributes \(\rho I\), so it does not add \(n\)
reduced columns. That is why the current implementation can handle many assets.
The same holds for the L1 turnover split \(x=z_t\) of §2 — expanding
\(c^\mathsf{T}|x-a|\) via an epigraph with \(2n\) general linear
constraints would make \(r\) grow as \(O(n)\) and remove the advantage,
which is why the epigraph form appears only as a test oracle
(`tests/l1.rs`).

## 4a. Workspace factorization reuse

`Solver::workspace(&QpProblem)` builds a `Workspace` that pays the
equilibration and the \(O(nr^2+r^3)\) reduced factorization once and reuses
both across solves. Between solves, `update_linear` and
`update_equality_rhs` / `update_inequality_rhs` replace \(q\), \(b_e\), or
\(b_i\) in place: the scalings \(E, D_e, D_i, c\) frozen at construction are
reapplied as exact transforms (\(q \mapsto cEq\), \(b \mapsto Db\)), so the
scaled problem remains an exact image of the updated original data and the
auditability rule of §3 is untouched. Because \(E\) and \(c\) are frozen, an
updated cost may be normalized slightly differently than a fresh
equilibration would choose; that affects conditioning only, never
correctness or termination.

Factorizations depend on \(Q\), \(A\), \(\sigma\), and \(\rho\) — not on
\(q\), \(b\), or bounds — so data updates never invalidate them. The
workspace keeps a small LRU cache keyed by the exact \(\rho\); every solve
replays the one-shot penalty policy (start at `settings.rho`, adapt by the
usual multiplicative ladder), so lookups are bit-exact, the iterate path is
identical to a fresh `Solver::solve` of the same data, and a warm rolling
sequence factorizes each visited penalty exactly once per workspace. A cache
miss remains a full \(O(nr^2)\) recomputation:
\(B=\operatorname{diag}(d)+(\sigma+\rho)I\) reweights every entry of \(S\),
so no rank-one update shortcut exists in this formulation.

## 4b. Rolling sequences (`solve_sequence`)

`PortfolioProblem::sequence()` wraps a workspace at the portfolio level
(module `sequence.rs`). Each `RebalanceStep` may replace expected returns
\(\mu\), the turnover anchor \(w_0\), the tracking benchmark \(b\), the
budget, or constraint right-hand sides; the sequence recomputes the folded
linear cost
\(q = -(\mu + \gamma_{\text{turnover}} w_0 + \lambda\Sigma b)\)
(the \(\Sigma b\) product runs through the factor structure) and pushes it
through the workspace update path, moves the L1 anchor via
`Workspace::update_l1_anchor` when the problem has proportional costs,
then solves warm-started from the previous date's full primal/dual
solution. Only factorization-preserving updates are expressible — the
turnover penalty and the L1 costs are part of the problem structure and
therefore stay fixed for the life of a sequence. Steps are validated in
full before any state changes (atomic on rejection). Python exposes the same
object as `PortfolioProblem.sequence()` /
`PortfolioSequence.solve_next(...)`.

## 5. KKT conventions and termination

Returned multipliers use these sign conventions:

- equality duals are unconstrained in sign;
- inequality duals for \(A_i x\le b_i\) are nonnegative;
- box duals are merged normal-cone multipliers: nonpositive at the lower bound,
  nonnegative at the upper bound;
- L1 duals \(y_t\) (present only with an L1 term) are subgradients of the
  weighted L1 cost: \(|(y_t)_i|\le c_i\) always, and
  \((y_t)_i=c_i\operatorname{sign}(x_i-a_i)\) where the asset trades.

Stationarity is

\[
Qx+q+A_e^\mathsf{T}y_e+A_i^\mathsf{T}y_i+y_b+y_t=0.
\]

`check_kkt` computes independently:

1. primal residual: max violation of equalities, positive parts of
   inequalities, and box violations;
2. dual residual: max of stationarity \(\infty\)-norm, dual-cone violations,
   and L1 subgradient-interval violations;
3. complementarity residual: max absolute product of multipliers and slacks;
   the L1 term charges the shortfall from the required signed cost times the
   distance moved in that direction,
   \(\max\big((c_i-(y_t)_i)\,(x_i-a_i)_+,\ (c_i+(y_t)_i)\,(x_i-a_i)_-\big)\).

Every `check_termination_every` iterations the solver checks

\[
r_p\le\epsilon_{\rm abs}+\epsilon_{\rm rel}s_p,\qquad
r_d\le\epsilon_{\rm abs}+\epsilon_{\rm rel}s_d.
\]

\(s_p\) is built from primal, constraint, and RHS magnitudes; \(s_d\) from
\(Qx\), \(q\), and \(A^\mathsf{T}y\) magnitudes. Since box complementarity is
scored continuously (multiplier magnitude times distance to the active
bound), termination additionally requires the complementarity residual to
meet \(\max(\epsilon_p,\epsilon_d)\), keeping `Solved` as strong as the
earlier activity-window dual test.

## 5a. Infeasibility certificates

On infeasible problems the ADMM iterates do not converge, but their
*differences* do (OSQP-style detection). Every termination check compares
the original-space iterates against the previous check and tests two
normalized (unit \(\infty\)-norm) candidate directions
(`certificate.rs`; `SolverSettings.infeasibility_tolerance`
\(\varepsilon = 10^{-5}\), `0` disables):

**Primal infeasibility** (status `PrimalInfeasible`): the dual difference
\(\delta y=(\delta y_e,\delta y_i,\delta y_b)\) is a Farkas certificate when

\[
\|A_e^\mathsf{T}\delta y_e+A_i^\mathsf{T}\delta y_i+\delta y_b\|_\infty
\le\varepsilon,\qquad
b_e^\mathsf{T}\delta y_e+b_i^\mathsf{T}\delta y_i
+u^\mathsf{T}(\delta y_b)_++\ell^\mathsf{T}(\delta y_b)_-
\le-\varepsilon,
\]

with \(\delta y_i\ge0\), positive parts of \(\delta y_b\) supported on
finite upper bounds, and negative parts on finite lower bounds (cone noise
up to \(\varepsilon\) is clamped to zero; larger violations disqualify the
direction). Any feasible \(x\) would make the first expression's inner
product with \(x\) both zero and bounded above by the strictly negative
support gap — a contradiction.

**Dual infeasibility** (status `DualInfeasible`): the primal difference
\(\delta x\) is an unbounded descent ray when

\[
\|Q\,\delta x\|_\infty\le\varepsilon,\quad
q^\mathsf{T}\delta x+\sum_i c_i|\delta x_i|\le-\varepsilon,\quad
\|A_e\,\delta x\|_\infty\le\varepsilon,\quad
A_i\,\delta x\le\varepsilon,
\]

and \(\delta x\) is nonpositive where an upper bound is finite and
nonnegative where a lower bound is finite. The \(\sum_i c_i|\delta x_i|\)
term is the L1 turnover's recession slope (zero without an L1 term): along
the ray the proportional costs grow linearly, so the linear objective must
outrun them before the problem is genuinely unbounded.

The auditability rule of §3 extends to certificates: detection and the
reported certificate live in the original data space, and the standalone
checkers `check_primal_certificate` / `check_dual_certificate` recompute
the defining residuals independently, exactly as `check_kkt` does for
solutions. The portfolio layer prepends a hint naming the participating
constraints in portfolio vocabulary (budget, inequality caps, bounds), and
a rolling sequence restarts cold after an infeasible date because the
diverging duals are a certificate ray, not a useful warm start.

## 5b. Solution polishing

With default settings (`polish = true`), a `Solved` iterate is refined by
one direct active-set solve before it is returned. The active set is
guessed from the final iterate: a constraint counts as active when the
iterate sits on it within \(10(\epsilon_{\rm abs}+\epsilon_{\rm rel}s)\)
(per-row scale \(s\)) *and* its multiplier does not point the other way —
tightness alone over-pins near degeneracy, multiplier signs alone
misclassify interior variables carrying \(\sim10^{-19}\) sign noise.

Treating the active rows as equalities, the polishing step solves the
regularized KKT system

\[
\begin{bmatrix}Q+\delta I & A_{\rm act}^\mathsf{T}\\
A_{\rm act} & -\delta I\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}-q\\b_{\rm act}\end{bmatrix},
\qquad \delta = \texttt{polish\_regularization},
\]

by eliminating the multiplier block onto
\(H=Q+\delta I+\delta^{-1}A_{\rm act}^\mathsf{T}A_{\rm act}\), which has
exactly the SMW shape of §4: general active rows enter as reduced columns
scaled by \(\delta^{-1/2}\), and active *bound* rows are unit vectors whose
\(\delta^{-1}e_ie_i^\mathsf{T}\) folds into the base diagonal — so a
long-only portfolio with hundreds of pinned weights costs
\(r' = k + m_{eq} + m_{\rm act}\), not \(O(n)\). The regularization error is
removed by `polish_refinement_iterations` rounds of iterative refinement
against the unregularized matrix (OSQP-style). Because an ADMM-accurate
iterate can misclassify marginal constraints, up to four classic
active-set passes drop rows whose polished multiplier has the wrong sign
and add rows the polished iterate violates.

An L1 turnover term partitions the assets by the ADMM iterate: no-trade
assets are pinned at the anchor (an extra class of bound-style rows), and
trading assets contribute their signed cost \(\pm c_i\) to the linear term
of the smooth subproblem. Refinement passes also flip trade signs the
candidate contradicts, and the recovered L1 multipliers (the pin
multipliers, clamped to \([-c_i,c_i]\); the signed costs where trading)
enter the same audit as every other candidate.

The §3 auditability rule extends to polishing: every candidate is
re-audited with `check_kkt` on the original data, and the best candidate is
adopted only when its **worst** KKT residual improves on the ADMM
iterate's (`Solution.polished` records the outcome). Degenerate active
sets whose multiplier split is not unique — for example a bang-bang
portfolio where every weight sits on a bound and the budget row becomes
linearly dependent on the pins — produce sign-violating duals, fail the
audit, and fall back to the ADMM iterate rather than ship uncertified
multipliers. Certificates are never polished; only `Solved` iterates enter
the step. On the smoke matrix polishing moves worst residuals from
~1e-5/1e-6 to 1e-11..1e-15 for ~7-13% extra wall time (see
[`SMOKE_TIMINGS.md`](SMOKE_TIMINGS.md)).

## 6. Numerical limits and follow-up work

The current dense reduced Cholesky is a small, unpivoted implementation. It
should not be described as a mature sparse-direct backend. Adaptive \(\rho\),
Ruiz / cost scaling, over-relaxation, the cross-solve factorization cache
(§4a), the rolling sequence API (§4b), infeasibility certificates (§5a),
solution polishing (§5b), and the exact L1 turnover block (§2) are
implemented; still outstanding:

1. rank updates or a ρ-independent reformulation so penalty changes avoid
   the full \(O(nr^2)\) refactorization (data updates already avoid it);
2. vector \(\rho\) (per-constraint penalties) as a further convergence aid;
3. abstract cone projection and linear-system backends before SOCP.

A candidate SOCP direction is homogeneous self-dual embedding with cone
projection, but there is no code for that today and no commitment to its
numerical behavior.
