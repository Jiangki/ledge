# Why factor structure is worth exploiting

*Technical note (roadmap 1.9). All numbers below are medians from the
published protocol reports under
[`benchmarks/results/`](../benchmarks/results/) — same instances, same
machine class, ≥10 repeats, every returned point re-verified with Ledge's
independent `check_kkt` on the original data. Nothing here is extrapolated.*

## 1. The shape of the problem

A factor risk model writes the asset covariance as

\[
\Sigma = F \Omega F^\mathsf{T} + \mathrm{diag}(d),
\qquad F \in \mathbb{R}^{n \times k},\; k \ll n .
\]

A mean-variance rebalance is then a convex QP over weights \(w\):

\[
\min_w\; \tfrac{\lambda}{2} w^\mathsf{T}\Sigma w - \mu^\mathsf{T}w
\quad \text{s.t.} \quad
\mathbf{1}^\mathsf{T}w = b,\;\; \ell \le w \le u,\;\; Aw \le c ,
\]

with \(n\) in the hundreds to thousands, \(k\) in the tens, and a modest
number \(m\) of explicit constraint rows. The covariance carries all the
dimensionality; the *information* in it is only \(O(nk)\) numbers.

## 2. Three ways to hand this to a solver

**Dense `Q` (the default thing that happens).** Materialize
\(\Sigma\) and pass an \(n \times n\) quadratic to a general QP solver.
This is what you get from a naive `cvxpy` model or a hand-rolled call.
Building \(Q\) costs \(O(n^2 k)\), storing it \(O(n^2)\), and the solver's
factorization no longer sees any sparsity: interior-point iterations pay
dense \(n \times n\) algebra.

**Lifted (the sophisticated manual fix).** Add \(k\) auxiliary variables
\(y = \Omega^{1/2} F^\mathsf{T} w\) and \(k\) equality rows; the objective
becomes sparse-diagonal
(\(\tfrac{1}{2} w^\mathsf{T}\!\mathrm{diag}(d)\,w +
\tfrac{1}{2}\lVert y\rVert^2\)). A general sparse solver handles this well
— it is the strongest external baseline and every published Ledge
comparison includes it.

**Native factor form (Ledge).** Take \(F, \Omega, d\) directly. The ADMM
x-update solves a system of the form
\(\bigl(\mathrm{diag}(\tilde d) + G G^\mathsf{T}\bigr) x = r\) with
\(G \in \mathbb{R}^{n \times r}\), and Sherman–Morrison–Woodbury reduces it
to an \(r \times r\) Gram factorization with

\[
r = k + m \quad (\text{factors} + \text{explicit rows}) .
\]

Per penalty value that is \(O(nr^2)\) once, then \(O(nr)\) per iteration —
at \(n = 5000, k = 100\) the factored system is \(\sim 200 \times 200\),
not \(5000 \times 5000\). Boxes and the L1 turnover term ride in consensus
blocks that never grow \(r\).

## 3. What the gap measures: dense vs lifted, same solver

The cleanest evidence needs no Ledge at all: give the *same* external
solver the same instance both ways
([`2026-07-workspace`](../benchmarks/results/2026-07-workspace/README.md)
report, cold solves, median ms):

| instance | OSQP dense-Q | OSQP lifted | gap | Clarabel dense-Q | Clarabel lifted | gap |
|---|---:|---:|---:|---:|---:|---:|
| n=500, k=10 | 108.2 | 14.1 | **7.7x** | 140.6 | 3.4 | **41x** |
| n=1000, k=20 | 919.9 | 81.7 | **11x** | 1233.8 | 10.7 | **115x** |

Rolling re-solves show the same one-to-two orders of magnitude (Clarabel
n=1000: 1214 vs 10.6 ms/step). Dense-Q is not benchmarked above
\(n = 1000\) because setup alone grows to seconds.

**This is the core claim, and it is solver-independent: ignoring factor
structure costs 1–2 orders of magnitude before any solver comparison even
starts.** If you take one thing from this note: never hand a factor-model
QP to a solver as a dense \(\Sigma\).

## 4. What a native factor engine adds beyond the lifted trick

If the lifted reformulation captures most of the structural win, why not
stop there? Because the lifted form recovers only the *cold-solve algebra*.
The rebalancing *workflow* has more structure than one QP:

**Rolling warm starts and factorization reuse.** Across dates, mostly
\(\mu\) and the previous weights change. Ledge's workspace freezes the
equilibration and caches the reduced factorization; a warm rolling step
reaches a steady state of zero refactorizations. Measured on the
repository's momentum backtest: 79 → 37 iterations and 1.71 → 0.91 ms per
warm date vs cold; on the protocol's rolling phase Ledge leads or ties both
external solvers up to \(n = 1000\) (interior-point Clarabel cannot warm
start; it pays its full ~9 iterations every date).

**Exact L1 turnover without growing the system.** Proportional transaction
costs \(\sum_i c_i \lvert w_i - w_{0,i}\rvert\) have no native QP encoding:
a general solver needs the epigraph reformulation — \(n\) extra variables
and \(2n\) inequality rows, which is exactly the structure-destroying move
from §3, now inside the constraint matrix. Ledge handles the term as a
soft-threshold prox block: \(r\) unchanged, no-trade region machine-exact.
The [`2026-07-l1`](../benchmarks/results/2026-07-l1/README.md) report
measures this head-to-head on L1-instrumented instances of the same smoke
matrix, and the effect is decisive: adding 10 bps proportional costs slows
Ledge by ~8% but the epigraph-fed external solvers by 2–6x, so **on L1
instances Ledge is the fastest solver at every size, cold and rolling** —
including \(n = 5000\), where lifted Clarabel wins the smooth cold solve
(470 vs 1093 ms) but loses the L1 one (1902 vs 1181 ms). Since
proportional transaction costs are the realistic rebalancing case, the L1
rows are the comparison production users should read.

**Portfolio semantics survive.** Duals come back per budget / box / row /
L1 subgradient (not per reformulated cone), infeasibility certificates name
the conflicting portfolio constraints, and audits (`check_kkt`) run on the
original data. With a hand-lifted model, mapping multipliers and
diagnostics back is user code that must be maintained and can silently rot.

## 5. What structure does *not* buy (honest limits)

- **Large-n smooth cold starts.** A hand-lifted Clarabel formulation is
  still the fastest cold baseline at \(n \ge 2000\) on smooth instances
  (n=5000: 470 ms vs Ledge's 1093 ms with polish-on defaults). First-order
  methods pay iteration counts that grow with n; interior-point pays ~9
  iterations regardless. Ledge narrows this with warm starts and wins it
  outright only once L1 costs enter (§4).
- **Accuracy at default tolerances** was a real caveat (~1e-5 residuals)
  until audit-gated polishing landed; polished defaults now return ~1e-11
  or better and objectives that match the tight-tolerance optimum digit
  for digit. Polish costs single-digit percent cold and more on cheap warm
  steps (its one direct solve is a larger fraction of a short step);
  hard-to-classify solutions can plateau above ~1e-11, reported honestly
  via `polished` and the audited residuals.
- **Many explicit rows.** The reduction is \(r = k + m\); if your
  formulation carries \(O(n)\) explicit rows, the factor advantage is gone
  by construction. Box-expressible constraints (concentration, short
  limits) are compiled to boxes precisely to keep \(m\) small.

## 6. Takeaways

1. Factor structure is worth 1–2 orders of magnitude versus dense-`Q` —
   measurable inside any single general-purpose solver, before Ledge enters
   the picture.
2. The lifted reformulation recovers the cold-solve algebra; a native
   factor engine additionally exploits the *workflow*: warm starts,
   factorization reuse across dates, prox-block L1 turnover, and audits in
   portfolio vocabulary.
3. The remaining honest gap — large-n cold starts against lifted
   interior-point — is documented in every published report, with raw
   samples, and is not the workload rebalancing systems actually run.

*Reports: [first protocol report](../benchmarks/results/2026-07/README.md)
· [over-relaxation re-run](../benchmarks/results/2026-07-over-relaxation/README.md)
· [workspace re-run](../benchmarks/results/2026-07-workspace/README.md)
· [polish + L1 re-run](../benchmarks/results/2026-07-l1/README.md)
· [batch throughput](../benchmarks/results/2026-07-batch/README.md).*
