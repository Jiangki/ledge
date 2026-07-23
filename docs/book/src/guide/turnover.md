# Turnover and tracking error

## Smooth L2 turnover

`turnover_penalty` (Python) / `with_turnover_penalty` (Rust) adds
\\(\tfrac{\gamma}{2}\lVert w - w_0 \rVert^2\\) around the previous weights.
It discourages trading everywhere but never produces exact no-trades. The
penalty sits on the diagonal of the quadratic, so changing it invalidates
cached factorizations — sequences therefore treat it as structure, not
data.

## Exact L1 transaction costs

`l1_turnover_costs` (Python) / `with_l1_turnover` (Rust) adds proportional
costs \\(\sum_i c_i \lvert w_i - w_{0,i} \rvert\\) — a scalar broadcasts to
all assets. This is the term with a genuine **no-trade region**: assets
whose expected-return edge does not cover the round-trip cost stay exactly
at the anchor, machine-exact.

Implementation matters here. The standard epigraph reformulation adds `n`
auxiliary variables and `2n` inequality rows, which destroys the
factor-structure advantage. Ledge instead handles the term as a dedicated
soft-threshold proximal block inside ADMM: the reduced factorization keeps
its `factors + constraints` dimension, and rolling sequences move the
anchor in `O(n)` without refactorizing.

The L1 multiplier is a first-class dual: the independent KKT audit scores
the subgradient conditions (\\(\lvert\lambda_i\rvert \le c_i\\), with
\\(\lambda_i = \pm c_i\\) on the trade sign when trading), polishing pins
no-trade assets at the anchor, and warm starts carry the L1 dual. The
implementation is validated against epigraph reformulations in Rust
(including property tests) and against cvxpy+Clarabel in Python.

L2 and L1 can be combined; both use the same `previous_weights` anchor.

## Tracking error

`benchmark_weights` (Python) / `with_tracking_benchmark` (Rust) turns the
risk term into active risk
\\(\tfrac{\lambda}{2}(w-b)^\mathsf{T}\Sigma(w-b)\\). Expanding the square
shows this is a pure linear-cost shift \\(-\lambda\Sigma b\\), computed
through the factor structure — the same QP underneath, no solver changes.
The constant \\(\tfrac{\lambda}{2}b^\mathsf{T}\Sigma b\\) is dropped from
reported objectives, as is conventional.

Because it is a linear-cost shift, sequences roll the benchmark
date-by-date (`benchmark_weights` in `solve_next`) with cached
factorizations intact. Industry-neutrality templates derive their targets
from the benchmark when one is present.

## Hard turnover caps

Ledge prices turnover in the objective; it does not model a hard cap
`norm1(w - w_prev) <= tau` (that needs a variable split). If your process
requires cap semantics, keep cvxpy for those dates or tune the cost until
realized turnover sits where the cap did — see
[Migrating from cvxpy](migration.md).
