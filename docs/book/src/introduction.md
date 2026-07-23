# Introduction

**Ledge is a rebalancing execution engine for factor risk models**, written in
Rust with first-class Python bindings.

It solves continuous convex mean-variance portfolio QPs whose covariance has
factor structure

\\[
\Sigma = F \Omega F^\mathsf{T} + \operatorname{diag}(d)
\\]

**without ever forming the dense \\(n \times n\\) matrix**. Inputs are the
factor loadings \\(F\\), the factor covariance \\(\Omega\\), idiosyncratic
variances \\(d\\), expected returns \\(\mu\\), constraints, and previous
weights. Outputs are auditable weights (independent KKT residuals), duals,
and diagnostics.

## What it covers

- Mean-variance objectives with budget, weight boxes, linear equality and
  inequality constraints.
- Smooth L2 turnover and **exact L1 proportional transaction costs** with a
  machine-exact no-trade region.
- Tracking-error objectives against a benchmark.
- Constraint templates: industry neutrality, group targets, style bands,
  concentration and short limits.
- Rolling sequences with automatic warm starts and factorization reuse, and
  multi-threaded batch over many accounts.
- Trust machinery on by default: automatic scaling, audit-gated solution
  polishing, infeasibility certificates, and independent KKT checks on the
  original (unscaled) data.

## What it deliberately does not cover

No mixed-integer constraints, no SOCP or general NLP, no GPU/distributed
execution, and no alpha or risk-model estimation — \\(F\\), \\(\Omega\\),
\\(d\\) are inputs. If you need a general modeling language, keep cvxpy;
[migrate only the rebalancing QP](guide/migration.md).

## Declared envelope

Default settings are tested to converge on continuous convex factor QPs with
roughly \\(n \le 5000\\) assets, \\(k \le 100\\) factors, and \\(m \le 200\\)
explicit linear constraint rows. Outside that envelope Ledge may still work,
but the project makes no default-settings promise.

## Status

Alpha (`0.2.x`). APIs and defaults may change before 1.0; every change is
recorded in the repository
[CHANGELOG](https://github.com/Jiangki/ledge/blob/main/CHANGELOG.md). Source is
licensed Apache-2.0; crates.io/PyPI availability is verified separately for
each tagged release on the [roadmap](reference/design.md).
