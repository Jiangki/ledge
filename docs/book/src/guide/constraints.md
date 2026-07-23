# Constraints and templates

## Native constraint blocks

Every problem has, in QP terms:

- one **budget equality** `sum(w) = budget`;
- per-asset **boxes** `lower <= w <= upper`;
- optional **equality rows** `A_eq @ w = b_eq`;
- optional **upper inequality rows** `A_in @ w <= b_in`.

Rows are cheap but not free: the reduced factorization Ledge solves each
iteration has dimension `r = factors + explicit rows`, so a formulation
with `O(n)` explicit rows forfeits the factor-structure advantage. Boxes
are handled by a dedicated projection and never grow `r`.

## Templates

Template builders compile portfolio vocabulary onto those blocks, with
eager validation (empty groups, crossing bands, caps contradicting existing
bounds fail at build time):

| Template (Rust / Python) | Compiles to |
|---|---|
| `with_industry_neutrality` / `industry_ids=` | one equality row per industry, targets from the tracking benchmark |
| `with_group_targets` / `industry_ids=` + `industry_targets=` | one equality row per group with explicit targets |
| `with_style_bounds` / `style_matrix=`, `style_lower=`, `style_upper=` | inequality rows for the finite sides of each exposure band; exact bands collapse to one equality row |
| `with_concentration_limit` / `max_weight=` | box tightening `|w_i| <= cap` — **no new rows** |
| `with_short_limit` / `max_short=` | box tightening `w_i >= -limit`; `0` = long-only — **no new rows** |

Templates **append** to the user constraint blocks; `with_equalities` /
`with_inequalities` have replace semantics, so call those first. Appended
rows become ordinary user rows: rolling sequences move industry or style
targets date-by-date through `equality_rhs` / `inequality_rhs` step updates
with cached factorizations intact.

## Python example

```python
problem = PortfolioProblem(
    F, omega, d, mu,
    risk_aversion=8.0,
    lower_bounds=np.zeros(n),
    upper_bounds=np.full(n, 0.05),
    benchmark_weights=benchmark,
    industry_ids=industry_ids,        # industry-neutral vs the benchmark
    style_matrix=style_exposures,     # k_style x n
    style_lower=-0.1 * np.ones(k_style),
    style_upper=+0.1 * np.ones(k_style),
    max_weight=0.03,                  # concentration cap, box-only
    max_short=0.0,                    # long-only, box-only
)
```

A total short-budget cap (`sum(max(-w, 0)) <= S`) is deliberately not a
template: it needs a long/short variable split the QP form does not model.
