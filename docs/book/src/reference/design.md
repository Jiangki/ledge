# Design notes and roadmap

The canonical engineering documents live in the repository and stay there
(single source of truth); this page is the map.

| Document | Content |
|---|---|
| [`docs/PLAN.md`](https://github.com/Jiangki/ledge/blob/main/docs/PLAN.md) | public technical plan: scope, architecture, quality, release policy |
| [`docs/ROADMAP.md`](https://github.com/Jiangki/ledge/blob/main/docs/ROADMAP.md) | executable milestones M0–M4 with exit criteria and technical notes |
| [`docs/DECISIONS.md`](https://github.com/Jiangki/ledge/blob/main/docs/DECISIONS.md) | decision log (ADR-style): what was chosen, why, what was rejected |
| [`docs/OPEN_CORE.md`](https://github.com/Jiangki/ledge/blob/main/docs/OPEN_CORE.md) | authoritative repository boundary and public-release runbook |
| [`docs/algorithm.md`](https://github.com/Jiangki/ledge/blob/main/docs/algorithm.md) | solver math: ADMM splitting, SMW reduction, scaling, certificates, polishing, L1 prox |
| [`docs/factor_structure_note.md`](https://github.com/Jiangki/ledge/blob/main/docs/factor_structure_note.md) | short technical note: why factor structure is worth exploiting, with measured evidence |
| [`docs/cvxpy_migration.md`](https://github.com/Jiangki/ledge/blob/main/docs/cvxpy_migration.md) | cvxpy → Ledge mappings, executed in CI |
| [`docs/SMOKE_TIMINGS.md`](https://github.com/Jiangki/ledge/blob/main/docs/SMOKE_TIMINGS.md) | self-timing smoke matrix numbers |

## Design pillars (short form)

1. **Never form Σ.** The ADMM x-update solves through a Sherman–
   Morrison–Woodbury reduction of dimension `r = factors + explicit rows`;
   every feature (scaling, polishing, L1, templates) is designed to keep
   `r` small.
2. **Trust before speed.** Residuals are audited independently on original
   data; polish is adopted only when the audit improves; infeasibility
   claims carry checkable certificates; benchmark claims follow a written
   protocol with raw samples.
3. **Rolling is the product.** Warm starts, the workspace factorization
   cache, sequences, and batch exist because rebalancing is a repeated
   solve, not a one-shot.
4. **Small dependency surface.** Default builds have no BLAS/LAPACK, no
   native dependencies; `serde` and `rayon` are opt-in features.

## Milestone status

M0 (foundations), M1 engineering (scaling, over-relaxation, comparisons),
M2 (L1, certificates, polishing, workspace, sequences, tracking), and the
M3 engineering items (templates, batch, serialization, this docs site) are
complete; the public `0.2.0` crates, Python wheels, and documentation site
are live. Remaining roadmap items are adoption- and decision-driven:
external real workloads and the 1.0 compatibility review. Sparse `F`
support is deliberately deferred until a real workload needs it.
