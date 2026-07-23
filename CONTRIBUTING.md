# Contributing to Ledge

Ledge uses an Apache-2.0 **inbound=outbound, no-CLA** contribution policy.
By intentionally submitting a contribution for inclusion, you agree that it
is licensed under Apache-2.0 as described in `LICENSE`, and that you have the
right to submit it. No Contributor License Agreement is required.

Keep changes narrow and include the failure mode they address.

## Before you open a PR

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If you touch the Python binding:

```bash
python -m pip install -e python/
python -m pytest python/tests
python python/examples/rebalance.py
```

## Guidelines

- Solver changes need a deterministic test with a known solution or KKT
  threshold.
- Do **not** commit hand-picked “beats X” timings. Self-timing smoke updates
  belong in `docs/SMOKE_TIMINGS.md` with machine + commit metadata. Cross-solver
  claims must follow `benchmarks/README.md`.
- New public APIs need rustdoc and must preserve sign conventions in
  `docs/algorithm.md`.
- Avoid `unsafe`, unrelated formatting churn, and dependencies that download
  large native artifacts.
- Prefer small, reviewable PRs with a clear problem statement.

## Scope

In scope: factor-model continuous convex portfolio QPs, warm starts, Python
ergonomics, scaling/robustness, documentation.

Out of scope for drive-by PRs: mixed-integer programming, GPU ports, or
rewriting the project into a general-purpose solver.

Public technical direction: [`docs/PLAN.md`](docs/PLAN.md).
Current priorities and exit criteria: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Recorded decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md).
Repository boundary and public-release runbook:
[`docs/OPEN_CORE.md`](docs/OPEN_CORE.md).
