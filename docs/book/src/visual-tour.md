# Visual tour

## From APIs to the numerical core

![Python and Rust entry points share the portfolio layer and ledge-core kernel](assets/architecture.svg)

Python and Rust callers use the same portfolio semantics. The portfolio layer
compiles budget, boxes, exposure templates, turnover, and tracking error into
the factor-structured kernel. `ledge-core` performs scaling, ADMM/SMW updates,
the exact L1 proximal step, polishing, and independent audits without forming
the dense covariance matrix.

## One model, many rebalance dates

![A rolling sequence updates data while reusing structure and warm starts](assets/rolling-rebalance.svg)

A `PortfolioSequence` holds fixed covariance and constraint structure. Each
date updates expected returns and holdings, then reuses cached reduced
factorizations and the preceding primal/dual solution. The annotation comes
from the repository's seeded 300-asset, 24-date example; run
`python docs/examples/rolling_backtest.py` to reproduce it.

## Measured comparison, not a marketing sketch

![Published L1 rolling comparison on a logarithmic milliseconds-per-step scale](assets/l1-rolling-comparison.svg)

The bars are parsed directly from the committed
`benchmarks/results/2026-07-l1/summary.md`: ten repeats with ten rolling steps
per repeat. Ledge uses its native L1 proximal block; OSQP and Clarabel receive
factor-lifted epigraph formulations. The chart is deliberately logarithmic
because the range spans four orders of magnitude.

The full report includes setup, cold, rolling, status, iteration, and
independently audited residual data. Read the [benchmark evidence
chapter](reference/benchmarks.md) before quoting a result.

## Reproduce the visuals

From the repository root:

```bash
python scripts/generate_demo_assets.py --check
python scripts/generate_demo_assets.py
```

The script uses the standard library for SVGs and Pillow for the README's
terminal GIF. Provenance and refresh instructions live in
`docs/assets/README.md`.
