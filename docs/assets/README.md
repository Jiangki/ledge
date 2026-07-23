# Documentation assets

Visuals are source-controlled so the README and docs render without an
external image host.

| Asset | Source / meaning |
|---|---|
| `factor-structure.svg` | Conceptual factor covariance layout; hand-authored SVG |
| `architecture.svg` | Current Python/Rust/core layers; generated |
| `rolling-rebalance.svg` | Sequence/cache flow plus the seeded rolling-example measurements; generated |
| `l1-rolling-comparison.svg` | Parsed from `benchmarks/results/2026-07-l1/summary.md`; generated |
| `smoke-timings.svg` | Self-timing smoke summary; hand-authored SVG |
| `terminal-demo.gif` | Real seeded release-build output from the command in `terminal-demo.json`; generated with Pillow |

Regenerate the deterministic diagrams and GIF:

```bash
python scripts/generate_demo_assets.py
python scripts/generate_demo_assets.py --check
```

Refresh the terminal capture by actually running the Rust example, then
render all assets:

```bash
python scripts/generate_demo_assets.py --capture-terminal
```

The static SVG path uses only the Python standard library. GIF rendering
requires Pillow. The script also writes the three mdBook copies under
`docs/book/src/assets/`; those copies are generated output, not a second
source of truth.

Rules:

- Never turn a timing into an illustration by hand. Parse a committed report
  or capture a documented command.
- Keep machine/commit/protocol caveats in the surrounding text.
- The terminal GIF demonstrates output; its wall time is machine-dependent
  and is not comparative evidence.
- Prefer SVG for diagrams and charts; use GIF only where motion explains a
  sequence.
