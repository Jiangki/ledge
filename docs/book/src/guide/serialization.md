# Saving and replaying problems

Bug reports and regression tests need a lossless way to capture exactly
what the solver saw and returned. Ledge ships this behind the non-default
`serde` cargo feature (enabled in the Python wheel).

## Python

```python
dump = problem.to_json()          # lossless problem dump
restored = PortfolioProblem.from_json(dump)

result = problem.solve()
report = result.to_json()         # status, weights, duals, residuals,
                                  # diagnostics, certificates
```

Attach the two JSON strings to a bug report and the maintainer can replay
your exact solve — round-trips are **bit-exact**, so the replayed problem
produces the identical iterate path.

## Rust

`QpProblem`, `PortfolioProblem`, `SolverSettings`, `WarmStart`, and
`Solution` implement `Serialize`/`Deserialize` with any serde format:

```rust,ignore
let json = serde_json::to_string(&problem)?;
let restored: PortfolioProblem = serde_json::from_str(&json)?;
```

Notes:

- **Validation cannot be bypassed.** Matrices rebuild through their
  constructors and `PortfolioProblem` replays its builder methods, so a
  tampered dump fails with the same errors as wrong constructor input.
- **JSON and infinities.** Unbounded box sides travel as `null`
  (`Option<f64>` per entry) because JSON has no representation for
  infinities. Any self-describing binary serde format works unchanged.
- **Bit-exact JSON parsing** requires `serde_json`'s `float_roundtrip`
  feature (default parsing may be off by 1 ULP).
- Solutions from `NumericalFailure` contain non-finite iterates and only
  round-trip through binary formats.
