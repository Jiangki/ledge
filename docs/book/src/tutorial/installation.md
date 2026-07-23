# Installation

Requirements: Rust 1.83+; Python 3.9+ for the bindings. There are no native
dependencies (no BLAS/LAPACK) in the default build.

## Rust

```toml
[dependencies]
ledge = { package = "ledge-portfolio", version = "0.2" }
```

Optional cargo features on `ledge` / `ledge-core`:

| Feature | Adds |
|---|---|
| `serde` | `Serialize`/`Deserialize` for problems, settings, warm starts, and solutions |
| `rayon` | multi-threaded `solve_batch` over the account axis (same API and identical results without it) |

## Python

Install the `0.2.0` registry artifact after it is published:

```bash
python -m pip install ledge-portfolio==0.2.0
```

Or build from source with [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/Jiangki/ledge.git
cd ledge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install -e python/
python python/examples/rebalance.py
```

The Python package is named `ledge-portfolio` and imports as `ledge`. The
only runtime dependency is NumPy. The wheel enables the `serde` and `rayon`
features, so `to_json` / `from_json` and parallel `solve_batch` work out of
the box.

## Verify the installation

```bash
cargo test --workspace          # Rust test suite
python -m pip install -e "python/[test]"
python -m pytest python/tests -q   # includes cvxpy+Clarabel gold tests
```
