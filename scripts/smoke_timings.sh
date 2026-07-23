#!/usr/bin/env bash
# Reproduce the README smoke table. Not a competitive benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "rustc: $(rustc --version)"
echo "host: $(uname -srm)"
echo
for args in "--n 100 --k 5 --seed 1" "--n 500 --k 10 --seed 42" "--n 1000 --k 20 --seed 7" "--n 2000 --k 50 --seed 3"; do
  cargo run -q -p ledge-portfolio --release --example synthetic -- $args
done
