#!/usr/bin/env bash
# Generate the Rust dependency notices bundled in Python wheels.
#
#   ./scripts/generate_third_party_licenses.sh
#   ./scripts/generate_third_party_licenses.sh --check
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  "") check=0 ;;
  --check) check=1 ;;
  -h|--help)
    sed -n '1,7p' "$0"
    exit 0
    ;;
  *)
    printf 'usage: %s [--check]\n' "$0" >&2
    exit 2
    ;;
esac

command -v cargo-about >/dev/null 2>&1 || {
  echo "cargo-about is required; install its latest CLI with:" >&2
  echo "  rustup run stable cargo install cargo-about --features cli" >&2
  exit 1
}

output="python/THIRD_PARTY_LICENSES.html"
temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT
cargo about generate --manifest-path python/Cargo.toml about.hbs > "$temporary"
python - "$temporary" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
while lines and not lines[-1]:
    lines.pop()
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [ "$check" -eq 1 ]; then
  cmp -s "$temporary" "$output" || {
    echo "$output is stale; run ./scripts/generate_third_party_licenses.sh" >&2
    exit 1
  }
  echo "$output is current"
else
  mv "$temporary" "$output"
  trap - EXIT
  echo "wrote $output"
fi
