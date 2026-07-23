#!/usr/bin/env bash
# Enforces docs/OPEN_CORE.md.
#
#   ./scripts/check_open_core.sh            advisory while private
#   ./scripts/check_open_core.sh --release  strict public-release gate
#
# Exit status:
#   0  no blocker for the selected mode
#   1  a blocker was found
#   2  invalid command-line arguments
set -uo pipefail
cd "$(dirname "$0")/.."

release=0
case "${1:-}" in
  "") ;;
  --release) release=1 ;;
  -h|--help)
    sed -n '1,10p' "$0"
    exit 0
    ;;
  *)
    printf 'usage: %s [--release]\n' "$0" >&2
    exit 2
    ;;
esac

have_rg=1
command -v rg >/dev/null 2>&1 || have_rg=0

pass=0 warn=0 fail=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
note() { printf '  \033[33mWARN\033[0m %s\n' "$1"; warn=$((warn+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
gate() {
  if [ "$release" -eq 1 ]; then
    bad "$1"
  else
    note "$1"
  fi
}

# Search tracked files only, excluding this script and the manifest that
# necessarily mention the markers being checked.
scan() { # scan <regex>; prints matching "path:line:text"
  local re="$1"
  local excl=':(exclude)scripts/check_open_core.sh :(exclude)docs/OPEN_CORE.md'
  if [ "$have_rg" -eq 1 ]; then
    git ls-files -- . $excl | tr '\n' '\0' \
      | xargs -0 rg -n --no-heading -e "$re" 2>/dev/null
  else
    git ls-files -- . $excl | tr '\n' '\0' \
      | xargs -0 grep -nEI -e "$re" 2>/dev/null
  fi
}

if [ "$release" -eq 1 ]; then
  echo "== Open-core boundary check (STRICT RELEASE MODE) =="
else
  echo "== Open-core boundary check (advisory pre-release mode) =="
fi
[ "$have_rg" -eq 1 ] || echo "  (ripgrep not found; using grep fallback)"
echo

# ---------------------------------------------------------------------------
echo "1. Secrets / credentials (HARD blocker)"
secret_re='(-----BEGIN [A-Z ]*PRIVATE KEY-----|x-access-token:[^@[:space:]]+@|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|(secret|token|passwd|password|api_key)[[:space:]]*=[[:space:]]*["'"'"'][^"'"'"']{12,})'
hits="$(scan "$secret_re")"
if [ -n "$hits" ]; then
  bad "possible secrets found:"
  printf '%s\n' "$hits" | sed 's/^/       /'
else
  ok "no secret-shaped strings in tracked files"
fi
echo

# ---------------------------------------------------------------------------
echo "2. External/private-surface code leaked into this tree (HARD blocker)"
# Real implementation modules named like the external surface, not prose.
private_paths="$(git ls-files -- 'crates/**' 'python/src/**' \
  | grep -Ei 'ledge[-_]pro|(persistent[_-]?service|audit[_-]?report|checkpoint[_-]?resume)\.rs' || true)"
if [ -n "$private_paths" ]; then
  bad "files that look like the external/private layer live in this repo:"
  printf '%s\n' "$private_paths" | sed 's/^/       /'
  echo "       -> these belong in a separate private repository (OPEN_CORE.md §2)"
else
  ok "no external/private-surface implementation files in crates/ or python/src/"
fi
echo

# ---------------------------------------------------------------------------
echo "3. License and public metadata"
if grep -qE '^license = "Apache-2\.0"$' Cargo.toml 2>/dev/null; then
  ok "Cargo workspace license is Apache-2.0"
else
  gate "Cargo workspace license is not Apache-2.0"
fi
if grep -q 'Apache License' LICENSE 2>/dev/null \
    && grep -q 'Version 2.0, January 2004' LICENSE 2>/dev/null; then
  ok "LICENSE contains the full Apache-2.0 text marker"
else
  gate "LICENSE is not the Apache-2.0 license text"
fi
proprietary_metadata="$(grep -Ein 'proprietary|private :: do not upload' \
  NOTICE README.md CONTRIBUTING.md python/pyproject.toml 2>/dev/null || true)"
if [ -n "$proprietary_metadata" ]; then
  gate "public-facing metadata still contains proprietary/private labels"
  [ "$release" -eq 0 ] || printf '%s\n' "$proprietary_metadata" | sed 's/^/       /'
else
  ok "NOTICE, README, CONTRIBUTING, and Python metadata have no proprietary labels"
fi
if grep -qE '^license = "Apache-2\.0"$' python/pyproject.toml 2>/dev/null \
    && grep -q 'License :: OSI Approved :: Apache Software License' python/pyproject.toml 2>/dev/null; then
  ok "Python package metadata declares Apache-2.0"
else
  gate "Python package metadata is not ready for Apache-2.0 publication"
fi
package_legal_missing=""
for dir in crates/ledge-core crates/ledge python; do
  for file in LICENSE NOTICE; do
    if [ ! -f "$dir/$file" ] || ! cmp -s "$file" "$dir/$file"; then
      package_legal_missing="${package_legal_missing}${dir}/${file}"$'\n'
    fi
  done
done
if [ -z "$package_legal_missing" ]; then
  ok "published Rust/Python package roots contain current LICENSE and NOTICE"
else
  gate "published package roots need copies of the current LICENSE and NOTICE"
  [ "$release" -eq 0 ] || printf '%s' "$package_legal_missing" | sed 's/^/       /'
fi
python_license_files="$(sed -n 's/^license-files = \(.*\)$/\1/p' python/pyproject.toml)"
if printf '%s' "$python_license_files" | grep -q 'LICENSE' \
    && printf '%s' "$python_license_files" | grep -q 'NOTICE' \
    && printf '%s' "$python_license_files" | grep -q 'THIRD_PARTY_LICENSES.html' \
    && [ -s python/THIRD_PARTY_LICENSES.html ]; then
  ok "Python distributions include project and third-party license notices"
else
  gate "Python license-files must include LICENSE, NOTICE, and the generated third-party report"
fi
if grep -q 'inbound=outbound' CONTRIBUTING.md 2>/dev/null \
    && grep -q 'no-CLA' CONTRIBUTING.md 2>/dev/null; then
  ok "CONTRIBUTING records the intended inbound=outbound, no-CLA policy"
else
  gate "CONTRIBUTING must record the Apache-2.0 inbound=outbound, no-CLA policy"
fi
stale_status_re='prepared, not released|pre-release legal (notice|status)|pre-release status|this repository is (currently|still) private|^repository is currently private|ordinary private repository issues|wheels are not published yet|not completed that gate yet'
stale_status="$(grep -Ein "$stale_status_re" \
  README.md CONTRIBUTING.md SECURITY.md python/README.md \
  docs/OPEN_CORE.md docs/PLAN.md docs/book/src/introduction.md 2>/dev/null || true)"
if [ -n "$stale_status" ]; then
  gate "reader-facing files still describe the repository/packages as pre-release or private"
  [ "$release" -eq 0 ] || printf '%s\n' "$stale_status" | sed 's/^/       /'
else
  ok "reader-facing release status is current"
fi
echo

# ---------------------------------------------------------------------------
echo "4. Crate publishability"
workspace_version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | sed -n '1p')"
python_version="$(sed -n 's/^version = "\(.*\)"$/\1/p' python/pyproject.toml | sed -n '1p')"
if [ -n "$workspace_version" ] && [ "$workspace_version" = "$python_version" ]; then
  ok "Cargo and Python versions agree ($workspace_version)"
else
  bad "Cargo ($workspace_version) and Python ($python_version) versions differ or are missing"
fi
dependency_versions_ok=1
for file in crates/ledge/Cargo.toml benchmarks/adapters/Cargo.toml; do
  if ! grep -qE "ledge-core = .*version = \"$workspace_version\"" "$file" 2>/dev/null; then
    bad "$file does not require ledge-core version $workspace_version"
    dependency_versions_ok=0
  fi
done
[ "$dependency_versions_ok" -eq 0 ] || ok "Rust path-dependency versions match the workspace version"
if grep -qF "## [$workspace_version] - " CHANGELOG.md 2>/dev/null; then
  ok "CHANGELOG has a dated section for $workspace_version"
else
  gate "CHANGELOG has no dated release section for $workspace_version"
fi
if grep -qE '^publish = true$' Cargo.toml 2>/dev/null; then
  ok "workspace crates are publishable by default"
else
  gate "workspace publish flag is not true"
fi
for file in python/Cargo.toml benchmarks/adapters/Cargo.toml; do
  if grep -qE '^publish = false$' "$file" 2>/dev/null; then
    ok "$file is explicitly excluded from crates.io"
  else
    gate "$file must set 'publish = false' when the workspace becomes publishable"
  fi
done
for file in crates/ledge-core/Cargo.toml crates/ledge/Cargo.toml; do
  if grep -qE '^publish\.workspace = true$' "$file" 2>/dev/null; then
    ok "$file inherits the publishable workspace setting"
  else
    bad "$file does not inherit the workspace publish setting"
  fi
done
portfolio_package="$(sed -n 's/^name = "\(.*\)"$/\1/p' crates/ledge/Cargo.toml | sed -n '1p')"
if [ "$portfolio_package" = "ledge" ]; then
  gate "portfolio crate still uses crates.io's occupied 'ledge' package name; select another name or document legitimate ownership and adjust this gate"
elif [ -n "$portfolio_package" ]; then
  ok "portfolio crate uses selected registry package name '$portfolio_package'"
  stale_package_refs="$(git grep -nE -- '-p ledge([^[:alnum:]_-]|$)' \
    -- README.md CONTRIBUTING.md docs .github scripts \
    ':(exclude)scripts/check_open_core.sh' 2>/dev/null || true)"
  if [ -n "$stale_package_refs" ]; then
    gate "commands still address the old Cargo package name '-p ledge'"
    [ "$release" -eq 0 ] || printf '%s\n' "$stale_package_refs" | sed 's/^/       /'
  else
    ok "documented Cargo commands use the selected portfolio package name"
  fi
else
  bad "crates/ledge/Cargo.toml has no package name"
fi
echo

# ---------------------------------------------------------------------------
echo "5. Public strategy-document boundary"
strategy_detail_re='(\$[0-9]|[0-9]+k[[:space:]]*[-–][[:space:]]*\$?[0-9]+k|[0-9]+[[:space:]]*[-–][[:space:]]*[0-9]+k[[:space:]]+ARR|pricing (range|starting)|revenue (target|expectation)|pilot customer)'
strategy_hits="$(grep -Ein "$strategy_detail_re" \
  docs/PLAN.md docs/ROADMAP.md docs/DECISIONS.md 2>/dev/null || true)"
if [ -n "$strategy_hits" ]; then
  bad "commercial-detail markers remain in public strategy documents:"
  printf '%s\n' "$strategy_hits" | sed 's/^/       /'
else
  ok "public plan/roadmap/decisions contain no commercial-detail markers"
fi
echo

# ---------------------------------------------------------------------------
echo "6. Release automation and documentation assets"
if [ -f .github/workflows/release.yml ]; then
  ok "manual release workflow is present"
else
  gate ".github/workflows/release.yml is missing"
fi
if [ -f docs/PUBLIC_RELEASE_CHECKLIST.md ]; then
  ok "maintainer public-release sign-off checklist is present"
else
  gate "docs/PUBLIC_RELEASE_CHECKLIST.md is missing"
fi
missing_assets=""
while IFS= read -r asset; do
  case "$asset" in
    http://*|https://*|"") continue ;;
  esac
  if [ ! -f "$asset" ]; then
    missing_assets="${missing_assets}${asset}"$'\n'
  fi
done < <(sed -n 's/.*<img[^>]*src="\([^"]*\)".*/\1/p' README.md)
if [ -n "$missing_assets" ]; then
  bad "README references missing local image assets:"
  printf '%s' "$missing_assets" | sed 's/^/       /'
else
  ok "all README local image assets exist"
fi
echo

# ---------------------------------------------------------------------------
echo "== Summary =="
printf '  %d pass, %d warn, %d fail\n' "$pass" "$warn" "$fail"
if [ "$fail" -gt 0 ]; then
  echo "  Blockers present — do NOT publish until resolved."
  exit 1
fi
if [ "$warn" -gt 0 ]; then
  echo "  No hard blockers in advisory mode. Clear every warning at the"
  echo "  roadmap 1.6 gate, then rerun with --release."
else
  echo "  Selected-mode checks passed."
fi
exit 0
