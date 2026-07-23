# Repository security and open-source audit

Audit date: 2026-07-23

Reviewed base: `9a68cf99fcbf667a27b415522f36442073e50874`

Scope: public Git refs, tracked source and assets, package metadata,
dependencies, release archives/configuration, documentation, and GitHub
Actions workflows.

## Executive conclusion

No credential, customer data, private dependency, private repository URL, or
proprietary implementation was found in the publicly reachable repository
history or current tree.

The public remote exposed only `main` and annotated tag `v0.2.0` during this
review. Deleted pre-release branch names seen in an existing local clone were
confirmed absent from the remote, no pull-request refs were exposed, and a
pre-release commit ID was not readable through the public GitHub API. A normal
push of a branch based on the clean public root does not transfer unrelated
local objects.

The repository does intentionally document its generic open-core boundary and
the existence of a separately retained historical archive. Those documents do
not contain an archive name/URL, customer identity, credential, or private
package coordinate. An obsolete former repository name and private-state
transition wording in `CHANGELOG.md` were unnecessary public detail and were
removed during this audit.

## Method

- Enumerated tracked files, Git authors, local and remote refs, tags, public
  branches, pull requests, and recent CI runs.
- Scanned the working tree and public history with Gitleaks `8.30.1`, plus
  targeted searches for tokens, keys, credentials, email addresses, private
  hosts/IPs, absolute workstation paths, private package coordinates, and
  commercial/customer markers.
- Ran `scripts/check_open_core.sh --release`.
- Inspected Cargo/Python metadata, lockfile sources, packaged legal notices,
  generated-asset provenance, and registry availability.
- Audited all 163 locked Rust dependencies with `cargo-audit 0.22.2` and the
  current RustSec database.
- Audited the published Python package and minimum declared NumPy runtime
  (`ledge-portfolio==0.2.0`, `numpy==1.24.0`) with `pip-audit 2.10.1`.
- Reviewed workflow permissions, event triggers, input handling, artifact
  flow, external downloads, and third-party Action references.
- Ran the repository's Rust, Python, documentation, asset, attribution, and
  packaging checks listed below.

## Findings and remediation

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| A-01 | Medium | Workflows referenced mutable Action tags; mdBook was streamed from the network directly into `tar` without an integrity check. | Third-party Actions are pinned to full commit SHAs. The mdBook archive is downloaded with fail-closed `curl`, checked against the release SHA-256 digest, then extracted. |
| A-02 | Medium | CI had no explicit top-level token permission, so behavior depended on repository defaults. | CI now declares `permissions: contents: read`; publishing/deployment jobs retain only their narrowly required job permissions. |
| A-03 | Low | A dispatch input was interpolated directly into shell source in the release preflight. Dispatch is write-restricted, but interpolation is avoidable. | The value is passed through the step environment and referenced as a quoted shell variable. |
| A-04 | Low | README, Python package docs, roadmap, and release-state copy still said artifacts or public deployment were pending although `0.2.0` was live. | Updated to the verified PyPI, crates.io, docs, tag, and clean-root status. |
| A-05 | Low | `CHANGELOG.md` exposed an obsolete former repository name and private-state transition details that were not useful to public users. | Removed the identifying/stale transition text; the generic public boundary documentation remains. |
| A-06 | Low | RustSec reports `atomic-polyfill 1.0.3` as unmaintained. It is target-specific and reachable only through the `postcard` dev-dependency (`postcard -> heapless -> atomic-polyfill`), not the published runtime graph. | No forced replacement: there is no known vulnerability and `postcard` is used only for serialization tests. Track upstream and remove/replace it if the warning reaches a runtime target. |
| A-07 | Informational | Gitleaks' generic-key heuristic treated a historical compatibility-roadmap phrase as a secret. Manual review confirmed plain prose. | Added a path- and exact-text-scoped allowlist; default Gitleaks rules remain enabled. |
| A-08 | Informational | GitHub rejected read-only API queries for repository-wide Actions/security settings with the current integration identity. | Branch protection was observable and enabled; secret scanning, push protection, environment approvers, and default token settings still require an owner/admin check. |

No known Rust or Python vulnerability was reported. Cargo sources resolve from
crates.io only; the Python runtime resolves from PyPI; workspace `path`
dependencies are local public crates and no private Git/registry dependency is
present.

## Additional hardening applied

- Added weekly Dependabot checks for Cargo, Python, and GitHub Actions.
- Added a repository Gitleaks configuration that extends the default rules.
- Replaced unsupported operator-name macros throughout public Markdown math,
  not only the first README formula.
- Added the package/docs status, limitations, platform coverage, and security
  posture to the README so users do not infer production guarantees.

## Verification

The completed change was checked with:

```text
gitleaks dir .
gitleaks git . --log-opts=origin/main
./scripts/check_open_core.sh --release
cargo audit
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo package -p ledge-core --allow-dirty
cargo package -p ledge-portfolio --allow-dirty --list
python -m pytest python/tests -q
python scripts/generate_demo_assets.py --check
./scripts/generate_third_party_licenses.sh --check
mdbook build docs/book
maturin sdist
actionlint
lychee README.md docs/SECURITY_AUDIT.md
```

## Residual owner checks

Repository administrators should confirm that GitHub secret scanning and push
protection, private vulnerability reporting, Dependabot alerts, Actions
default read permissions, and release-environment approvers are enabled.
These settings are not stored in Git and could not all be read by the audit
identity.

This review does not establish copyright ownership, assess unpublished
systems, or make Ledge safe for hostile multi-tenant inputs. Security reports
should follow [`../SECURITY.md`](../SECURITY.md).
