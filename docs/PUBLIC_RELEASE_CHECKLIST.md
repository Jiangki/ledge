# Public release sign-off checklist

This is the operational companion to [`OPEN_CORE.md`](OPEN_CORE.md), which
remains authoritative for the repository boundary and release procedure.
Copy this checklist into a **private maintainer issue** for each release and
record links or commit IDs as evidence. Do not check boxes speculatively.

Release candidate: `________________`
Target version/tag: `0.2.0` / `v0.2.0`
Private-archive gate commit SHA: `________________`
Gate tree ID: `________________`
Public clean-root commit SHA: `________________`
Maintainer approving legal/visibility changes: `________________`

Approved project defaults: Apache-2.0; clean-root public history with this
repository retained as a private archive; Rust packages `ledge-core` and
`ledge-portfolio` with library name `ledge`; Python distribution/import
`ledge-portfolio` / `ledge`.

## 1. Irreversible decisions

- [ ] Confirm the copyright owner has the right to license every first-party
      file and generated asset under the selected OSS license.
- [ ] Confirm the approved license choice: `Apache-2.0`.
- [ ] Confirm the approved history model: `clean root`.
- [ ] Review author names/emails and all historical commercial text under the
      selected history model.
- [ ] Confirm the Rust registry package names:
      core `ledge-core`, portfolio `ledge-portfolio`, Rust library `ledge`.
      The unrelated crates.io package `ledge` is not used.
- [ ] Confirm the Python distribution/import names:
      `ledge-portfolio` / `ledge`.
- [ ] Recheck all registry names immediately before the gate. Search results
      do not reserve a name.
- [ ] Record whether release automation actions will use moving major tags or
      reviewed immutable commit SHAs.

## 2. History and sensitive-data review

- [ ] Run a dedicated full-history secret scanner, not only the current-tree
      check. For example:

  ```bash
  gitleaks git . --redact --log-opts="--all"
  git log --all --format='%h %an <%ae> %s'
  ```

- [ ] Review the full tree at the gate commit for credentials, customer data,
      internal URLs, proprietary datasets, generated dumps, and non-public
      strategy:

  ```bash
  ./scripts/check_open_core.sh
  git ls-files
  git status --short
  ```

- [ ] Confirm the existing history will not be published; it contains removed
      commercial planning and the former proprietary-license era.
- [ ] Preserve this repository as a private archive and verify the new
      clean-root public repository contains only the reviewed release tree.

## 3. Release-gate commit

- [ ] Replace `LICENSE` with the complete, unmodified selected OSS license.
- [ ] Replace `NOTICE` with reviewed project attribution and remove
      proprietary wording.
- [ ] Copy the reviewed `LICENSE` and `NOTICE` into `crates/ledge-core/`,
      `crates/ledge/`, and `python/`; workspace-root legal files are not
      automatically included in member crate archives.
- [ ] Update the workspace version, SPDX license, and publish default in
      `Cargo.toml`.
- [ ] Resolve `crates/ledge/Cargo.toml`'s occupied registry package name;
      preserve `[lib] name = "ledge"` if the public Rust import should remain
      `use ledge::...`.
- [ ] Confirm README/docs/workflows use Cargo package `ledge-portfolio`;
      keep `use ledge::...` examples because `[lib]` preserves that import
      name.
- [ ] Synchronize path-dependency versions in
      `crates/ledge/Cargo.toml` and `benchmarks/adapters/Cargo.toml`, then
      refresh `Cargo.lock`.
- [ ] Set `publish = false` in `python/Cargo.toml` and
      `benchmarks/adapters/Cargo.toml`.
- [ ] Synchronize `python/pyproject.toml`'s version; set its SPDX license and
      public classifier; remove `Private :: Do Not Upload`; set
      `license-files` to include `LICENSE`, `NOTICE`, and
      `THIRD_PARTY_LICENSES.html` so wheels and the sdist carry project and
      statically linked dependency notices.
- [ ] Replace pre-release status wording in `README.md`, `CONTRIBUTING.md`,
      `SECURITY.md`, `docs/OPEN_CORE.md`, `docs/PLAN.md`,
      `docs/book/src/introduction.md`, and `python/README.md`.
- [ ] Move `CHANGELOG.md`'s release notes out of `Unreleased`, add the release
      date/link, and update version badges/examples that describe the current
      release.
- [ ] Review third-party dependency and bundled documentation-asset licenses,
      including obligations for statically linked Python wheels.
- [ ] Regenerate `python/THIRD_PARTY_LICENSES.html` with
      `./scripts/generate_third_party_licenses.sh` and review any newly
      accepted license in `about.toml`.
- [ ] Commit the gate as one reviewable change; record its SHA above.

## 4. Verify the exact private gate commit

- [ ] The working tree is clean and `HEAD` equals the recorded private gate
      SHA; record `git rev-parse HEAD^{tree}` as the gate tree ID.
- [ ] Run:

  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  cargo test --workspace
  python -m pytest python/tests -q
  python scripts/generate_demo_assets.py --check
  ./scripts/generate_third_party_licenses.sh --check
  mdbook build docs/book
  ./scripts/check_open_core.sh --release
  ```

- [ ] Build and inspect publication archives, not only the checkout:

  ```bash
  cargo package -p ledge-core
  cargo package -p ledge-portfolio --list
  (cd python && maturin sdist --out dist)
  tar -tf target/package/ledge-core-*.crate
  tar -tf python/dist/ledge_portfolio-*.tar.gz
  ```

- [ ] Confirm archives/file lists contain the intended
      README/license/notice/source and contain no private files, build output,
      credentials, or absolute paths. The dependent portfolio `.crate` cannot
      be assembled until the exact `ledge-core` version is visible in the
      crates.io index; `--no-verify` does not bypass dependency resolution.
- [ ] Confirm required CI checks pass on the exact gate commit.

## 5. Publish source, then immutable artifacts

- [ ] Make the selected source repository public (or push the reviewed clean
      root); verify it anonymously before tagging.
- [ ] Confirm the public root's `git rev-parse HEAD^{tree}` equals the gate
      tree ID, then record its new commit SHA. A clean-root commit cannot have
      the same SHA as the private archive commit.
- [ ] Verify README SVG/GIF rendering, license detection, issues, Actions, and
      source archives from a logged-out browser.
- [ ] Create and push the version tag from the recorded **public clean-root
      commit SHA**, not the private archive SHA.
- [ ] Publish `ledge-core` first; wait for
      `cargo info ledge-core@<version>` to resolve, then package and inspect
      the selected dependent Rust portfolio crate before publishing it.
- [ ] Run `release.yml` at the tag with PyPI publishing disabled; download and
      inspect wheels/sdist.
- [ ] Configure/verify the PyPI Trusted Publisher and `pypi` environment, then
      rerun/approve publishing at the same tag.
- [ ] Test `cargo add` and `pip install ledge-portfolio==<version>` in clean
      temporary projects.
- [ ] Enable GitHub Pages via Actions and run `docs-deploy.yml`.
- [ ] Create the GitHub Release from the existing tag and link the benchmark
      protocol, limitations, changelog, crates.io packages, PyPI package, and
      docs.

## 6. Post-release controls

- [ ] Enable a `main` ruleset/branch protection with required CI.
- [ ] Enable private vulnerability reporting, Dependabot alerts/updates,
      secret scanning, and push protection where available.
- [ ] Review GitHub Actions default token permissions and environment
      approvers.
- [ ] Re-run `./scripts/check_open_core.sh --release` after metadata or
      boundary changes.
- [ ] Record recovery owners. Public Git history can be cloned immediately,
      and published package versions are immutable; use credential rotation,
      advisory/yank procedures, and a new patch release rather than assuming
      visibility reversal or deletion erases an exposure.
