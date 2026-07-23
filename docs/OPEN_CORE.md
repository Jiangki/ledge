# Open-core boundary & public-release manifest

**This document is the single source of truth for "what is open" in Ledge.**

> **Current state (2026-07-23): `0.2.0` published.** The reviewed Apache-2.0
> tree is live in the clean-root public repository, Rust packages
> `ledge-core` / `ledge-portfolio` and Python distribution `ledge-portfolio`
> are published, and the documentation site is deployed. Section 5 is
> retained as the release-control runbook for future versions.

It answers three questions the maintainer keeps asking:

1. Is there a separate "open-source folder" I should copy the public code into?
   — **No.** See §1.
2. When I open-source, what exactly goes public, and where is the authority?
   — **This repository's working tree at release time, scoped by §2, is the
   authority.** No parallel copy exists or should exist.
3. What must I do to publish the source and packages safely? — The runbook in §5,
   enforced by [`../scripts/check_open_core.sh`](../scripts/check_open_core.sh).

[`PLAN.md`](PLAN.md) contains public technical direction. This file alone is
authoritative for repository paths, external/private boundaries, and the
release procedure. [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md)
is the copyable maintainer sign-off companion; it does not replace this
manifest.

---

## 1. Why there is no separate "open" folder

The instinct to extract the open parts into a dedicated folder assumes the
repository is a mix of open and closed code that must be separated. **It is
not.** As of this writing:

- Every crate and every file in this repository is the **Apache-2.0 open
  core**. See the inventory in §2.
- There is **no proprietary extension implementation code anywhere in the
  tree.** No private extension repository has been created yet.

So a folder that "extracts the open part" would copy the entire repository
into a subdirectory. That is actively harmful:

- **Drift.** A copy becomes a second source of truth that silently diverges
  from the real one the moment either side is edited.
- **Broken build.** The Cargo workspace (`Cargo.toml` members), CI
  (`.github/workflows/`), Python packaging (`python/pyproject.toml`,
  `maturin`), and every relative path (`readme = "../../README.md"`, doc
  links) assume the current layout. A subfolder copy either breaks all of it
  or forces maintaining two build systems.
- **No benefit.** The thing a folder would give you — "one authoritative
  definition of the open surface" — is delivered by *this document plus the
  repository itself*, with zero duplication.

**The correct boundary is repository-level, and it is already the plan:**

```text
this repository (public, Apache-2.0)     ← the open core; single source of truth
        ▲  public API only
        │
future extension (a SEPARATE private repository) ← outside this tree
```

Any private extension is separated by living in a **different private
repository** that depends on the *published* `ledge-core` /
`ledge-portfolio` crate APIs — never by carving a folder out of this tree.
That keeps the open
surface honest, forces public API quality, and keeps this repository
publishable.

---

## 2. Open-core inventory (everything here is open)

At the 1.6 public-release gate, the following go public under the chosen OSS
license. This is the complete tree; the list is descriptive, not a filter.

| Path | Contents | Public? |
|---|---|---|
| `crates/ledge-core/` | Solver kernel: ADMM, SMW, scaling, certificates, polishing, L1 prox, workspace, sequence, batch, KKT audit, generator | **Yes** |
| `crates/ledge/` | Portfolio-facing Rust API + examples | **Yes** |
| `python/` | PyO3 bindings, `ledge-portfolio` package, tests, examples | **Yes** |
| `benchmarks/adapters/` | OSQP / Clarabel comparison adapters (non-default features) | **Yes** |
| `benchmarks/README.md`, `benchmarks/results/` | Comparison protocol + published reports | **Yes** |
| `docs/algorithm.md`, `docs/factor_structure_note.md`, `docs/cvxpy_migration.md`, `docs/examples/`, `docs/book/`, `docs/SMOKE_TIMINGS.md` | Reader-facing technical docs and the mdBook site | **Yes** |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/` | Project meta / CI / templates | **Yes** |
| `LICENSE`, `NOTICE` | Apache-2.0 terms and project attribution | **Yes** |
| `docs/PLAN.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`, `docs/OPEN_CORE.md`, `docs/PUBLIC_RELEASE_CHECKLIST.md` | Sanitized public technical plan, roadmap, engineering decisions, boundary/runbook, and release sign-off template | **Yes — §3 decision applied** |

### Never in this repository (external/private surface)

If built, these persistent/customer-specific capabilities belong in a
separate private repository, **not** in this tree:

- Batch **rebalance engine** as a persistent product: scheduling,
  checkpoint/resume, result storage (Parquet/Arrow), orchestration UI.
  (The in-process parallel `solve_batch` loop **is** open and stays here; the
  external surface is persistent operations around it, never the loop.)
- Persistent **workspace service** (long-lived process, socket/gRPC re-solve).
- **Compliance / audit report** generation (PDF/JSON attribution trails).
- Anything backing a **support SLA** (private backport branches, customer
  tuning configs, customer data).

If a change would add any of the above to this tree, it belongs in
the separate private repository instead. The verification script (§4) checks
for telltale implementation paths.

---

## 3. Strategy-document decision — option 1 selected

The maintainer selected the recommended boundary:

- [`PLAN.md`](PLAN.md) is now a public technical plan. Revenue targets,
  pricing, customer segmentation, and pilot strategy were removed.
- [`ROADMAP.md`](ROADMAP.md) is a public engineering roadmap.
- [`DECISIONS.md`](DECISIONS.md) contains technical, API, evidence, and
  repository-boundary decisions only.
- Commercial strategy is not maintained in this repository. If it is needed,
  copy it into the future private extension repository or another private
  planning system.

This sanitizes the **current tree**, not its history. Earlier commits still
contain the old strategy text and the proprietary license. The maintainer
selected one of these history models:

1. **Publish existing history (not selected).** Simplest, but old commercial
   text and the proprietary-license era become permanently visible.
2. **Publish a clean/squashed history (selected).**
   Keep this original repository private as the internal archive; publish the
   release-gate tree as a new root commit in the public repository.
3. **Filter and force-rewrite this repository.** Possible, but disruptive to
   every clone and easy to get wrong; use only with a backup and an explicit
   maintainer decision.

The strict release check detects commercial-detail markers in the current
strategy documents. The clean-root export in §5 enforces the selected history
choice without rewriting the archive.

---

## 4. The verification script

[`../scripts/check_open_core.sh`](../scripts/check_open_core.sh) enforces this
manifest in two modes:

```bash
./scripts/check_open_core.sh            # advisory while still private
./scripts/check_open_core.sh --release  # strict; must pass before public
```

Both modes fail for secret-shaped strings and private-surface implementation
paths. Strict mode additionally fails until all release-state checks pass:

- Leaked **secrets** (tokens, private keys, `x-access-token` URLs).
- **External/private-surface code** accidentally added to this tree (§2).
- Apache-2.0 text and absence of proprietary labels in `LICENSE`, `NOTICE`,
  README, Cargo metadata, and Python package metadata.
- Synchronized Rust/Python release versions and no stale “currently private”
  notices in reader-facing files.
- Publishable `ledge-core` / selected portfolio crate with a resolved registry
  name, while Python and benchmark helper crates remain excluded from
  crates.io.
- Absence of pricing/revenue/pilot details in the public strategy documents.
- Presence of release automation and public contributor wording.

The advisory mode reports expected pre-gate items as warnings. Read the
output; do not rely only on the exit status.

This script checks the **current tracked tree**, not all Git history, legal
ownership, dependency-license obligations, remote visibility, registry
ownership, or the contents of a package after registry processing. Those are
separate sign-offs in the runbook and
[`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md).

---

## 5. Public-release runbook (roadmap 1.6 gate)

Do these in order. Steps that change legal status, repository visibility, or
package registries require an explicit maintainer action.

Approved defaults are Apache-2.0, clean-root history, version `0.2.0`, Rust
packages `ledge-core` / `ledge-portfolio` (library `ledge`), and Python
distribution/import `ledge-portfolio` / `ledge`. Before the external steps,
copy
[`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) into a private
maintainer issue and record the release version, gate commit, approver,
gate tree ID, public root commit, license, history model, and Rust/Python
package names. A public-repository
visibility flip, a cloned Git history, and a published package version cannot
be reliably “undone” by making the repository private again.

1. **Confirm rights and the current-tree boundary.**
   - Confirm the copyright owner has the right to license all first-party
     source, generated assets, benchmark data, and documentation.
   - Review third-party dependency and bundled asset licenses, including
     obligations created by statically linked Python wheels.
   - Re-read §2.
   - Run `./scripts/check_open_core.sh`.
   - The applied release tree must report zero warnings and zero failures.

2. **Apply the selected clean-root history model before making anything public.**
   - Run a dedicated full-history scanner (for example
     `gitleaks git . --redact --log-opts="--all"`) and review
     `git log --all --format='%h %an <%ae> %s'`. The current-tree script is
     not a history scanner.
   - Do not publish the existing history: old commercial strategy,
     proprietary-license commits, commit messages, and author identities
     would become visible.
   - Keep this
     repository private as the archive and publish the release-gate tree as a
     new root commit. To retain the `Jiangki/ledge` public URL, first rename
     this private archive (GitHub **Settings → General → Repository name**),
     then create the new `ledge` repository. Do not delete the archive.
     Dry-run the clean-root export locally before any external change:

     ```bash
     gate_sha="$(git rev-parse HEAD)"
     gate_tree="$(git rev-parse "$gate_sha^{tree}")"
     export_dir="$(mktemp -d)"
     git archive "$gate_sha" | tar -x -C "$export_dir"
     cd "$export_dir"
     git init -b main
     git add .
     git commit -m "Initial open-source release"
     test "$(git rev-list --count HEAD)" = 1
     test "$(git rev-parse HEAD^{tree})" = "$gate_tree"
     ./scripts/check_open_core.sh --release
     ```

     The new root commit has a different SHA from the private gate commit even
     though their tree IDs are identical. Never force-push the rewritten root
     over the private archive.
   - Record the choice in a maintainer-visible release issue/checklist.

3. **Verify strategy separation.** Option 1 is already applied to the current
   tree (§3). Copy any commercial strategy still needed from the private
   history into a private planning location. Do not add a private-doc folder
   or gitignored private code to this repository.

4. **Review the applied release-gate change.**
   - Confirm [`../LICENSE`](../LICENSE) contains the unmodified full
     Apache-2.0 text.
   - Confirm [`../NOTICE`](../NOTICE) contains the project attribution and no
     proprietary wording.
   - Confirm the reviewed `LICENSE` and `NOTICE` are copied into each
     published package root:
     `crates/ledge-core/`, `crates/ledge/`, and `python/`. Cargo does not
     automatically put a workspace-root license into member `.crate`
     archives. Set Python `license-files` to include `LICENSE`, `NOTICE`, and
     the generated `THIRD_PARTY_LICENSES.html` so both the sdist and wheels
     carry project and statically linked dependency notices.
   - Confirm release version `0.2.0` in the Cargo
     workspace and `python/pyproject.toml`; synchronize Rust path-dependency
     version requirements and `Cargo.lock`.
   - Confirm release notes are outside `CHANGELOG.md`'s `Unreleased` section
     with the release date/version link.
   - Confirm the README badge, legal status, current version, install
     instructions, and license section.
   - Confirm no stale release wording remains in `CONTRIBUTING.md`,
     `SECURITY.md`, this document, `PLAN.md`, `book/src/introduction.md`, and
     `python/README.md`.
   - Confirm workspace `license = "Apache-2.0"` and `publish = true`.
   - Confirm `publish = false` overrides in `python/Cargo.toml` and
     `benchmarks/adapters/Cargo.toml`; only `ledge-core` and the
     portfolio-library package go to crates.io.
   - Confirm the Rust registry name: as checked on 2026-07-22,
     **`ledge` is already a different crates.io package** (a time-tracking
     CLI). The selected package is `ledge-portfolio` with
     `[lib] name = "ledge"`, preserving `use ledge::…`.
   - Confirm `python/pyproject.toml` uses the Apache-2.0 SPDX license and
     classifier with no private/proprietary labels.
   - Confirm `CONTRIBUTING.md` records the
     Apache-2.0 inbound=outbound, no-CLA contributor policy.

5. **Configure registries without stored publish credentials.**
   - Recheck registry names immediately before release. On 2026-07-22,
     `ledge-core` had no exact crates.io match and PyPI `ledge-portfolio`
     returned 404, but neither observation reserves a name; crates.io
     `ledge` was occupied. Check the selected Rust names with `cargo info` and
     PyPI with `https://pypi.org/project/<name>/`; a search result is not proof
     of ownership.
   - On PyPI, configure a pending **Trusted Publisher** for owner `Jiangki`,
     repository `ledge`, workflow `release.yml`, environment `pypi`.
   - On GitHub, create the `pypi` environment and require maintainer approval
     if desired. The release workflow is kept manual until this gate.

6. **Verify the release commit before exporting the clean root.**

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace
   cargo package -p ledge-core
   cargo package -p ledge-portfolio --list
   python -m pytest python/tests -q
   python scripts/generate_demo_assets.py --check
   ./scripts/generate_third_party_licenses.sh --check
   mdbook build docs/book
   (cd python && maturin sdist --out dist)
   ./scripts/check_open_core.sh --release
   ```

   The final command must report zero failures. List and inspect the generated
   `ledge-core` `.crate`, the dependent crate's preflight file list, and the
   Python sdist: verify license/notice/readme/source, intended file boundaries,
   and the absence of credentials, build output, private files, and absolute
   paths. The dependent portfolio `.crate` cannot be assembled before its
   exact `ledge-core` dependency exists in the crates.io index; `--no-verify`
   does not bypass registry dependency resolution. Confirm the working tree is
   clean and record both `git rev-parse HEAD` as the private gate SHA and
   `git rev-parse HEAD^{tree}` as the gate tree ID.

7. **Publish the clean-root source repository (selected history model).**
   - Merge the reviewed gate changes into the historical repository, then
     rename that repository to an archive name in GitHub **Settings →
     General → Repository name**. Keep its visibility private.
   - Create a new, empty public `Jiangki/ledge`, repeat the clean-root export
     from the exact private gate SHA, and verify the exported tree ID equals
     the recorded gate tree ID. Then add the **new** public repository URL,
     record the public root SHA, and push:

     ```bash
     test "$(git rev-parse HEAD^{tree})" = "<recorded-gate-tree-id>"
     public_sha="$(git rev-parse HEAD)"
     git remote add origin git@github.com:Jiangki/ledge.git
     git remote -v
     git push -u origin main
     test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$public_sha"
     ```

     Never change the historical archive itself to public, and never use its
     commit SHA for the public tag.
   - Immediately check the anonymous GitHub view for README images, license,
     source archives, issues, and Actions permissions. Do not tag until the
     anonymous view points at the recorded public root SHA.

8. **Tag the exact verified source, then publish packages.**
   - Create and push `v0.2.0` from the recorded public root SHA (change the
     value if the approved release version changed):

     ```bash
     public_sha="<recorded-public-root-sha>"
     test "$(git rev-parse HEAD)" = "$public_sha"
     git tag -a v0.2.0 "$public_sha" -m "Ledge 0.2.0"
     git push origin v0.2.0
     ```

   - Publish in dependency order:
     `cargo publish -p ledge-core`, wait until
     `cargo info ledge-core@0.2.0` resolves, then run
     `cargo package -p ledge-portfolio`, inspect that `.crate`, and publish it
     with `cargo publish -p ledge-portfolio`.
   - Run the manual `release.yml` workflow with the **tag ref** selected and
     input `v0.2.0` and PyPI publishing disabled. Download and inspect every
     wheel and the sdist; then rerun/approve the same tag with the `pypi`
     environment deployment enabled.
   - Verify in a clean environment with
     `pip install ledge-portfolio==0.2.0` and crates.io package checks.

9. **Enable docs.** GitHub **Settings → Pages → Build and deployment →
   Source: GitHub Actions**, then manually run `docs-deploy.yml`. Check links
   and images from the public Pages URL.

10. **Announce only verified artifacts.** Create the GitHub Release from the
    existing `v0.2.0` tag, linking the smoke/comparison reports, limitations,
    changelog, crates.io packages, PyPI package, and docs. Do not claim
    performance beyond the committed reports.

11. **Post-release controls.** Enable branch protection/rulesets for `main`,
    require CI, enable private vulnerability reporting, Dependabot,
    secret scanning/push protection where available, review Actions token
    permissions and environment approvers, and re-run
    `./scripts/check_open_core.sh --release` whenever the boundary or release
    metadata changes.

---

## 6. Maintenance

- This file is updated whenever the open/external boundary moves. Any change to
  what is open must be logged in [`DECISIONS.md`](DECISIONS.md) and reflected
  in §2 here.
- If a private extension repository is created, record only its existence
  here (not its private URL) and confirm it depends on **published** crate
  versions, never on a path into this tree.
