# Security Policy

Ledge is alpha numerical software. It is **not** hardened for untrusted
inputs in a multi-tenant service.

## Supported versions

Only the latest commit on `main` receives security fixes. There are no
long-term support releases yet.

## Reporting a vulnerability

Contact the repository maintainers privately (GitHub private security
advisories if available, otherwise a direct message). Do not discuss
exploitable issues in shared channels until a fix is available.

Please include:

- affected commit SHA;
- minimal reproduction;
- impact assessment (crash, wrong solution used as trusted output, etc.).

## Non-security numerical bugs

Incorrect optima, slow convergence, and residual failures are **correctness /
quality bugs**, not security issues. File ordinary GitHub issues for those,
using synthetic or redacted problem data.
