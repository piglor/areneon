# @areneon/private-team-store

The private team store (ADR-014). Today this package holds no features, only
its exact-pinned dependencies and the packaging rule that keeps it out of every
image.

- `"private": true` and **never shipped**: no Dockerfile, compose file, Coolify
  build or other workspace package refers to it. `test/packaging.test.ts`
  enforces that.
- `age-encryption` (BSD-3-Clause) and the `@noble/*` and `@scure/base` packages
  (MIT) are pinned to exact versions. Dependabot ignores them for version
  updates, so a bump is a manual, reviewed change (security-update PRs may
  still open, and the `osv` job still scans). `make licences` checks them like
  every production dependency.
- The DuckDB CLI, the `age` CLI and OpenSSL are user-installed tools, never
  listed in a `package.json`, `pyproject.toml` or Dockerfile.
