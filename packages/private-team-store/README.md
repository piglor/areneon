# @areneon/private-team-store

The private team store (ADR-015). So far this package holds the pure import
planner (`src/import-plan.ts`: staged files plus the parent snapshot's records
in, the next records and a counts-only summary out, or a typed refusal that
never repeats input text), its exact-pinned dependencies and the packaging rule
that keeps it out of every image.

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

## Dependency note

`@noble/post-quantum` 0.5.4 declares the tilde ranges `~2.0.0` for
`@noble/curves` and `@noble/hashes`, which resolve to 2.0.1. The lockfile
therefore holds two copies of each: the hybrid ML-KEM path runs on the 2.0.1
copies, while age's X25519 uses the exact 2.4.0 pins. A later
`@noble/post-quantum` bump must re-check which copies it resolves.

## Operator rule

A repository test sees only tracked files. A Coolify application configured in
the UI, or a Nixpacks or Railpack build that runs a root `pnpm install` through
the `packages/*` workspace glob, is not detectable by it. Configure every build
with an explicit filter (`pnpm install --filter <package>...`) and never build
from the repository root without one.
