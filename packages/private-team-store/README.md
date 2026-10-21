# @areneon/private-team-store

The private team store (ADR-015). So far this package holds the pure import
planner (`src/import-plan.ts`, with `src/staged-file.ts` for the staged-file
header and `src/import-error.ts` for the typed refusals), the private snapshot
writer and reader (`src/snapshot-writer.ts`, `src/snapshot-reader.ts`: records
to Parquet shards and manifest through the pinned DuckDB tool, and the full
verification of a stored snapshot; fixed-message refusals only), the object
store port with its in-memory and R2 implementations (`src/object-store.ts`,
`src/memory-store.ts`, `src/r2-client.ts`, `src/r2-settings.ts`), the publish
sequence (`src/publish.ts`, `src/local-copy.ts`, `src/current-pointer.ts`,
`src/snapshot-store.ts`), its exact-pinned dependency and the packaging rule
that keeps it out of every image. The import and export commands are not built.

- `"private": true` and **never shipped**: no Dockerfile, compose file, Coolify
  build or other workspace package refers to it. `test/packaging.test.ts`
  enforces that.
- `aws4fetch` (MIT, no dependencies) signs the R2 requests (SigV4) and is
  pinned to an exact version. Dependabot ignores it for version updates, so a
  bump is a manual, reviewed change (security-update PRs may still open, and
  the `osv` job still scans). `make licences` checks it like every production
  dependency. Only `src/r2-settings.ts` imports it, and only
  `r2-client.ts` calls `fetch`.
- The DuckDB CLI, the `age` CLI and OpenSSL are user-installed tools, never
  listed in a `package.json`, `pyproject.toml` or Dockerfile.

## Operator rule

A repository test sees only tracked files. A Coolify application configured in
the UI, or a Nixpacks or Railpack build that runs a root `pnpm install` through
the `packages/*` workspace glob, is not detectable by it. Configure every build
with an explicit filter (`pnpm install --filter <package>...`) and never build
from the repository root without one.
