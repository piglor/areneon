import { defineConfig } from "vitest/config";

// The rules store's tests need a running Postgres 18 (`make db-up`), so they
// run only through `make db-test` and the CI `database` job, not in the root
// `pnpm test` (vitest.config.ts excludes this package).
export default defineConfig({
  test: { include: ["test/**/*.test.ts"], fileParallelism: false },
});
