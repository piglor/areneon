import { defineConfig } from "vitest/config";

// The rules store's tests need a running Postgres 18 (`make db-up`), so they
// run only through `make db-test` and the CI `database` job, not in the root
// `pnpm test` (vitest.config.ts excludes this package). That run carries the
// same per-file coverage gate as the root one.
export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    fileParallelism: false,
    coverage: {
      enabled: true,
      provider: "v8",
      include: ["src/**"],
      thresholds: { perFile: true, lines: 90, branches: 90, functions: 90, statements: 90 },
    },
  },
});
