import { defineConfig } from "vitest/config";

// One root config so every workspace package is tested and coverage-gated
// (≥90%, AGENTS.md) without each package having to opt in. perFile makes the
// threshold apply to every source file, so a well-covered package cannot
// hide an uncovered one.
export default defineConfig({
  test: {
    // rules-store needs Postgres: it runs in `make db-test` and the CI database job.
    projects: ["packages/*", "!packages/rules-store"],
    coverage: {
      provider: "v8",
      include: ["packages/*/src/**", "packages/*/testing/**", "packages/replay-view/web/**/*.cjs"],
      // Gated by its own run in `make db-test` (packages/rules-store/vitest.config.ts).
      exclude: ["packages/rules-store/**"],
      thresholds: { perFile: true, lines: 90, branches: 90, functions: 90, statements: 90 },
    },
  },
});
