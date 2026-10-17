import { defineConfig } from "vitest/config";

// One root config so every workspace package is tested and coverage-gated
// (≥90%, AGENTS.md) without each package having to opt in. perFile makes the
// threshold apply to every source file, so a well-covered package cannot
// hide an uncovered one.
export default defineConfig({
  test: {
    projects: ["packages/*"],
    coverage: {
      provider: "v8",
      include: ["packages/*/src/**"],
      thresholds: { perFile: true, lines: 90, branches: 90, functions: 90, statements: 90 },
    },
  },
});
