import { defineConfig } from "vitest/config";

// One root config so every workspace package is tested and coverage-gated
// (≥90%, AGENTS.md) without each package having to opt in.
export default defineConfig({
  test: {
    projects: ["packages/*"],
    coverage: {
      provider: "v8",
      include: ["packages/*/src/**"],
      thresholds: { lines: 90, branches: 90, functions: 90, statements: 90 },
    },
  },
});
