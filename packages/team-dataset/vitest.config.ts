import { defineProject } from "vitest/config";

// The generator tests run the pinned engine's validator, so a run of a few
// thousand draws takes seconds, and several times that under coverage.
export default defineProject({
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 60_000,
  },
});
