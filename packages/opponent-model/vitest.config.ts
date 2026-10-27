import { defineProject } from "vitest/config";

// The real-channel smoke test runs the exact belief at every request of several battles, several
// times that under coverage.
export default defineProject({
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 60_000,
  },
});
