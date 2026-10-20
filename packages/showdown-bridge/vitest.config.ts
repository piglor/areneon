import { defineConfig } from "vitest/config";

// Some tests here start the Python policy worker or the whole training loop
// (torch import, real battles): seconds on a loaded CI runner, where the 5 s
// default is too tight. One limit for the package, instead of one per test.
export default defineConfig({
  test: {
    testTimeout: 120_000,
    hookTimeout: 600_000,
  },
});
