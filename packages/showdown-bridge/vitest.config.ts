import { configDefaults, defineConfig } from "vitest/config";

// Some tests here start the Python policy worker or the whole training loop
// (torch import, real battles): seconds on a loaded CI runner, where the 5 s
// default is too tight. One limit for the package, instead of one per test.
const limits = { testTimeout: 120_000, hookTimeout: 600_000 };

// The four battle-state projection corpus shards (ADR-030 G4, #1464) each replay a quarter of every committed corpus and keep a
// Python child busy for many minutes. Run beside the rest of the pool they starved `battle-state-spec.test.ts`, whose corpus
// hook replays the same battles on the engine (CI, run 45423769365: the hook needed more than its 400 s while four shards and
// their children shared the runner; the shards took 940 to 1,090 s each). They get a project of their own that runs one file
// at a time, so the heavy work is one worker and one child at any moment and the rest of the package keeps its workers.
const PROJECTION_SHARDS = ["test/battle-state-projection-corpus-*.python.test.ts"];

export default defineConfig({
  test: {
    projects: [
      { extends: true, test: { name: "@areneon/showdown-bridge", ...limits, exclude: [...configDefaults.exclude, ...PROJECTION_SHARDS] } },
      { extends: true, test: { name: "@areneon/showdown-bridge:projection-shards", ...limits, include: PROJECTION_SHARDS, fileParallelism: false } },
    ],
  },
});
