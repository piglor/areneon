import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  battleSummary,
  type BattleSummary,
  GOLDEN_DIR,
  GOLDEN_FORMAT,
  GOLDEN_PHASES,
  GOLDEN_SUMMARY,
  goldenBattle,
  goldenFixtures,
  syntheticTeam,
} from "../src/golden.ts";
import { type BattleRecord, loadEngine, replayActor, runBattle } from "../src/index.ts";

const committed = <T>(file: string): T =>
  JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as T;

describe("golden request fixtures", () => {
  // A Showdown pin bump that changes behaviour fails here; re-capture with
  // scripts/capture-golden.ts and review the diff.
  it("match a fresh capture from the pinned engine", async () => {
    const fresh = goldenFixtures(await goldenBattle(loadEngine()));
    expect(Object.keys(fresh).sort()).toEqual(Object.keys(GOLDEN_PHASES).sort());
    for (const [file, fixture] of Object.entries(fresh)) {
      expect(committed(file), file).toEqual(fixture);
    }
  });

  // Cross-process determinism: this process reproduces the log hash that
  // the capture run committed (same commit + seed + choices).
  it("reproduce the committed battle log hash in a fresh process", async () => {
    const summary = committed<BattleSummary>(GOLDEN_SUMMARY);
    expect(battleSummary(await goldenBattle(loadEngine()))).toEqual(summary);
  });

  it("replay the committed choices to the same log hash", async () => {
    const summary = committed<BattleSummary>(GOLDEN_SUMMARY);
    const team = syntheticTeam();
    const replayed = await runBattle({
      engine: loadEngine(),
      formatId: GOLDEN_FORMAT,
      seed: summary.seed,
      teams: { p1: team, p2: team },
      actors: { p1: replayActor(summary.choices, "p1"), p2: replayActor(summary.choices, "p2") },
    });
    expect(battleSummary(replayed).omniscient_log_sha256).toBe(summary.omniscient_log_sha256);
  });

  it("refuse a battle missing a golden phase", () => {
    const record = { requests: { p1: [], p2: [] } } as unknown as BattleRecord;
    expect(() => goldenFixtures(record)).toThrow(/no p1 preview request/);
  });
});
