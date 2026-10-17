import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_DIR, GOLDEN_PHASES, goldenBattle, goldenFixtures } from "../src/golden.ts";
import { type BattleRecord, loadEngine } from "../src/index.ts";

describe("golden request fixtures", () => {
  // A Showdown pin bump that changes behaviour fails here; re-capture with
  // scripts/capture-golden.ts and review the diff.
  it("match a fresh capture from the pinned engine", async () => {
    const fresh = goldenFixtures(await goldenBattle(loadEngine()));
    expect(Object.keys(fresh).sort()).toEqual(Object.keys(GOLDEN_PHASES).sort());
    for (const [file, fixture] of Object.entries(fresh)) {
      const committed: unknown = JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8"));
      expect(committed, file).toEqual(fixture);
    }
  });

  it("refuse a battle missing a golden phase", () => {
    const record = { requests: { p1: [], p2: [] } } as unknown as BattleRecord;
    expect(() => goldenFixtures(record)).toThrow(/no p1 preview request/);
  });
});
