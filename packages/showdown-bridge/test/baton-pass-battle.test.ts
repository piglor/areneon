import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playRandomBattle, type RandomBattleFile, randomBattleFile } from "../src/random-corpus.ts";

// A seeded random-legal battle on a team of Swords Dance + Baton Pass users
// (scripts/capture-baton-pass-battle.ts): raised stages are passed on, with no line of their own.
const committed = JSON.parse(
  readFileSync(join(GOLDEN_DIR, "baton-pass", "battle.json"), "utf8"),
) as RandomBattleFile;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

describe("a battle with Baton Pass", () => {
  it("re-plays to the same channels", async () => {
    const { id, seed, actor_seeds, teams } = committed;
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
    expect(randomBattleFile({ id, seed, actor_seeds, teams }, record)).toEqual(committed);
    expect(record.forfeit).toBeUndefined();
  }, 60_000);

  it("shows both players the switch line that names the pass, and no line for the stages", () => {
    for (const player of ["p1", "p2"] as const) {
      const lines = committed.channels[player].lines;
      const passes = lines.filter((l) => l.startsWith("|switch|") && l.endsWith("|[from] Baton Pass"));
      expect(passes.length).toBeGreaterThan(0);
      for (const l of passes) expect(lines[lines.indexOf(l) + 1]).not.toMatch(/boost/);
    }
  });
});
