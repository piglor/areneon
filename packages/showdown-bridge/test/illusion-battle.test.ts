import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playRandomBattle, type RandomBattleFile, randomBattleFile } from "../src/random-corpus.ts";

// A seeded random-legal battle on the golden team with a Zoroark (Illusion) in the last place
// of both teams (scripts/capture-illusion-battle.ts): each side's disguise breaks.
const committed = JSON.parse(
  readFileSync(join(GOLDEN_DIR, "illusion", "battle.json"), "utf8"),
) as RandomBattleFile;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

describe("a battle with Illusion", () => {
  it("re-plays to the same channels", async () => {
    const { id, seed, actor_seeds, teams } = committed;
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
    expect(randomBattleFile({ id, seed, actor_seeds, teams }, record)).toEqual(committed);
    expect(record.forfeit).toBeUndefined();
  }, 60_000);

  it("shows both players each side's disguise ending", () => {
    for (const player of ["p1", "p2"] as const) {
      const lines = committed.channels[player].lines;
      expect(lines).toContain("|replace|p1a: Zoroark|Zoroark, L50, M");
      expect(lines).toContain("|replace|p2b: Zoroark|Zoroark, L50, M");
    }
  });
});
