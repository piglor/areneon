import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playRandomBattle, type RandomBattleFile, randomBattleFile } from "../src/random-corpus.ts";

// A seeded random-legal battle with a Grassy Seed Dedenne that Flings an Arboliva (Seed Sower)
// (scripts/capture-fling-battle.ts): the seed is eaten before the Fling's own item end, which
// then names no item (#1162).
const committed = JSON.parse(readFileSync(join(GOLDEN_DIR, "fling", "battle.json"), "utf8")) as RandomBattleFile;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

describe("a battle with a Fling of an item that is already gone", () => {
  it("re-plays to the same channels", async () => {
    const { id, seed, actor_seeds, teams } = committed;
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
    expect(randomBattleFile({ id, seed, actor_seeds, teams }, record)).toEqual(committed);
    expect(record.forfeit).toBeUndefined();
  }, 60_000);

  it("shows both players the seed eaten and then an item end with no name", () => {
    for (const player of ["p1", "p2"] as const) {
      const lines = committed.channels[player].lines;
      const empty = lines.indexOf("|-enditem|p1b: Dedenne||[from] move: Fling");
      expect(empty).toBeGreaterThan(0);
      expect(lines.slice(0, empty).some((l) => l === "|-enditem|p1b: Dedenne|Grassy Seed")).toBe(true);
    }
  });
});
