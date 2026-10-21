import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playRandomBattle, type RandomBattleFile, randomBattleFile } from "../src/random-corpus.ts";
import { type LegalChoiceSet } from "@areneon/schema/types/v1/legal-choice-set.ts";

// A seeded random-legal battle in which p1's Pawmot uses Revival Blessing
// (scripts/capture-revival-battle.ts): runBattle enumerates, judges and plays
// the `reviving` requests with no refusal.
const dir = join(GOLDEN_DIR, "revival");
const committed = JSON.parse(readFileSync(join(dir, "battle.json"), "utf8")) as RandomBattleFile;
const legal = JSON.parse(readFileSync(join(dir, "legal.json"), "utf8")) as LegalChoiceSet[];

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

describe("a battle with Revival Blessing", () => {
  it("re-plays to the same channels and the same legal sets", async () => {
    const { id, seed, actor_seeds, teams } = committed;
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
    expect(randomBattleFile({ id, seed, actor_seeds, teams }, record)).toEqual(committed);
    expect(record.forfeit).toBeUndefined();
    const reviving = record.requests.p1.filter((r) =>
      JSON.stringify(r.request).includes('"reviving":true'),
    );
    expect(reviving.length).toBeGreaterThan(0);
    expect(reviving.map((r) => r.legal)).toEqual(legal);
    for (const r of reviving) {
      expect(r.phase).toBe("forceSwitch");
      expect(r.legal?.pending).toEqual([]);
      expect(r.legal?.choices.length).toBeGreaterThan(0);
    }
  }, 60_000);

  it("shows the engine reviving the Pokémon the choice named", () => {
    const lines = committed.channels.p1.lines;
    const text = lines.join("\n");
    expect(text).toContain("[from] move: Revival Blessing");
  });
});
