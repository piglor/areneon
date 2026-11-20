// Gate 1 on ADR-026's targeted search-fidelity corpus (200 battles, #1462): each battle recorded on the unmodified engine and
// replayed on a fresh engine object with its draws forced gives no difference at any step of either seat. Apart from
// differential-battle.test.ts so the two run in parallel.
import { type BattleTrace, battleCorpusSummary } from "@areneon/differential/battle";
import { committedBattles, loadEngine } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { battleBackend, compareTraces, loadInteractionClasses, newBattleCorpus, traceOf } from "../src/differential-battle.ts";
import { engine } from "./support.ts";

describe("gate 1 on the targeted corpus", () => {
  it("replays every search-fidelity battle with its draws forced, with no difference", { timeout: 900_000 }, async () => {
    const classes = loadInteractionClasses();
    const replay = loadEngine(process.env.SHOWDOWN_DIR || undefined);
    const backend = battleBackend(replay);
    const corpus = newBattleCorpus({ id: "search-fidelity", kind: "regression", seed: "0".repeat(64), pool: "2".repeat(64) }, classes);
    const battles = committedBattles(["search-fidelity"]);
    expect(battles.length).toBe(200);
    // In chunks, so the traces of the whole corpus are never held at once.
    for (let first = 0; first < battles.length; first += 25) {
      const traces: BattleTrace[] = [];
      for (const { spec } of battles.slice(first, first + 25)) traces.push(await traceOf(engine, spec));
      compareTraces(classes, traces, corpus, backend, 25);
    }
    expect(corpus.differences).toEqual([]);
    expect(corpus.unreached).toBe(0);
    expect(corpus.battles).toBe(200);
    const summary = battleCorpusSummary(corpus) as { classes: { id: string; tagged: number; equal: number }[] };
    const tagged = new Map(summary.classes.map((c) => [c.id, c.tagged]));
    for (const id of ["item_consumption", "protection_multi_target", "speed_control", "weather_terrain_readers"]) expect(tagged.get(id) ?? 0, id).toBeGreaterThan(0);
  });
});
