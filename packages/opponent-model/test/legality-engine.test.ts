// The sampler against the pinned engine (ADR-025 section 4, V1): on the pinned format's real vocabulary and
// the committed random-play channels, with a prior that is uniform over what the vocabulary allows, every
// sampled brought set passes the target format's `validateSet` and is the engine's canonical paste (the
// bridge's `SetLegality` port), the world respects Item Clause and every revealed fact, and no redraw was
// needed. The bridge is imported by path only for this test: the package itself is engine-free and depends
// on neither the bridge nor the engine.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { createSetLegality } from "../../showdown-bridge/src/set-legality.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import type { Engine } from "../../showdown-bridge/src/engine.ts";
import { type SetLegality, type SpreadSampler, type WorldSet, belief, createSampler } from "../src/index.ts";
import { DIR, realVocabulary, uniformPrior } from "./real.ts";
import { config } from "./support.ts";

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

// The prior's other-spread entry: a nature and two Stat Point stats from the world's stream, within the format's limits.
const spreads: SpreadSampler = {
  other: (_species, _forme, below) => ({
    nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id,
    stat_points: { hp: Number(below(33n)), atk: Number(below(33n)), def: 0, spa: 0, spd: 0, spe: Number(below(2n)) },
  }),
};

describe("sampled worlds on the committed random-play channels", () => {
  it("every brought set is legal and canonical, Item Clause and the revealed facts hold, and no redraw was needed", () => {
    const port: SetLegality = createSetLegality({ engine, vocabulary: realVocabulary });
    const sampler = createSampler({ legality: port, spreads });
    const roster = new Map(realVocabulary.roster.map((r) => [r.id, r]));
    const files = readdirSync(join(DIR, "random")).filter((f) => /^b\d+\.json$/.test(f)).sort().slice(0, 2);
    let worlds = 0;
    for (const file of files) {
      const battle = JSON.parse(readFileSync(join(DIR, "random", file), "utf8")) as { channels: Record<"p1" | "p2", { lines: string[] }> };
      for (const player of ["p1", "p2"] as const) {
        const run = interpretChannel(player, battle.channels[player].lines);
        const states = run.frames.filter((f) => f.kind === "request").map((f) => f.state);
        for (const state of [states[0]!, states[Math.floor(states.length / 2)]!, states[states.length - 1]!]) {
          const ids = state.opponent.roster.map((m) => m.base_species);
          const b = belief({ prior: uniformPrior(ids), vocabulary: realVocabulary, config, opponent: state.opponent, flags: run.flags });
          const out = sampler.sampleWorlds(b, Uint8Array.from({ length: 32 }, (_, i) => (worlds + i) & 255), 4);
          expect(out.redraws).toBe(0);
          for (const { world } of out.worlds) {
            worlds += 1;
            const sets = world.slots.flatMap((s) => (s.set === null ? [] : [s.set]));
            expect(sets).toHaveLength(realVocabulary.rules.picked_team_size);
            const items = sets.flatMap((s) => (s.item === null ? [] : [s.item]));
            expect(new Set(items).size).toBe(items.length);
            for (const set of sets) expect(port.legal(set), `${set.forme} is legal`).toBe(true);
            state.opponent.roster.forEach((mon, i) => {
              const slot = world.slots[i] as { brought: boolean; set: WorldSet | null };
              if (mon.brought === true || mon.active_slot !== null || mon.fainted) expect(slot.brought).toBe(true);
              if (slot.set === null) return;
              const pool = new Set((roster.get(slot.set.forme) as { moves: string[] }).moves);
              for (const move of mon.moves_revealed.filter((m) => pool.has(m))) expect(slot.set.moves, `${mon.name} revealed ${move}`).toContain(move);
              if (mon.item_known && mon.item !== null) expect(slot.set.item).toBe(mon.item);
              if (mon.ability !== null && !mon.mega) expect(slot.set.ability).toBe(mon.ability);
            });
          }
        }
      }
    }
    expect(worlds).toBeGreaterThan(40);
  });
});
