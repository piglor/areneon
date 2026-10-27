// A smoke test on the pinned format's real vocabulary and the committed random-play channels: with
// a prior that is uniform over what the vocabulary allows (constraints alone), the belief exists at
// every request of both players' channels. The corpus-wide soundness, invariance and cost tests are
// a later part of ADR-025 slice 2; this checks only that real shapes (Mega formes, 170 items, long
// move pools) go through the filter.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import { describe, expect, it } from "vitest";
import { belief } from "../src/index.ts";
import { DIR, realVocabulary, uniformPrior } from "./real.ts";
import { config, text } from "./support.ts";

const vocabulary = realVocabulary;
const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));

describe("the belief on real channels with a uniform prior", () => {
  const files = readdirSync(join(DIR, "random")).filter((f) => /^b\d+\.json$/.test(f)).sort().slice(0, 3);
  const hiding = { ...config, identity_hiding_abilities: vocabulary.roster.some((r) => r.abilities.includes("illusion")) ? ["illusion"] : [] };

  it("has a belief at every request, a species that can hide its identity included", () => {
    let beliefs = 0;
    for (const file of files) {
      const battle = JSON.parse(readFileSync(join(DIR, "random", file), "utf8")) as { channels: Record<"p1" | "p2", { lines: string[] }> };
      for (const player of ["p1", "p2"] as const) {
        const run = interpretChannel(player, battle.channels[player].lines);
        expect(run.flags).toEqual([]);
        const states = run.frames.filter((f) => f.kind === "request").map((f) => f.state);
        expect(states.length).toBeGreaterThan(2);
        for (const state of states) {
          const ids = state.opponent.roster.map((m) => m.base_species);
          expect(ids.every((id) => roster.has(id))).toBe(true);
          const b = belief({ prior: uniformPrior(ids), vocabulary, config: hiding, opponent: state.opponent, flags: run.flags });
          beliefs += 1;
          // Probabilities are exact: every slot's item marginals add up to one.
          for (const s of b.slots) expect(s.items.reduce((sum, m) => sum + Number(m.p.num) / Number(m.p.den), 0)).toBeCloseTo(1, 9);
          expect(b.itemClause.enforced).toBe(true);
          expect(text(belief({ prior: uniformPrior(ids), vocabulary, config: hiding, opponent: state.opponent, flags: run.flags }))).toBe(text(b));
        }
      }
    }
    expect(beliefs).toBeGreaterThan(10);
  });
});
