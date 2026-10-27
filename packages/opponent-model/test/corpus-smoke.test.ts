// A smoke test on the pinned format's real vocabulary and the committed random-play channels: with
// a prior that is uniform over what the vocabulary allows (constraints alone), the belief exists at
// every request of both players' channels. The corpus-wide soundness, invariance and cost tests are
// a later part of ADR-025 slice 2; this checks only that real shapes (Mega formes, 170 items, long
// move pools) go through the filter.

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretChannel } from "@areneon/public-state";
import type { TeamVocabulary } from "@areneon/schema/types/v1/team-vocabulary.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { describe, expect, it } from "vitest";
import { BeliefError, belief } from "../src/index.ts";
import { config, text } from "./support.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const DIR = join(ROOT, "fixtures", "showdown", "gen9championsvgc2026regmc");
const vocabulary = JSON.parse(readFileSync(join(DIR, "team-vocabulary.json"), "utf8")) as TeamVocabulary;
const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));

type Forme = UsagePrior["species"][number]["formes"][number];
const uniformForme = (id: string, weight: number): Forme => ({
  id,
  weight,
  abilities: { denominator: 1, entries: [], residual: 1, unmapped: 0 },
  items: { denominator: 2, entries: [], no_item: 1, residual: 1, unmapped: 0 },
  spreads: { denominator: 1, entries: [], residual: 1, unmapped: 0 },
  moves: { denominator: 2, slots: 4, entries: [], empty_slots: 0, residual: 8, unmapped: 0 },
});

/** A prior uniform over the vocabulary for the given preview species: the species and its Mega formes. */
function uniformPrior(ids: readonly string[]): UsagePrior {
  const species = [...ids].sort().map((id) => {
    const formes = vocabulary.roster.filter((r) => r.id === id || (r.required_item !== undefined && r.id.startsWith(`${id}mega`))).map((r) => uniformForme(r.id, 1));
    return { id, denominator: formes.length, formes: formes as [Forme, ...Forme[]] };
  });
  return { schema_version: 1, format_id: vocabulary.format_id, engine_data_hash: vocabulary.engine_data_hash, vocabulary_hash: "0".repeat(64), source: { kind: "generator-matched", generator_config_hash: "0".repeat(64), generator_commit: "a".repeat(40) }, derivation: { script_version: "test", k_items: 16, k_moves: 24, k_spreads: 32 }, species: species as UsagePrior["species"] };
}

describe("the belief on real channels with a uniform prior", () => {
  const files = readdirSync(join(DIR, "random")).filter((f) => /^b\d+\.json$/.test(f)).sort().slice(0, 3);
  const hiding = { ...config, identity_hiding_abilities: vocabulary.roster.some((r) => r.abilities.includes("illusion")) ? ["illusion"] : [] };

  it("has a belief at every request, or a typed refusal for a species that can hide its identity", () => {
    let beliefs = 0;
    let refused = 0;
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
          try {
            const b = belief({ prior: uniformPrior(ids), vocabulary, config: hiding, opponent: state.opponent, flags: run.flags });
            beliefs += 1;
            // Probabilities are exact: every slot's item marginals add up to one.
            for (const s of b.slots) expect(s.items.reduce((sum, m) => sum + Number(m.p.num) / Number(m.p.den), 0)).toBeCloseTo(1, 9);
            expect(b.itemClause.enforced).toBe(true);
            expect(text(belief({ prior: uniformPrior(ids), vocabulary, config: hiding, opponent: state.opponent, flags: run.flags }))).toBe(text(b));
          } catch (e) {
            if (!(e instanceof BeliefError) || e.code !== "identity_hiding_species") throw e;
            refused += 1;
          }
        }
      }
    }
    expect(beliefs).toBeGreaterThan(10);
    expect(refused).toBeLessThan(beliefs);
  });
});
