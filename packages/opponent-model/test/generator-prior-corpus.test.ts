// The belief under the committed `generator-matched` prior on the committed random-play corpus (#1169). The damage-aware@v2 cost run
// (40 games, this corpus's teams, the committed prior) refused 12 requests with `evidence_has_no_support`. The cause is not an evidence
// class the slot rules mishandle: a Mega Stone held by a Pokemon of another species (a legal set the corpus draws and the generator
// never draws: `generator.json` weighs every stone 0 so a stone appears only with its Mega species) has no mass in that prior, and
// the belief's contract is that a fact outside the prior's support has no world (`belief.test.ts`, "without a residual ..."). This
// test pins that cause exactly and holds everything else sound: at every request of both seats, the opponent's real forme, ability
// and moves are possible, and so is the real item, but for those stones. The generator's own teams are tested in
// `packages/team-dataset/test/generator-belief.test.ts`.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { describe, expect, it } from "vitest";
import { BeliefError, belief } from "../src/index.ts";
import { admits, games, hiding, points } from "./corpus.ts";
import { ROOT, realVocabulary } from "./real.ts";

const prior = JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", realVocabulary.format_id, "generator-matched.uniform-v1.json"), "utf8")) as UsagePrior;
const stones = new Set(realVocabulary.roster.flatMap((r) => (r.required_item === undefined ? [] : [r.required_item])));
const idOf = (value: string): string | null => (value === "" ? null : value.toLowerCase().replace(/[^a-z0-9]/g, ""));

/** Whether no forme of the species' prior entry gives the item any mass. */
const massless = (species: string, item: string): boolean => {
  const entry = prior.species.find((s) => s.id === species);
  return entry !== undefined && entry.formes.every((f) => !f.items.entries.some((e) => e.id === item && e.weight > 0) && f.items.residual === 0);
};

const run = () => {
  const refused: { key: string; code: string; slot: number | null; item: string | null }[] = [];
  const mismatches: string[] = [];
  let beliefs = 0;
  for (const g of games()) {
    for (const p of points(g)) {
      try {
        const b = belief({ prior, vocabulary: realVocabulary, config: hiding, opponent: p.state.opponent, flags: p.flags });
        beliefs += 1;
        if (p.truth === null) continue;
        for (const a of b.assignments.length > 0 ? [b.assignments[0]!] : []) {
          p.truth.team.forEach((set, i) => {
            const v = admits(a, i, set, p.state.opponent.roster[i]!.mega);
            if (!v.forme || v.ability === false || !v.moves) mismatches.push(`${p.game} ${p.seat} ${p.request} slot ${i}: forme/ability/moves`);
            else if (!v.item) {
              const item = idOf(set.item);
              // the only item the belief may miss: another species' Mega Stone the prior gives no mass
              if (item === null || !stones.has(item) || !massless(p.state.opponent.roster[i]!.base_species, item)) mismatches.push(`${p.game} ${p.seat} ${p.request} slot ${i}: item`);
            }
          });
        }
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        const slot = e.slot;
        const mon = slot === null ? null : p.state.opponent.roster[slot]!;
        refused.push({ key: `${p.game} ${p.seat} slot ${slot}`, code: e.code, slot, item: mon === null || !mon.item_known || mon.item_received ? null : mon.item });
      }
    }
  }
  return { refused, mismatches, beliefs };
};

describe("the committed generator-matched prior on the committed corpora", () => {
  const result = run();

  it("refuses only a Mega Stone on another species (no mass in the prior) and the Illusion twin of the reader", { timeout: 300_000 }, () => {
    const codes = new Set(result.refused.map((r) => r.code));
    expect([...codes].sort()).toEqual(["evidence_has_no_support", "reader_flagged"]);
    const unsupported = result.refused.filter((r) => r.code === "evidence_has_no_support");
    // each is a revealed stone of a Pokemon that is not its holder, and the belief under a prior with a residual would have support
    for (const r of unsupported) {
      expect(r.item, r.key).not.toBeNull();
      expect(stones.has(r.item as string), r.key).toBe(true);
    }
    // the five slots of the cost run (games 8, 14, 15, 20 and 26 of the random corpus), over every request after the stone is shown
    expect([...new Set(unsupported.map((r) => r.key))]).toEqual(["random/b08 p2 slot 5", "random/b14 p2 slot 5", "random/b15 p1 slot 1", "random/b20 p1 slot 2", "random/b26 p2 slot 0"]);
    // the reader's flag is the one Illusion twin of random/b39 (the whole game after the twin appears: its phantom roster entry stays)
    expect([...new Set(result.refused.filter((r) => r.code === "reader_flagged").map((r) => r.key.split(" slot")[0]))]).toEqual(["random/b39 p2"]);
  });

  it("holds the real forme, ability and moves possible at every other request, and the real item but for those stones", { timeout: 300_000 }, () => {
    expect(result.mismatches).toEqual([]);
    expect(result.beliefs).toBeGreaterThan(2000);
  });
});
