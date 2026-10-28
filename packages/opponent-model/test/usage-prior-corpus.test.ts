// The belief's species coverage under the committed Smogon usage priors on the committed corpora (#1209). A belief over the real prior
// costs seconds a request (24-move lists), so the whole corpus is not run through `belief`: whether a request is refused as
// `species_not_in_prior` is exactly whether one of its six previewed species has no entry (the lookup `belief` makes first), which
// is cheap, and the rosters that held an uncovered species under the first record are then run through `belief` once, at their
// Team Preview request, under the second.
//
// The first record (`usage-prior.2026-09-1630.json`) had no entry for 69 of the vocabulary's 300 preview species (two of them only
// because it mapped no `Farfetch'd` or `Sirfetch'd`); the second maps those two by the engine's own id normalisation and fills the 67
// others from the committed `generator-matched` prior, labelled in its `source.fallback`.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { describe, expect, it } from "vitest";
import { BeliefError, belief } from "../src/index.ts";
import { games, hiding, points } from "./corpus.ts";
import { ROOT, realVocabulary } from "./real.ts";

const dir = join(ROOT, "config", "opponent-model", realVocabulary.format_id);
const load = (file: string): UsagePrior => JSON.parse(readFileSync(join(dir, file), "utf8")) as UsagePrior;
const first = load("usage-prior.2026-09-1630.json");
const second = load("usage-prior.2026-09-1630.v2.json");

type Point = ReturnType<typeof points>[number];

/** The species of the roster that the prior has no entry for (the lookup `belief` makes, slot by slot). */
const uncovered = (prior: UsagePrior, p: Point): string[] => {
  const ids = new Set(prior.species.map((s) => s.id));
  return p.state.opponent.roster.flatMap((m) => (ids.has(m.base_species) || (m.species !== null && ids.has(m.species)) ? [] : [m.base_species]));
};

const all = () => games().flatMap((g) => points(g));

describe("the committed usage priors on the committed corpora", () => {
  it("the second record covers every previewed species of the corpora, the first did not", () => {
    const pts = all();
    const requests = (prior: UsagePrior) => pts.filter((p) => uncovered(prior, p).length > 0);
    const before = requests(first);
    const species = (list: Point[], prior: UsagePrior) => [...new Set(list.flatMap((p) => uncovered(prior, p)))].sort();
    const rosters = (list: Point[]) => new Set(list.map((p) => `${p.game} ${p.seat}`)).size;
    expect({ requests: pts.length, refused: before.length, rosters: rosters(before), species: species(before, first).length }).toMatchInlineSnapshot(`
      {
        "refused": 874,
        "requests": 2152,
        "rosters": 40,
        "species": 23,
      }
    `);
    expect(requests(second)).toEqual([]);
  });

  it("answers the Team Preview of every roster the first record refused, or refuses it for another reason it names", { timeout: 600_000 }, () => {
    const pts = all();
    const refusedRosters = new Map<string, Point>();
    for (const p of pts) if (uncovered(first, p).length > 0 && !refusedRosters.has(`${p.game} ${p.seat}`)) refusedRosters.set(`${p.game} ${p.seat}`, p);
    const codes: Record<string, number> = {};
    for (const p of refusedRosters.values()) {
      try {
        belief({ prior: second, vocabulary: realVocabulary, config: hiding, opponent: p.state.opponent, flags: p.flags });
        codes.answered = (codes.answered ?? 0) + 1;
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        codes[e.code] = (codes[e.code] ?? 0) + 1;
      }
    }
    expect(codes).toMatchInlineSnapshot(`
      {
        "answered": 40,
      }
    `);
  });
});
