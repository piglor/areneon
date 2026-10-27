// The pinned format's real vocabulary and a usage prior that is uniform over it, for tests that run the
// belief on committed channels. With the uniform prior the constraints are the evidence alone, and the
// residual mass gives every legal set a weight above zero.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { TeamVocabulary } from "@areneon/schema/types/v1/team-vocabulary.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
export const DIR = join(ROOT, "fixtures", "showdown", "gen9championsvgc2026regmc");
export const realVocabulary = JSON.parse(readFileSync(join(DIR, "team-vocabulary.json"), "utf8")) as TeamVocabulary;

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
export function uniformPrior(ids: readonly string[]): UsagePrior {
  const species = [...ids].sort().map((id) => {
    const formes = realVocabulary.roster.filter((r) => r.id === id || (r.required_item !== undefined && r.id.startsWith(`${id}mega`))).map((r) => uniformForme(r.id, 1));
    return { id, denominator: formes.length, formes: formes as [Forme, ...Forme[]] };
  });
  return { schema_version: 1, format_id: realVocabulary.format_id, engine_data_hash: realVocabulary.engine_data_hash, vocabulary_hash: "0".repeat(64), source: { kind: "generator-matched", generator_config_hash: "0".repeat(64), generator_commit: "a".repeat(40) }, derivation: { script_version: "test", k_items: 16, k_moves: 24, k_spreads: 32 }, species: species as UsagePrior["species"] };
}
