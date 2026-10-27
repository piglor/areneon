// The opponent usage prior (ADR-025 section 2): per preview species and forme, the marginal
// distributions an unrevealed opponent set is drawn from. The schema checks the shape; this reader
// adds what the contract profile cannot say, against the team vocabulary the prior was built for:
// sorted and unique ids, every id resolving (abilities and moves against that forme's roster
// entry), the exact integer sums (review finding N4: every list states its denominator, and the
// weights, the no-item, empty-slot, residual and unmapped masses add up to it), the truncation
// limits and the vocabulary's Stat Point limit. `usagePriorHash` is the `prior_hash` of the belief
// identity. The module holds no game rule and no Smogon data: deriving a prior is another module.

import { ajvFor } from "@areneon/schema/ajv";
import usagePriorSchema from "@areneon/schema/contracts/v1/usage-prior.schema.json" with { type: "json" };
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";

export type { UsagePrior };

export class UsagePriorError extends Error {
  override name = "UsagePriorError";
}

const validPrior = ajvFor("contract").compile(usagePriorSchema);

function bad(message: string): never {
  throw new UsagePriorError(message);
}

/** Strictly ascending by code unit: sorted, no repeat. */
function ordered(ids: readonly string[], what: string): void {
  ids.forEach((id, k) => {
    if (k > 0 && !((ids[k - 1] as string) < id)) bad(`${what}: ${id} is repeated or out of order`);
  });
}

type Spread = UsagePrior["species"][number]["formes"][number]["spreads"]["entries"][number];
const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;

/** Nature id, then the six Stat Points in stat order, each by code unit or number. */
function compareSpreads(a: Spread, b: Spread): number {
  if (a.nature !== b.nature) return a.nature < b.nature ? -1 : 1;
  for (const stat of STATS) if (a.stat_points[stat] !== b.stat_points[stat]) return a.stat_points[stat] - b.stat_points[stat];
  return 0;
}

const total = (values: readonly number[]): number => values.reduce((sum, v) => sum + v, 0);

/** The sums and limits of one forme's four lists. */
function checkForme(forme: UsagePrior["species"][number]["formes"][number], species: string, where: string, prior: UsagePrior, vocabulary: TeamVocabulary): void {
  const entry = vocabulary.roster.find((r) => r.id === forme.id) ?? bad(`${where}: forme ${forme.id} is not in the vocabulary roster`);
  // A Mega forme's Abilities list is the post-Evolution ability, which the roster entry (the form Team Preview shows) may not list: accept any ability the roster lists.
  const abilityPool = entry.required_item === undefined ? new Set(entry.abilities) : new Set(vocabulary.roster.flatMap((r) => r.abilities));
  const { abilities, items, spreads, moves } = forme;

  ordered(abilities.entries.map((e) => e.id), `${where} abilities`);
  for (const e of abilities.entries) if (!abilityPool.has(e.id)) bad(`${where}: ability ${e.id} is not one the vocabulary lists for ${forme.id}`);
  if (total(abilities.entries.map((e) => e.weight)) + abilities.residual + abilities.unmapped !== abilities.denominator) bad(`${where} abilities: the weights do not sum to the denominator`);
  const holder = forme.holder_abilities;
  if (holder !== undefined) {
    // Only a Mega forme has a state before the Evolution; its abilities are the preview species' (the form Team Preview shows).
    const preview = entry.required_item !== undefined ? vocabulary.roster.find((r) => r.id === species) : undefined;
    if (preview === undefined || species === forme.id) bad(`${where}: holder_abilities on a forme that is not a Mega forme`);
    const pool = new Set(preview.abilities);
    ordered(holder.entries.map((e) => e.id), `${where} holder abilities`);
    for (const e of holder.entries) if (!pool.has(e.id)) bad(`${where}: holder ability ${e.id} is not one the vocabulary lists for ${species}`);
    if (total(holder.entries.map((e) => e.weight)) + holder.residual + holder.unmapped !== holder.denominator) bad(`${where} holder abilities: the weights do not sum to the denominator`);
  }

  ordered(items.entries.map((e) => e.id), `${where} items`);
  if (items.entries.length > prior.derivation.k_items) bad(`${where} items: more entries than k_items`);
  for (const e of items.entries) if (!vocabulary.items.includes(e.id)) bad(`${where}: item ${e.id} is not in the vocabulary`);
  if (total(items.entries.map((e) => e.weight)) + items.no_item + items.residual + items.unmapped !== items.denominator) bad(`${where} items: the weights do not sum to the denominator`);

  if (spreads.entries.length > prior.derivation.k_spreads) bad(`${where} spreads: more entries than k_spreads`);
  spreads.entries.forEach((s, k) => {
    if (!vocabulary.natures.some((n) => n.id === s.nature)) bad(`${where}: nature ${s.nature} is not in the vocabulary`);
    if (total(Object.values(s.stat_points)) > vocabulary.rules.ev_limit) bad(`${where}: spread ${k} holds more Stat Points than the format's limit`);
    if (k > 0 && compareSpreads(spreads.entries[k - 1] as Spread, s) >= 0) bad(`${where} spreads: spread ${k} is repeated or out of order`);
  });
  if (total(spreads.entries.map((s) => s.weight)) + spreads.residual + spreads.unmapped !== spreads.denominator) bad(`${where} spreads: the weights do not sum to the denominator`);

  ordered(moves.entries.map((e) => e.id), `${where} moves`);
  if (moves.entries.length > prior.derivation.k_moves) bad(`${where} moves: more entries than k_moves`);
  for (const e of moves.entries) {
    if (!entry.moves.includes(e.id)) bad(`${where}: move ${e.id} is not in the roster move pool of ${forme.id}`);
    if (e.inclusion > moves.denominator) bad(`${where}: move ${e.id} is included more often than there are sets`);
  }
  if (moves.denominator !== items.denominator) bad(`${where} moves: the denominator is not the items denominator (the number of sets)`);
  if (moves.empty_slots > (moves.slots - 1) * moves.denominator) bad(`${where} moves: a set cannot average ${moves.slots} or more empty slots`);
  if (total(moves.entries.map((e) => e.inclusion)) + moves.empty_slots + moves.residual + moves.unmapped !== moves.slots * moves.denominator) bad(`${where} moves: the masses do not sum to slots times the denominator`);
}

/**
 * Parse a stored prior: the schema, then what the profile cannot say, against `vocabulary`, which it
 * must name by hash (and so its format and engine-data table). Refuses anything else.
 */
export function readUsagePrior(value: unknown, vocabulary: TeamVocabulary): UsagePrior {
  if (!validPrior(value)) return bad(`invalid usage prior: ${JSON.stringify(validPrior.errors)}`);
  const prior = structuredClone(value) as unknown as UsagePrior;
  if (prior.vocabulary_hash !== teamVocabularyHash(vocabulary)) bad("vocabulary_hash is not the hash of the given vocabulary");
  if (prior.format_id !== vocabulary.format_id) bad("format_id differs from the vocabulary's");
  if (prior.engine_data_hash !== vocabulary.engine_data_hash) bad("engine_data_hash differs from the vocabulary's");
  if (prior.source.kind === "smogon-chaos") ordered(prior.source.inputs.map((i) => i.source_id), "source inputs");

  ordered(prior.species.map((s) => s.id), "species");
  for (const species of prior.species) {
    if (!vocabulary.roster.some((r) => r.id === species.id)) bad(`species ${species.id} is not in the vocabulary roster`);
    ordered(species.formes.map((f) => f.id), `species ${species.id} formes`);
    if (total(species.formes.map((f) => f.weight)) !== species.denominator) bad(`species ${species.id}: the forme weights do not sum to the denominator`);
    for (const forme of species.formes) checkForme(forme, species.id, `species ${species.id} forme ${forme.id}`, prior, vocabulary);
  }
  return prior;
}

/** SHA-256 of the prior's JCS bytes: `prior_hash`, a part of the belief identity (ADR-025 section 5). */
export const usagePriorHash = (prior: UsagePrior): string => sha256Hex(jcs(prior as unknown as Json));
