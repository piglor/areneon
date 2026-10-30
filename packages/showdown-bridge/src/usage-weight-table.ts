// The usage weight table (ADR-020 section 4): per species (a roster entry), integer weights of items,
// abilities, moves, Stat Point spreads and teammates, derived offline from one recorded monthly Smogon
// chaos file. The schema checks the shape; this reader adds what the contract profile cannot say,
// against the team vocabulary the table was built for: sorted and unique ids, every id resolving
// (abilities and moves against the species' own roster entry), the exact sums (every list is
// 1,000,000 ppm: entries, truncated, unmapped and the semantic key), the top-K limits, the Stat Point
// limit and the integer roots of `w_species` and the teammate weights. `usageTableHash` is SHA-256 of
// the JCS bytes. The module holds no game rule and no Smogon data: deriving a table is a script.

import { ajvFor } from "@areneon/schema/ajv";
import usageWeightTableSchema from "@areneon/schema/contracts/v1/usage-weight-table.schema.json" with { type: "json" };
import type { UsageWeightTable } from "@areneon/schema/types/v1/usage-weight-table.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";

export type { UsageWeightTable };

export class UsageWeightTableError extends Error {
  override name = "UsageWeightTableError";
}

const validTable = ajvFor("contract").compile(usageWeightTableSchema);

const PPM_TOTAL = 1_000_000;
const PPM_DIGITS = 6n;

function bad(message: string): never {
  throw new UsageWeightTableError(message);
}

/** Strictly ascending by code unit: sorted, no repeat. */
function ordered(ids: readonly string[], what: string): void {
  ids.forEach((id, k) => {
    if (k > 0 && !((ids[k - 1] as string) < id)) bad(`${what}: ${id} is repeated or out of order`);
  });
}

/** The largest x with x ** root <= value, in exact integer arithmetic. */
export function integerRoot(value: bigint, root: bigint): bigint {
  if (value < 0n || root < 1n) bad("integer root of a negative value or a root below 1");
  if (root === 1n || value < 2n) return value;
  let low = 1n;
  let high = 1n << (BigInt(value.toString(2).length) / root + 1n);
  while (low < high) {
    const mid = (low + high + 1n) / 2n;
    if (mid ** root <= value) low = mid;
    else high = mid - 1n;
  }
  return low;
}

/** floor((ppm * 10^(6 * (root - 1)))^(1/root)): the weight a ppm becomes under `root`. */
export function rootedWeight(ppm: number, root: number): number {
  return Number(integerRoot(BigInt(ppm) * 10n ** (PPM_DIGITS * BigInt(root - 1)), BigInt(root)));
}

type Species = UsageWeightTable["species"][number];
type Spread = Species["spreads"]["entries"][number];
const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;

/** Nature id, then the six Stat Points in stat order. */
function compareSpreads(a: Spread, b: Spread): number {
  if (a.nature !== b.nature) return a.nature < b.nature ? -1 : 1;
  for (const stat of STATS) if (a.stat_points[stat] !== b.stat_points[stat]) return a.stat_points[stat] - b.stat_points[stat];
  return 0;
}

const total = (values: readonly number[]): number => values.reduce((sum, v) => sum + v, 0);

function sumsToTotal(entries: readonly { ppm: number }[], masses: readonly number[], where: string): void {
  if (total(entries.map((e) => e.ppm)) + total(masses) !== PPM_TOTAL) bad(`${where}: the masses do not sum to 1,000,000 ppm`);
}

/** The sums, limits and roots of one species' five lists. */
function checkSpecies(species: Species, table: UsageWeightTable, vocabulary: TeamVocabulary): void {
  const where = `species ${species.id}`;
  const entry = vocabulary.roster.find((r) => r.id === species.id) ?? bad(`${where} is not in the vocabulary roster`);
  const limits = table.derivation;
  if (species.w_species !== rootedWeight(species.usage_ppm, limits.species_root)) bad(`${where}: w_species is not the integer root of usage_ppm`);

  const { abilities, items, moves, spreads, teammates } = species;
  ordered(abilities.entries.map((e) => e.id), `${where} abilities`);
  for (const e of abilities.entries) if (!entry.abilities.includes(e.id)) bad(`${where}: ability ${e.id} is not one the roster lists`);
  sumsToTotal(abilities.entries, [abilities.unmapped_ppm], `${where} abilities`);

  ordered(items.entries.map((e) => e.id), `${where} items`);
  if (items.entries.length > limits.k_items) bad(`${where} items: more entries than k_items`);
  for (const e of items.entries) if (!vocabulary.items.includes(e.id)) bad(`${where}: item ${e.id} is not in the vocabulary`);
  sumsToTotal(items.entries, [items.truncated_ppm, items.unmapped_ppm, items.no_item_ppm], `${where} items`);

  ordered(moves.entries.map((e) => e.id), `${where} moves`);
  if (moves.entries.length > limits.k_moves) bad(`${where} moves: more entries than k_moves`);
  for (const e of moves.entries) if (!entry.moves.includes(e.id)) bad(`${where}: move ${e.id} is not in the roster move pool`);
  sumsToTotal(moves.entries, [moves.truncated_ppm, moves.unmapped_ppm, moves.empty_ppm], `${where} moves`);

  if (spreads.entries.length > limits.k_spreads) bad(`${where} spreads: more entries than k_spreads`);
  spreads.entries.forEach((s, k) => {
    if (!vocabulary.natures.some((n) => n.id === s.nature)) bad(`${where}: nature ${s.nature} is not in the vocabulary`);
    if (total(Object.values(s.stat_points)) > vocabulary.rules.ev_limit) bad(`${where}: spread ${k} holds more Stat Points than the format's limit`);
    if (k > 0 && compareSpreads(spreads.entries[k - 1] as Spread, s) >= 0) bad(`${where} spreads: spread ${k} is repeated or out of order`);
  });
  sumsToTotal(spreads.entries, [spreads.truncated_ppm, spreads.unmapped_ppm], `${where} spreads`);

  ordered(teammates.entries.map((e) => e.id), `${where} teammates`);
  if (teammates.entries.length > limits.k_teammates) bad(`${where} teammates: more entries than k_teammates`);
  for (const e of teammates.entries) {
    if (!vocabulary.roster.some((r) => r.id === e.id)) bad(`${where}: teammate ${e.id} is not in the roster`);
    if (e.weight !== rootedWeight(e.ppm, limits.teammate_root)) bad(`${where}: a teammate weight is not the root of its ppm`);
  }
  // A species with no positive teammate weight is the one list that may sum to 0.
  if (teammates.entries.length > 0 || teammates.truncated_ppm > 0 || teammates.unmapped_ppm > 0) sumsToTotal(teammates.entries, [teammates.truncated_ppm, teammates.unmapped_ppm], `${where} teammates`);
}

/**
 * Parse a stored table: the schema, then what the profile cannot say, against `vocabulary`, which it
 * must name by hash (and so its format and engine-data table). Refuses anything else.
 */
export function readUsageWeightTable(value: unknown, vocabulary: TeamVocabulary): UsageWeightTable {
  if (!validTable(value)) return bad(`invalid usage weight table: ${JSON.stringify(validTable.errors)}`);
  const table = structuredClone(value) as unknown as UsageWeightTable;
  if (table.vocabulary_hash !== teamVocabularyHash(vocabulary)) bad("vocabulary_hash is not the hash of the given vocabulary");
  if (table.format_id !== vocabulary.format_id) bad("format_id differs from the vocabulary's");
  if (table.engine_data_hash !== vocabulary.engine_data_hash) bad("engine_data_hash differs from the vocabulary's");
  ordered(table.sources.map((s) => s.source_id), "sources");
  ordered(table.species.map((s) => s.id), "species");
  if (total(table.species.map((s) => s.usage_ppm)) + table.unmapped_species_ppm !== PPM_TOTAL) bad("species: the usage shares do not sum to 1,000,000 ppm");
  for (const species of table.species) checkSpecies(species, table, vocabulary);
  return table;
}

/** SHA-256 of the table's JCS bytes: `usageTableHash`. */
export const usageTableHash = (table: UsageWeightTable): string => sha256Hex(jcs(table as unknown as Json));
