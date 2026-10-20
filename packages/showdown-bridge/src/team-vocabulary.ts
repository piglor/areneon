// What the team generator may draw from (ADR-013 section 2), read from the
// pinned engine for one format. It reuses the engine-data export of ADR-010
// section 3 (`exportEngineData`: the species, move, item and ability tables
// and their hash) and adds what that table deliberately leaves out: the
// roster the validator accepts, each roster species' abilities and move pool,
// the natures, the items the format does not ban, and the rule-table values
// (adjustLevel, evLimit, Item Clause, team sizes).
//
// Nothing here is a game rule. Whether a species, ability, move, nature or
// item is allowed is the pinned `TeamValidator`'s verdict on a probe set; the
// rule values are read from the format's resolved rule table. A probe holds
// one Stat Point so the engine's "exactly 0 Stat Points" check stays quiet,
// and nothing else about it carries a rule. Every list is sorted by id, and
// the export is hashed (`teamVocabularyHash`) so a pin bump that changes any
// verdict changes the hash and never silently remaps the roster.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import teamVocabularySchema from "@areneon/schema/contracts/v1/team-vocabulary.schema.json" with { type: "json" };
import type { TeamVocabulary } from "@areneon/schema/types/v1/team-vocabulary.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type EngineData, engineDataHash, exportEngineData, readEngineData } from "./engine-data.ts";
import { type Engine, EngineError, type SimFormat, type SimRuleTable } from "./engine.ts";

export type { TeamVocabulary };

export class TeamVocabularyError extends Error {
  override name = "TeamVocabularyError";
}

/** The golden export, beside the engine-data table. */
export const TEAM_VOCABULARY_FILE = "team-vocabulary.json";
/** The vocabulary's hash and counts, small enough to review in a diff. */
export const TEAM_VOCABULARY_SUMMARY = "team-vocabulary.summary.json";

const validVocabulary = ajvFor("contract").compile(teamVocabularySchema);

type Raw = Record<string, unknown>;

/** The slice of the pinned validator the export reads. */
interface RawValidator {
  validateSet(set: Raw, teamHas: Raw): string[] | null;
  checkItem(set: Raw, item: unknown, setHas: Raw): string | null;
}

/** The slice of a Showdown `ModdedDex` the export reads. */
interface RawDex {
  species: { all(): unknown[] };
  moves: { all(): unknown[] };
  items: { all(): unknown[] };
  natures: { all(): unknown[] };
}

interface RawSim {
  Dex: {
    formats: { get(id: string): SimFormat; getRuleTable(format: SimFormat): SimRuleTable };
    mod(name: string): RawDex;
  };
  TeamValidator: { get(formatId: string): RawValidator };
}

const bad = (message: string): never => {
  throw new TeamVocabularyError(message);
};

function rec(value: unknown, what: string): Raw {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad(`${what}: not an object`);
  return value as Raw;
}
function str(value: unknown, what: string): string {
  if (typeof value !== "string" || value === "") bad(`${what}: not a non-empty string`);
  return value as string;
}

const byId = <T extends { id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : 1);
const unique = (values: string[]): string[] => [...new Set(values)].sort();

/** Entries the format's mod plays with: the engine marks the rest nonstandard. */
const standard = (all: unknown[], what: string): Raw[] =>
  all.map((e) => rec(e, what)).filter((e) => e.exists === true && e.isNonstandard === null);

/** A rule-table value the engine leaves unset is refused, never defaulted. */
function ruleInt(value: number | null, what: string, min: number): number {
  if (value === null || !Number.isSafeInteger(value) || value < min) bad(`rule table ${what}: ${String(value)} is not an integer of at least ${min}`);
  return value as number;
}

function ruleValues(table: SimRuleTable): TeamVocabulary["rules"] {
  const clause = table.valueRules.get("itemclause");
  return {
    adjust_level: ruleInt(table.adjustLevel, "adjustLevel", 1),
    ev_limit: ruleInt(table.evLimit, "evLimit", 1),
    item_clause: ruleInt(clause === undefined ? null : Number(clause), "itemclause", 1),
    min_team_size: ruleInt(table.minTeamSize, "minTeamSize", 1),
    max_team_size: ruleInt(table.maxTeamSize, "maxTeamSize", 1),
    picked_team_size: ruleInt(table.pickedTeamSize, "pickedTeamSize", 1),
  };
}

const STATS = ["atk", "def", "spa", "spd", "spe"] as const;

/** One nature as the engine lists it: the stat it raises and lowers, or "" for neutral. */
function natureOf(entry: Raw): TeamVocabulary["natures"][number] {
  const stat = (value: unknown, what: string): TeamVocabulary["natures"][number]["plus"] => {
    if (value === undefined) return "";
    return (STATS as readonly unknown[]).includes(value) ? (value as (typeof STATS)[number]) : bad(`nature ${what}: ${String(value)} is not a stat`);
  };
  return {
    id: str(entry.id, "nature id"),
    name: str(entry.name, "nature name"),
    plus: stat(entry.plus, "plus"),
    minus: stat(entry.minus, "minus"),
  };
}

/**
 * Export the team vocabulary of `formatId` from the pinned engine. `table` is
 * the engine-data table for the same format (exported here when omitted); the
 * result names it by hash and is checked against it (`readTeamVocabulary`).
 * Refuses any value of a shape it does not know.
 */
export function exportTeamVocabulary(engine: Engine, formatId: string, table?: EngineData): TeamVocabulary {
  const sim = engine.sim as unknown as RawSim;
  const format = sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  const data = table ?? exportEngineData(engine, formatId);
  const dex = sim.Dex.mod(format.mod);
  const validator = sim.TeamValidator.get(formatId);
  const rules = ruleValues(sim.Dex.formats.getRuleTable(format));

  const natureEntries = standard(dex.natures.all(), "nature");
  // A neutral nature, so a probe never trips the "0 Stat Points" check that the engine reserves for non-neutral ones.
  const neutral = natureEntries.map(natureOf).filter((n) => n.plus === n.minus).sort(byId)[0] ?? bad("the engine lists no neutral nature");
  const moveNames = standard(dex.moves.all(), "move").map((m) => ({ id: str(m.id, "move id"), name: str(m.name, "move name") }));

  const probe = (species: Raw, ability: string, move: string, nature: string = neutral.name): Raw => ({
    name: "",
    species: str(species.name, "species name"),
    item: itemOf(species),
    ability,
    moves: [move],
    nature,
    gender: "",
    evs: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
  });
  const accepted = (set: Raw): boolean => validator.validateSet(set, {}) === null;

  const roster: TeamVocabulary["roster"] = [];
  let reference: { species: Raw; ability: string; move: string } | undefined;
  for (const species of standard(dex.species.all(), "species")) {
    const slots = Object.values(rec(species.abilities, "species abilities")).map((a) => str(a, "ability name"));
    // The pool is judged with the first ability the validator takes with any move.
    let pool: string[] = [];
    let first = "";
    for (const ability of unique(slots)) {
      pool = moveNames.filter((m) => accepted(probe(species, ability, m.name))).map((m) => m.id);
      if (pool.length > 0) {
        first = ability;
        break;
      }
    }
    if (pool.length === 0) continue;
    const move = moveNames.find((m) => m.id === pool[0]) as { name: string };
    const abilities = unique(slots).filter((a) => accepted(probe(species, a, move.name)));
    reference ??= { species, ability: first, move: move.name };
    const required = itemOf(species);
    roster.push({
      id: str(species.id, "species id"),
      abilities: abilities.map((a) => toId(a)).sort() as [string, ...string[]],
      moves: pool.sort() as [string, ...string[]],
      ...(required === "" ? {} : { required_item: toId(required) }),
    });
  }
  const base = reference ?? bad("the validator accepts no species");

  const natures = natureEntries
    .map(natureOf)
    .filter((n) => accepted(probe(base.species, base.ability, base.move, n.name)))
    .sort(byId);
  const items = standard(dex.items.all(), "item")
    .filter((i) => validator.checkItem(probe(base.species, base.ability, base.move), i, {}) === null)
    .map((i) => str(i.id, "item id"))
    .sort();

  return readTeamVocabulary(
    {
      schema_version: 1,
      showdown_commit: engine.commit,
      format_id: formatId,
      mod: format.mod,
      engine_data_hash: engineDataHash(data),
      rules,
      natures,
      items,
      roster: roster.sort(byId),
    },
    data,
  );
}

const toId = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The item a forme must hold (a Mega forme's stone), or "". */
function itemOf(species: Raw): string {
  if (typeof species.requiredItem === "string") return species.requiredItem;
  const several = species.requiredItems;
  return Array.isArray(several) && typeof several[0] === "string" ? several[0] : "";
}

/** Strictly ascending: sorted by code unit, no repeat. */
function ordered(ids: string[], what: string): void {
  ids.forEach((id, k) => {
    if (k > 0 && !((ids[k - 1] as string) < id)) bad(`${what}: ${id} is repeated or out of order`);
  });
}

/**
 * Parse a stored vocabulary: the schema, then what the contract profile cannot
 * say (sorted and unique ids, sane team sizes). With the engine-data table it
 * was exported with, also that every id resolves in it and that it names that
 * table by hash. Refuses anything else.
 */
export function readTeamVocabulary(value: unknown, table?: EngineData): TeamVocabulary {
  if (!validVocabulary(value)) return bad(`invalid team vocabulary: ${JSON.stringify(validVocabulary.errors)}`);
  const vocabulary = value as unknown as TeamVocabulary;
  const { rules } = vocabulary;
  if (rules.min_team_size > rules.max_team_size) bad("rules: min_team_size exceeds max_team_size");
  if (rules.picked_team_size > rules.max_team_size) bad("rules: picked_team_size exceeds max_team_size");
  ordered(vocabulary.natures.map((n) => n.id), "natures");
  ordered(vocabulary.items, "items");
  ordered(vocabulary.roster.map((r) => r.id), "roster");
  for (const entry of vocabulary.roster) {
    ordered(entry.abilities, `roster ${entry.id} abilities`);
    ordered(entry.moves, `roster ${entry.id} moves`);
  }
  if (table === undefined) return vocabulary;

  readEngineData(table);
  if (vocabulary.engine_data_hash !== engineDataHash(table)) bad("engine_data_hash is not the hash of the given engine-data table");
  for (const key of ["showdown_commit", "format_id", "mod"] as const) {
    if (vocabulary[key] !== table[key]) bad(`${key} differs from the engine-data table's`);
  }
  const species = new Map(table.species.map((s) => [s.id, s]));
  const abilities = new Set(table.abilities.map((a) => a.id));
  const moves = new Set(table.moves.map((m) => m.id));
  const items = new Set(table.items.map((i) => i.id));
  for (const item of vocabulary.items) if (!items.has(item)) bad(`item ${item} is not in the engine-data table`);
  for (const entry of vocabulary.roster) {
    const known = species.get(entry.id) ?? bad(`roster species ${entry.id} is not in the engine-data table`);
    for (const ability of entry.abilities) {
      if (!abilities.has(ability) || !known.abilities.includes(ability)) bad(`roster ${entry.id}: ability ${ability} is not one of its engine-data abilities`);
    }
    for (const move of entry.moves) if (!moves.has(move)) bad(`roster ${entry.id}: move ${move} is not in the engine-data table`);
    if (entry.required_item !== undefined && !items.has(entry.required_item)) bad(`roster ${entry.id}: required item ${entry.required_item} is not in the engine-data table`);
  }
  return vocabulary;
}

/** SHA-256 of the vocabulary's JCS bytes: the vocabulary hash (`vocabulary_hash` in the team manifests). */
export function teamVocabularyHash(vocabulary: TeamVocabulary): string {
  return createHash("sha256").update(jcs(vocabulary as unknown as Json), "utf8").digest("hex");
}

/** The reviewable summary committed beside the golden vocabulary. */
export function teamVocabularySummary(vocabulary: TeamVocabulary): {
  vocabulary_hash: string;
  engine_data_hash: string;
  showdown_commit: string;
  counts: Record<string, number>;
} {
  return {
    vocabulary_hash: teamVocabularyHash(vocabulary),
    engine_data_hash: vocabulary.engine_data_hash,
    showdown_commit: vocabulary.showdown_commit,
    counts: {
      roster: vocabulary.roster.length,
      natures: vocabulary.natures.length,
      items: vocabulary.items.length,
      pool_moves: vocabulary.roster.reduce((sum, r) => sum + r.moves.length, 0),
    },
  };
}
