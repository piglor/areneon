// The role sets of the pinned engine for Champions doubles random battles
// (ADR-020 section 3), read for one format. The source is
// `data/random-battles/<mod>/doubles-sets.json` in the pinned Showdown checkout
// (MIT). Each entry names a species and its sets: a role, a move pool and the
// abilities the role plays, as display names. This module maps every name to an
// engine id through the engine-data table and checks the result against the
// team vocabulary: a species must be in the roster, and a move must be one the
// pinned validator accepts on that species (the roster lists exactly those).
// An ability must be one the roster lists too, except for a forme that needs a
// held item (a Mega forme): its sets name the abilities of the forme it comes
// from, the form Team Preview shows, so a consumer plays such an entry as that
// base forme plus the stone, never as a literal Mega forme with those abilities.
// Anything that does not resolve refuses the whole export; the
// table is never partial. The random-battle level, preferred types and Tera
// types are dropped, and no role logic is ported: nothing turns a role into an
// item or a spread. Limit: JSON.parse keeps the last of two equal species keys
// in the source. Every list is sorted, and the export is hashed
// (`roleSetTableHash`) so a pin bump that changes the file changes the hash.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import roleSetTableSchema from "@areneon/schema/contracts/v1/role-set-table.schema.json" with { type: "json" };
import type { RoleSetTable } from "@areneon/schema/types/v1/role-set-table.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type EngineData, engineDataHash, exportEngineData } from "./engine-data.ts";
import { DEFAULT_SHOWDOWN_DIR, type Engine, EngineError, type SimFormat } from "./engine.ts";
import { readTeamVocabulary, type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";

export type { RoleSetTable };

export class RoleSetError extends Error {
  override name = "RoleSetError";
}

/** The golden export, beside the team vocabulary. */
export const ROLE_SETS_FILE = "role-sets.json";
/** The table's hash and counts, small enough to review in a diff. */
export const ROLE_SETS_SUMMARY = "role-sets.summary.json";

const validTable = ajvFor("contract").compile(roleSetTableSchema);

type Raw = Record<string, unknown>;

const bad = (message: string): never => {
  throw new RoleSetError(message);
};

function rec(value: unknown, what: string): Raw {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad(`${what}: not an object`);
  return value as Raw;
}
function names(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || value.length === 0 || value.some((v) => typeof v !== "string" || v === "")) bad(`${what}: not a non-empty list of names`);
  return value as string[];
}
/** A list the source may carry that this table drops: its shape is still checked. */
function droppedList(value: unknown, what: string): void {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) bad(`${what}: not a list of strings`);
}

const SET_KEYS = ["role", "movepool", "abilities", "preferredTypes", "teraTypes"];
const ENTRY_KEYS = ["level", "sets"];

const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const unique = (values: string[]): string[] => [...new Set(values)].sort(cmp);

/** The slice of a Showdown `ModdedDex` the export reads: an item's `megaStone` maps each pre-Mega forme's name to its Mega forme's name. */
interface RawDex {
  items: { get(id: string): { megaStone?: unknown } };
}

/** The source file for a format: the format's mod, doubles only (the file is the doubles sets). */
export function roleSetSourcePath(format: SimFormat): string {
  if (format.gameType !== "doubles") bad(`format ${format.id} is ${format.gameType}, not doubles: no doubles role sets apply`);
  return `data/random-battles/${format.mod}/doubles-sets.json`;
}

/**
 * Export the role sets of `formatId` from the pinned engine checkout at `dir`
 * (it defaults to the pinned checkout; a test passes a doctored copy).
 * `vocabulary` is the team vocabulary of the same format and `table` its
 * engine-data table (exported here when omitted). Refuses a missing or
 * malformed source, a name that does not resolve, a species outside the
 * roster, and an ability or move the roster does not list for the species.
 */
export function exportRoleSets(engine: Engine, formatId: string, vocabulary: TeamVocabulary, table?: EngineData, dir: string = DEFAULT_SHOWDOWN_DIR): RoleSetTable {
  const format = engine.sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  const data = table ?? exportEngineData(engine, formatId);
  readTeamVocabulary(vocabulary, data);
  if (vocabulary.format_id !== formatId) bad(`the vocabulary is for ${vocabulary.format_id}, not ${formatId}`);
  const sourcePath = roleSetSourcePath(format);
  let bytes: Buffer;
  try {
    bytes = readFileSync(join(dir, sourcePath));
  } catch {
    return bad(`cannot read ${sourcePath} from the engine checkout`);
  }
  let source: unknown;
  try {
    source = JSON.parse(bytes.toString("utf8"));
  } catch {
    return bad(`${sourcePath}: not valid JSON`);
  }

  const speciesIds = new Set(data.species.map((s) => s.id));
  const moveIds = new Map(data.moves.map((m) => [m.name, m.id]));
  const abilityIds = new Map(data.abilities.map((a) => [a.name, a.id]));
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const dex = (engine.sim as unknown as { Dex: { mod(name: string): RawDex } }).Dex.mod(format.mod);
  const speciesNames = new Map(data.species.map((s) => [s.name, s.id]));

  /** A Mega forme's sets name the abilities of the forme it comes from (the form a player's Team Preview shows): the roster entries of the formes its stone changes into it. */
  const preMegaAbilities = (id: string, item: string): string[] => {
    const stone = dex.items.get(item).megaStone;
    if (typeof stone !== "object" || stone === null || Array.isArray(stone)) return bad(`species ${id}: item ${item} names no Mega forme map (a roster entry with a required item must be a Mega forme)`);
    const mega = data.species.find((x) => x.id === id)?.name;
    return Object.entries(stone as Record<string, unknown>)
      .filter(([, to]) => to === mega)
      .flatMap(([from]) => roster.get(speciesNames.get(from) ?? "")?.abilities ?? []);
  };

  const species: RoleSetTable["species"] = [];
  for (const [id, raw] of Object.entries(rec(source, sourcePath))) {
    const entry = rec(raw, `species ${id}`);
    for (const key of Object.keys(entry)) if (!ENTRY_KEYS.includes(key)) bad(`species ${id}: unknown field ${key}`);
    if (!Number.isSafeInteger(entry.level)) bad(`species ${id}: level is not an integer`);
    if (!speciesIds.has(id)) bad(`species ${id} is not in the engine-data table`);
    const known = roster.get(id) ?? bad(`species ${id} is not in the team vocabulary roster`);
    const list = entry.sets;
    if (!Array.isArray(list) || list.length === 0) bad(`species ${id}: no sets`);
    const allowed = new Set([...known.abilities, ...(known.required_item === undefined ? [] : preMegaAbilities(id, known.required_item))]);
    const sets: SetOf[] = (list as unknown[]).map((item, k) => {
      const where = `species ${id} set ${k}`;
      const set = rec(item, where);
      for (const key of Object.keys(set)) if (!SET_KEYS.includes(key)) bad(`${where}: unknown field ${key}`);
      if (typeof set.role !== "string" || set.role === "") bad(`${where}: role is not a non-empty string`);
      if (set.preferredTypes !== undefined) droppedList(set.preferredTypes, `${where} preferredTypes`);
      if (set.teraTypes !== undefined) droppedList(set.teraTypes, `${where} teraTypes`);
      const moves = names(set.movepool, `${where} movepool`).map((name) => {
        const move = moveIds.get(name) ?? bad(`${where}: move ${name} is not in the engine-data table`);
        if (!known.moves.includes(move)) bad(`${where}: move ${name} is not accepted by the validator on ${id}`);
        return move;
      });
      const abilities = names(set.abilities, `${where} abilities`).map((name) => {
        const ability = abilityIds.get(name) ?? bad(`${where}: ability ${name} is not in the engine-data table`);
        if (!allowed.has(ability)) bad(`${where}: ability ${name} is not accepted by the validator on ${id}`);
        return ability;
      });
      return { role: set.role as string, moves: unique(moves) as [string, ...string[]], abilities: unique(abilities) as [string, ...string[]] };
    });
    species.push({ id, sets: sets.sort(compareSets) as RoleSetTable["species"][number]["sets"] });
  }
  species.sort((a, b) => cmp(a.id, b.id));

  return readRoleSetTable(
    {
      schema_version: 1,
      showdown_commit: engine.commit,
      format_id: formatId,
      engine_data_hash: engineDataHash(data),
      vocabulary_hash: teamVocabularyHash(vocabulary),
      source_path: sourcePath,
      source_sha256: createHash("sha256").update(bytes).digest("hex"),
      species,
    },
    vocabulary,
  );
}

type SetOf = RoleSetTable["species"][number]["sets"][number];

/** Role, then moves, then abilities, each by code unit; a list compares as its joined ids. */
function compareSets(a: SetOf, b: SetOf): number {
  return cmp(a.role, b.role) || cmp(a.moves.join(" "), b.moves.join(" ")) || cmp(a.abilities.join(" "), b.abilities.join(" "));
}

/** Strictly ascending: sorted by code unit, no repeat. */
function ordered(ids: string[], what: string): void {
  ids.forEach((id, k) => {
    if (k > 0 && !((ids[k - 1] as string) < id)) bad(`${what}: ${id} is repeated or out of order`);
  });
}

/**
 * Parse a stored table: the schema, then what the contract profile cannot say
 * (sorted and unique ids and sets). With the team vocabulary it was exported
 * with, also that it names that vocabulary and its engine-data table by hash,
 * and that every species is in the roster and every ability and move is one
 * the roster lists for it. Refuses anything else.
 */
export function readRoleSetTable(value: unknown, vocabulary?: TeamVocabulary): RoleSetTable {
  if (!validTable(value)) return bad(`invalid role-set table: ${JSON.stringify(validTable.errors)}`);
  const table = value as unknown as RoleSetTable;
  ordered(table.species.map((s) => s.id), "species");
  for (const entry of table.species) {
    entry.sets.forEach((set, k) => {
      ordered(set.moves, `species ${entry.id} set ${k} moves`);
      ordered(set.abilities, `species ${entry.id} set ${k} abilities`);
      if (k > 0 && compareSets(entry.sets[k - 1] as SetOf, set) >= 0) bad(`species ${entry.id}: set ${k} is repeated or out of order`);
    });
  }
  if (vocabulary === undefined) return table;

  if (table.vocabulary_hash !== teamVocabularyHash(vocabulary)) bad("vocabulary_hash is not the hash of the given vocabulary");
  for (const key of ["showdown_commit", "format_id", "engine_data_hash"] as const) {
    if (table[key] !== vocabulary[key]) bad(`${key} differs from the vocabulary's`);
  }
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  // A Mega forme's sets name the abilities of the forme it comes from, which the exporter checks against the engine's item data; this reader has only the vocabulary, so it accepts any ability a non-Mega roster entry lists.
  const baseAbilities = new Set(vocabulary.roster.filter((r) => r.required_item === undefined).flatMap((r) => r.abilities));
  for (const entry of table.species) {
    const known = roster.get(entry.id) ?? bad(`species ${entry.id} is not in the vocabulary roster`);
    for (const set of entry.sets) {
      for (const move of set.moves) if (!known.moves.includes(move)) bad(`species ${entry.id}: move ${move} is not in its roster move pool`);
      for (const ability of set.abilities) {
        if (!(known.required_item === undefined ? known.abilities.includes(ability) : known.abilities.includes(ability) || baseAbilities.has(ability))) bad(`species ${entry.id}: ability ${ability} is not in its roster abilities`);
      }
    }
  }
  return table;
}

/** SHA-256 of the table's JCS bytes: the role-set table hash (named by the generator config). */
export function roleSetTableHash(table: RoleSetTable): string {
  return createHash("sha256").update(jcs(table as unknown as Json), "utf8").digest("hex");
}

/** The reviewable summary committed beside the golden table. */
export function roleSetSummary(table: RoleSetTable): {
  role_set_table_hash: string;
  vocabulary_hash: string;
  engine_data_hash: string;
  showdown_commit: string;
  source_sha256: string;
  counts: Record<string, number>;
} {
  const sets = table.species.flatMap((s) => s.sets);
  return {
    role_set_table_hash: roleSetTableHash(table),
    vocabulary_hash: table.vocabulary_hash,
    engine_data_hash: table.engine_data_hash,
    showdown_commit: table.showdown_commit,
    source_sha256: table.source_sha256,
    counts: {
      species: table.species.length,
      sets: sets.length,
      roles: new Set(sets.map((s) => s.role)).size,
      set_moves: sets.reduce((sum, s) => sum + s.moves.length, 0),
      set_abilities: sets.reduce((sum, s) => sum + s.abilities.length, 0),
    },
  };
}
