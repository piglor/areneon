// The reviewed record of the feature set `probe-outcome-likely/1` (ADR-022 erratum E-22.7.6): config/features/<format>/probe-outcome-likely*.json.
// Integers and closed words only; it chooses among values the engine and the committed tables allow and holds no game rule and no move or
// ability name. A change of any value is a new file and a new `likely_threats_record_hash` (so a new feature set hash), never an edit.
//
// - `source`: where the likelihood of a foe's moves comes from, named by the hash of the committed file it reads.
//   `generator-matched`: the forme's move list of the committed generator-matched usage prior (inclusion numerator over the list's
//   denominator). `usage-table`: the species' `moves` list of the committed Smogon usage weight table (ppm of the species' move-slot
//   mass; `slots` is the number of move slots a set holds, the multiplier of its inclusion).
// - `per_foe` (k): at most this many likely threats per foe and own variant, 4 to 8.
// - `rank_candidates_max` (m) and `screen`: optional, together. With m set a static pre-screen keeps the m candidates with the highest
//   `inclusion x base power x type-chart value x same-type factor x spread factor` before any damage query. `effectiveness_ppm` is the
//   factor of each of the engine's `damage_taken` codes (index = code), `same_type_ppm` the same-type bonus, `spread_ppm` the factor of a
//   move that hits every foe, `variable_power` the base power a move with no fixed one (a callback) counts as.
// - `hp_key_handlers`: the engine's event handler names that can make a damage value read a Pokemon's CURRENT hit points; a move, ability
//   or item that defines one has the hit points in its cache key (E-22.7.4), all others share rolls across hit points. Sorted, unique.
// - `engine_data_hash`: the engine-data table the ids resolve in (and `damage_taken` codes are read from).

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Json, jcs } from "./canonical-json.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "./golden.ts";
import { OUTCOME_CONFIG_DIR } from "./outcome-config.ts";
import { sha256Hex } from "./rules.ts";

export const LIKELY_CONFIG_DIR = OUTCOME_CONFIG_DIR;
const TABLE_DIR = fileURLToPath(new URL("../../../config", import.meta.url));

export class LikelyConfigError extends Error {
  override name = "LikelyConfigError";
}

export const PER_FOE_MIN = 4;
export const PER_FOE_MAX = 8;
export const RANK_MAX_LIMIT = 64;
/** The engine's `damage_taken` codes (0 normal, 1 super effective, 2 resisted, 3 immune): the length of `effectiveness_ppm`. */
export const DAMAGE_TAKEN_CODES = 4;

export type LikelySource = { kind: "generator-matched"; prior_hash: string } | { kind: "usage-table"; usage_table_hash: string; slots: number };

export interface LikelyScreen {
  effectiveness_ppm: number[];
  same_type_ppm: number;
  spread_ppm: number;
  variable_power: number;
}

export interface LikelyConfig {
  schema_version: 1;
  format_id: string;
  engine_data_hash: string;
  source: LikelySource;
  per_foe: number;
  rank_candidates_max?: number;
  screen?: LikelyScreen;
  hp_key_handlers: string[];
}

const HASH = /^[0-9a-f]{64}$/;
const HANDLER = /^[A-Za-z][A-Za-z0-9]*$/;
const MILLION = 1_000_000;

const fail = (message: string): never => {
  throw new LikelyConfigError(message);
};
const record = (value: unknown, what: string): Record<string, unknown> => (typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : fail(`${what} must be an object`));
const exactKeys = (value: Record<string, unknown>, keys: readonly string[], what: string): void => {
  const found = Object.keys(value).sort();
  const wanted = [...keys].sort();
  if (found.length !== wanted.length || found.some((key, index) => key !== wanted[index])) fail(`${what} must have exactly its keys`);
};
const integer = (value: unknown, min: number, max: number, what: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max) fail(`${what} must be an integer from ${min} to ${max}`);
  return value as number;
};
const hash = (value: unknown, what: string): string => {
  if (typeof value !== "string" || !HASH.test(value)) fail(`${what} must be a SHA-256 hex digest`);
  return value as string;
};

/** Read a record: closed keys, integers in range, closed words. Whether a handler name exists in the engine is `checkLikelyConfig`'s. */
export function readLikelyConfig(value: unknown): LikelyConfig {
  const root = record(value, "the probe-outcome-likely record");
  const optional = ["rank_candidates_max", "screen"].filter((key) => key in root);
  exactKeys(root, ["engine_data_hash", "format_id", "hp_key_handlers", "per_foe", "schema_version", "source", ...optional], "the probe-outcome-likely record");
  if (root.schema_version !== 1) fail("schema_version must be 1");
  if (typeof root.format_id !== "string" || root.format_id === "") fail("format_id must be a string");
  if (("rank_candidates_max" in root) !== ("screen" in root)) fail("rank_candidates_max and screen go together");
  const sourceRoot = record(root.source, "source");
  let source: LikelySource;
  if (sourceRoot.kind === "generator-matched") {
    exactKeys(sourceRoot, ["kind", "prior_hash"], "source");
    source = { kind: "generator-matched", prior_hash: hash(sourceRoot.prior_hash, "prior_hash") };
  } else if (sourceRoot.kind === "usage-table") {
    exactKeys(sourceRoot, ["kind", "slots", "usage_table_hash"], "source");
    source = { kind: "usage-table", usage_table_hash: hash(sourceRoot.usage_table_hash, "usage_table_hash"), slots: integer(sourceRoot.slots, 1, 8, "source.slots") };
  } else return fail("source.kind is not a known word");
  if (!Array.isArray(root.hp_key_handlers) || root.hp_key_handlers.length === 0) fail("hp_key_handlers must be a non-empty list");
  const handlers = (root.hp_key_handlers as unknown[]).map((name) => (typeof name === "string" && HANDLER.test(name) ? name : fail("hp_key_handlers holds a handler name")));
  if (handlers.some((name, index) => index > 0 && (handlers[index - 1] as string) >= name)) fail("hp_key_handlers must be sorted and unique");
  const out: LikelyConfig = {
    schema_version: 1,
    format_id: root.format_id as string,
    engine_data_hash: hash(root.engine_data_hash, "engine_data_hash"),
    source,
    per_foe: integer(root.per_foe, PER_FOE_MIN, PER_FOE_MAX, "per_foe"),
    hp_key_handlers: handlers,
  };
  if ("rank_candidates_max" in root) {
    out.rank_candidates_max = integer(root.rank_candidates_max, 1, RANK_MAX_LIMIT, "rank_candidates_max");
    const screen = record(root.screen, "screen");
    exactKeys(screen, ["effectiveness_ppm", "same_type_ppm", "spread_ppm", "variable_power"], "screen");
    const effectiveness = screen.effectiveness_ppm;
    if (!Array.isArray(effectiveness) || effectiveness.length !== DAMAGE_TAKEN_CODES) fail(`effectiveness_ppm must list one factor per damage_taken code (${DAMAGE_TAKEN_CODES})`);
    out.screen = {
      effectiveness_ppm: (effectiveness as unknown[]).map((entry) => integer(entry, 0, 16 * MILLION, "effectiveness_ppm")),
      same_type_ppm: integer(screen.same_type_ppm, MILLION, 16 * MILLION, "same_type_ppm"),
      spread_ppm: integer(screen.spread_ppm, 1, MILLION, "spread_ppm"),
      variable_power: integer(screen.variable_power, 1, 999, "variable_power"),
    };
  }
  return out;
}

/** SHA-256 of the record's JCS bytes: the `likely_threats_record_hash` of the feature set's identity. */
export const likelyConfigHash = (config: LikelyConfig): string => sha256Hex(jcs(config as unknown as Json));

type DexData = { [id: string]: unknown };
interface DexLike {
  data: { Moves?: DexData; Abilities?: DexData; Items?: DexData };
}

/** The handler names (of the given list) that some move, ability or item of the mod defines as a function, so a record naming none of the engine's is refused. */
export function handlersInUse(dex: DexLike, names: readonly string[]): Set<string> {
  const used = new Set<string>();
  for (const table of [dex.data.Moves, dex.data.Abilities, dex.data.Items]) {
    for (const entry of Object.values(table ?? {})) {
      for (const name of names) if (typeof (entry as Record<string, unknown>)[name] === "function") used.add(name);
    }
  }
  return used;
}

/** Problems of the record against the pinned engine's data and the golden vocabulary (none means ok). */
export function checkLikelyConfig(config: LikelyConfig, dex: DexLike, engineDataHash: string): string[] {
  const problems: string[] = [];
  if (config.engine_data_hash !== engineDataHash) problems.push("engine_data_hash is not the hash of the pinned engine-data table");
  const used = handlersInUse(dex, config.hp_key_handlers);
  for (const name of config.hp_key_handlers) if (!used.has(name)) problems.push(`hp_key_handlers names ${name}, which no move, ability or item of the engine defines`);
  return problems;
}

/** The hit-point-reading handler of an effect's engine data, if any (a `true` means the effect's cache key holds hit points). */
export const readsHitPoints = (data: Record<string, unknown> | undefined, handlers: readonly string[]): boolean => data !== undefined && handlers.some((name) => typeof data[name] === "function");

// ---------------------------------------------------------------- the source tables

/** One forme's (or species') move list: the mass of each move and the total the masses are shares of; an inclusion is `slots x mass / total` of a set. */
export interface MoveList {
  slots: number;
  total: number;
  masses: ReadonlyMap<string, number>;
}
/** The likelihood table of a record: lists by forme id (generator-matched) or species id (usage-table). */
export type MoveTable = ReadonlyMap<string, MoveList>;

const tables = new Map<string, { hash: string; table: MoveTable }>();

interface PriorFile {
  species: { id: string; formes: { id: string; moves: { slots: number; denominator: number; entries: { id: string; inclusion: number }[] } }[] }[];
}
interface UsageFile {
  species: { id: string; moves: { entries: { id: string; ppm: number }[] } }[];
}

/**
 * Read the committed source the record names and check its hash (SHA-256 of the file's JCS bytes), once per path. The file's content is
 * trusted for its shape beyond that: its own readers (usage prior, usage weight table) validate it where it is committed.
 */
export function loadMoveTable(source: LikelySource, formatId: string, dir: string = TABLE_DIR): MoveTable {
  const path = source.kind === "generator-matched" ? join(dir, "opponent-model", formatId, "generator-matched.uniform-v1.json") : join(dir, "teams", formatId, "usage", usageFile(dir, formatId, source.usage_table_hash));
  const wanted = source.kind === "generator-matched" ? source.prior_hash : source.usage_table_hash;
  const cached = tables.get(path);
  if (cached !== undefined && cached.hash === wanted) return cached.table;
  const value = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (sha256Hex(jcs(value as Json)) !== wanted) fail("the source file the likelihood record names has another hash");
  const table = new Map<string, MoveList>();
  if (source.kind === "generator-matched") {
    for (const species of (value as PriorFile).species) {
      for (const forme of species.formes) {
        const moves = forme.moves;
        table.set(forme.id, { slots: moves.slots, total: moves.slots * moves.denominator, masses: new Map(moves.entries.map((entry) => [entry.id, entry.inclusion])) });
      }
    }
  } else {
    for (const species of (value as UsageFile).species) table.set(species.id, { slots: source.slots, total: MILLION, masses: new Map(species.moves.entries.map((entry) => [entry.id, entry.ppm])) });
  }
  tables.set(path, { hash: wanted, table });
  return table;
}

/** The usage weight table file of a given hash among the committed ones of the format (the name carries the month and cutoff). */
function usageFile(dir: string, formatId: string, wanted: string): string {
  for (const name of readdirSync(join(dir, "teams", formatId, "usage")).filter((file) => file.endsWith(".json")).sort()) {
    if (sha256Hex(jcs(JSON.parse(readFileSync(join(dir, "teams", formatId, "usage", name), "utf8")) as Json)) === wanted) return name;
  }
  return fail("no committed usage weight table has the hash the likelihood record names");
}

/** A record as loaded for a run: the reviewed values and the table they name. */
export interface LikelySetup {
  config: LikelyConfig;
  table: MoveTable;
  /**
   * Test seam (the cache-equality test of E-22.7.6): `direct` prices every row with one uncached probe query at the real hit points and
   * derives its values the way `probe-damage/3` does; `drop_key` takes one part out of the cache's full key (a negative control: the
   * cache then returns the value of another state). Never set by a run.
   */
  direct?: boolean;
  drop_key?: "stages" | "hp";
}

/** The records of the format, by file name (sorted): every `probe-outcome-likely*.json` of `<dir>/<format>/`. */
export function likelyConfigFiles(formatId: string, dir: string = LIKELY_CONFIG_DIR): string[] {
  return readdirSync(join(dir, formatId)).filter((name) => /^probe-outcome-likely.*\.json$/.test(name)).sort();
}

/** Read one record file and its source table. Refuses a record of another format or engine-data table. */
export function loadLikelySetup(formatId: string, file: string, dir: string = LIKELY_CONFIG_DIR): LikelySetup {
  const config = readLikelyConfig(JSON.parse(readFileSync(join(dir, formatId, file), "utf8")) as unknown);
  if (config.format_id !== formatId) fail("the probe-outcome-likely record is for another format");
  if (formatId === GOLDEN_FORMAT && config.engine_data_hash !== goldenTeamVocabulary().engine_data_hash) fail("the probe-outcome-likely record is for another engine-data table");
  return { config, table: loadMoveTable(config.source, formatId) };
}
