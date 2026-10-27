// The move classes of the move-use diagnostic (#1164), read from the pinned
// engine's move data for one format's mod. A class is a rule over the engine's
// own fields, never a list of moves written here: the Status category, a boosts
// table that lowers a foe target's stage (or a secondary with chance 100 that
// does), one that raises the user's or an ally's stage, a side condition, a
// weather, terrain or pseudo-weather, and the engine's stallingMove (the
// Protect family). The table names the engine-data table by hash and its ids
// resolve in it, but it is a table of its own: adding fields to the engine-data
// export would be a schema_version 2 and move the engine-data hash that every
// checkpoint and encoder version is bound to, for a read-only diagnostic.
//
// Limits, stated in the contract: an effect a move produces in a callback
// (onHit and the like: Parting Shot, Belly Drum, Haze) and a secondary with a
// chance below 100 are not read. The target kinds are grouped by who the move
// names (the user and allies, foes, no Pokemon); a target kind the engine adds
// in a later pin that is in no group refuses the export. The table is derived
// from the pinned Showdown data files (MIT, not CC0; NOTICE-engine-data.txt
// beside it keeps the upstream notice).

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import moveClassSchema from "@areneon/schema/contracts/v1/move-class-table.schema.json" with { type: "json" };
import type { MoveClassTable } from "@areneon/schema/types/v1/move-class-table.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type EngineData, engineDataHash, exportEngineData } from "./engine-data.ts";
import { type Engine, EngineError } from "./engine.ts";

export type { MoveClassTable };

export class MoveClassError extends Error {
  override name = "MoveClassError";
}

/** The golden export, beside the engine-data table. */
export const MOVE_CLASSES_FILE = "move-classes.json";
/** The table's hash and counts, small enough to review in a diff. */
export const MOVE_CLASSES_SUMMARY = "move-classes.summary.json";

/** The class names, in the table's order. */
export const MOVE_CLASS_NAMES = ["status", "lowers_target_stat", "raises_own_stat", "side_condition", "field_condition", "protection"] as const;

/** Who a move's target kind names: the user or an ally, a foe, or no Pokemon (a side, the field, a script). */
const TARGET_KINDS: Record<string, "friendly" | "foe" | "none"> = {
  adjacentAlly: "friendly",
  adjacentAllyOrSelf: "friendly",
  allies: "friendly",
  allyTeam: "friendly",
  self: "friendly",
  allAdjacent: "foe",
  allAdjacentFoes: "foe",
  any: "foe",
  normal: "foe",
  randomNormal: "foe",
  all: "none",
  allySide: "none",
  foeSide: "none",
  scripted: "none",
};

const STAGES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

const validTable = ajvFor("contract").compile(moveClassSchema);

type Raw = Record<string, unknown>;

const bad = (message: string): never => {
  throw new MoveClassError(message);
};

function rec(value: unknown, what: string): Raw {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad(`${what}: not an object`);
  return value as Raw;
}

/** The engine's `boosts` table of an effect: stage names to integers, or none. */
function boosts(value: unknown, what: string): { lowers: boolean; raises: boolean } {
  if (value === undefined) return { lowers: false, raises: false };
  const table = rec(value, what);
  let lowers = false;
  let raises = false;
  for (const [stage, amount] of Object.entries(table)) {
    if (!STAGES.includes(stage) || typeof amount !== "number" || !Number.isSafeInteger(amount)) bad(`${what}: ${stage} is not a stage change`);
    if ((amount as number) < 0) lowers = true;
    if ((amount as number) > 0) raises = true;
  }
  return { lowers, raises };
}

/** A string-or-absent field (an empty string or `undefined` is none). */
function named(value: unknown, what: string): boolean {
  if (value === undefined || value === "") return false;
  if (typeof value !== "string") bad(`${what}: not a string`);
  return true;
}

const unique = (ids: string[]): string[] => [...new Set(ids)].sort();

/**
 * Export the move classes of `formatId`'s mod from the pinned engine, for the
 * moves of the engine-data table (`table`, exported when not given).
 */
export function exportMoveClasses(engine: Engine, formatId: string, table: EngineData = exportEngineData(engine, formatId)): MoveClassTable {
  const sim = engine.sim as unknown as { Dex: { formats: { get(id: string): { exists: boolean; mod: string } }; mod(name: string): { moves: { get(id: string): unknown } } } };
  const format = sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  const dex = sim.Dex.mod(format.mod);
  for (const target of table.move_targets) if (!(target in TARGET_KINDS)) bad(`move target ${target}: not in a group of target kinds`);

  const classes: Record<(typeof MOVE_CLASS_NAMES)[number], string[]> = {
    status: [],
    lowers_target_stat: [],
    raises_own_stat: [],
    side_condition: [],
    field_condition: [],
    protection: [],
  };
  for (const entry of table.moves) {
    const move = rec(dex.moves.get(entry.id), `move ${entry.id}`);
    if (move.exists !== true) bad(`move ${entry.id}: the engine has no such move`);
    const what = `move ${entry.id}`;
    const kind = TARGET_KINDS[entry.target] as "friendly" | "foe" | "none";
    if (entry.category === "Status") classes.status.push(entry.id);

    const own = boosts(move.boosts, `${what} boosts`);
    const self = boosts(rec(move.self ?? {}, `${what} self`).boosts, `${what} self boosts`);
    const selfBoost = boosts(rec(move.selfBoost ?? {}, `${what} selfBoost`).boosts, `${what} selfBoost boosts`);
    let lowersTarget = kind === "foe" && own.lowers;
    let raisesOwn = (kind === "friendly" && own.raises) || self.raises || selfBoost.raises;
    const secondaries = move.secondaries === undefined ? [] : move.secondaries;
    if (!Array.isArray(secondaries)) bad(`${what} secondaries: not an array`);
    for (const raw of secondaries as unknown[]) {
      const secondary = rec(raw, `${what} secondary`);
      if (secondary.chance !== 100) continue; // a chance below 100 is not read
      if (boosts(secondary.boosts, `${what} secondary boosts`).lowers) lowersTarget = true;
      if (boosts(rec(secondary.self ?? {}, `${what} secondary self`).boosts, `${what} secondary self boosts`).raises) raisesOwn = true;
    }
    if (lowersTarget) classes.lowers_target_stat.push(entry.id);
    if (raisesOwn) classes.raises_own_stat.push(entry.id);
    if (named(move.sideCondition, `${what} sideCondition`)) classes.side_condition.push(entry.id);
    if (named(move.pseudoWeather, `${what} pseudoWeather`) || named(move.weather, `${what} weather`) || named(move.terrain, `${what} terrain`)) classes.field_condition.push(entry.id);
    if (move.stallingMove !== undefined && typeof move.stallingMove !== "boolean") bad(`${what} stallingMove: not a boolean`);
    if (move.stallingMove === true) classes.protection.push(entry.id);
  }
  return readMoveClassTable(
    {
      schema_version: 1,
      showdown_commit: engine.commit,
      format_id: formatId,
      mod: format.mod,
      engine_data_hash: engineDataHash(table),
      classes: Object.fromEntries(MOVE_CLASS_NAMES.map((name) => [name, unique(classes[name])])),
    },
    table,
  );
}

/**
 * Parse a stored table: the schema, then what the profile cannot say (sorted
 * unique ids) and, given the engine-data table, its hash, that every id is one
 * of its moves and that the status list is exactly its Status moves.
 */
export function readMoveClassTable(value: unknown, table?: EngineData): MoveClassTable {
  if (!validTable(value)) return bad(`invalid move class table: ${JSON.stringify(validTable.errors)}`);
  const found = value as unknown as MoveClassTable;
  for (const name of MOVE_CLASS_NAMES) {
    found.classes[name].forEach((id, k) => {
      if (k > 0 && !((found.classes[name][k - 1] as string) < id)) bad(`${name}: ${id} is repeated or out of order`);
    });
  }
  if (table !== undefined) {
    if (found.engine_data_hash !== engineDataHash(table)) bad("the table was made against another engine-data table");
    if (found.format_id !== table.format_id || found.mod !== table.mod || found.showdown_commit !== table.showdown_commit) bad("the table is for another format, mod or pin than the engine-data table");
    const ids = new Set(table.moves.map((m) => m.id));
    for (const name of MOVE_CLASS_NAMES) for (const id of found.classes[name]) if (!ids.has(id)) bad(`${name}: ${id} is not a move of the engine-data table`);
    const statusMoves = table.moves.filter((m) => m.category === "Status").map((m) => m.id);
    if (JSON.stringify(statusMoves) !== JSON.stringify(found.classes.status)) bad("the status class is not exactly the Status moves of the engine-data table");
  }
  return found;
}

/** SHA-256 of the table's JCS bytes. */
export function moveClassTableHash(table: MoveClassTable): string {
  return createHash("sha256").update(jcs(table as unknown as Json), "utf8").digest("hex");
}

/** The reviewable summary committed beside the golden table. */
export function moveClassSummary(table: MoveClassTable): { move_class_table_hash: string; engine_data_hash: string; showdown_commit: string; counts: Record<string, number> } {
  const counts: Record<string, number> = {};
  for (const name of MOVE_CLASS_NAMES) counts[name] = table.classes[name].length;
  return { move_class_table_hash: moveClassTableHash(table), engine_data_hash: table.engine_data_hash, showdown_commit: table.showdown_commit, counts };
}
