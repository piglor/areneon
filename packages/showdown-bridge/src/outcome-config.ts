// The reviewed record of the feature set `probe-outcome/1` (ADR-022 erratum E-22.5.3): config/features/<format>/probe-outcome.v1.json.
// Integers and closed words only; it chooses among values the engine allows and holds no game rule. A change of any value is a new
// file and a new `outcome_config_hash` (so a new feature set hash), never an edit.
//
// - `assumed_branch`: how a foe that has revealed no move gets its one assumed opposing-action branch. The only rule is
//   `assumed-set-filler`: the move the assumed set (ADR-018) holds in place of an unrevealed one (its filler), else its first move.
// - `volatiles`: whether the volatiles of the facts are declared through the engine's own `addVolatile` (`declared`, committed) or left out
//   (`omitted`, a test seam: a known fact left out would be a confident default). A volatile the engine refuses in the scenario (Encore needs a
//   last move) makes the answers that hold it `history_unknown`.
// - `move_path_conditions`: the conditions the engine's move path reads (an `onBeforeMove`, `onModifyMove`, `onTryMove`, `onOverrideAction` or
//   `onLockMove` handler on a volatile; `check-outcome-config.ts` holds the list equal to the pinned engine's), sorted. When the channel cannot
//   settle a Pokemon's volatiles (an Illusion stint, a refused line) any of them may apply: if there are at most `unknown_volatile_limit` they are
//   the second end (none against all started), else every answer of a query that holds such a Pokemon is `history_unknown`.
// - `max_branches`: at most this many opposing-action branches per own move part (the `switch` branch and the revealed moves of
//   the targeted foe, cut in move-id order).
// - `unknown_count_end`: the upper end a history count the channel cannot settle is asked at (the two-ends rule asks the lower end 0 and this one).
// - `unknown_item`, `unknown_move`: engine ids standing for "some item" and "some move" at the second end of an unknown last item or last move
//   (the first end is none): they only make the two ends differ, so the answers say whether the fact matters.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import { conditionIds } from "./volatile-copy.ts";

export const OUTCOME_CONFIG_DIR = fileURLToPath(new URL("../../../config/features", import.meta.url));

export class OutcomeConfigError extends Error {
  override name = "OutcomeConfigError";
}

export const ASSUMED_BRANCH_RULES = ["assumed-set-filler"] as const;
export const VOLATILES_MODES = ["declared", "omitted"] as const;

export interface OutcomeConfig {
  schema_version: 1;
  format_id: string;
  assumed_branch: (typeof ASSUMED_BRANCH_RULES)[number];
  max_branches: number;
  volatiles: (typeof VOLATILES_MODES)[number];
  unknown_count_end: number;
  move_path_conditions: string[];
  unknown_volatile_limit: number;
  unknown_item: string;
  unknown_move: string;
}

const KEYS = ["assumed_branch", "format_id", "max_branches", "schema_version", "unknown_count_end", "unknown_item", "unknown_move", "unknown_volatile_limit", "volatiles"].concat(["move_path_conditions"]).sort();
const ID = /^[a-z0-9]+$/;
const integer = (value: unknown, min: number, max: number, what: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max) throw new OutcomeConfigError(`${what} must be an integer from ${min} to ${max}`);
  return value;
};
const ids = (value: unknown, what: string): string[] => {
  if (!Array.isArray(value) || value.length === 0) throw new OutcomeConfigError(`${what} must be a list of engine ids`);
  const list = value.map((entry) => id(entry, what));
  if (list.some((entry, index) => index > 0 && (list[index - 1] as string) >= entry)) throw new OutcomeConfigError(`${what} must be sorted and unique`);
  return list;
};
const closed = <T extends string>(value: unknown, words: readonly T[], what: string): T => {
  if (!(words as readonly unknown[]).includes(value)) throw new OutcomeConfigError(`${what} is not a known word`);
  return value as T;
};
const id = (value: unknown, what: string): string => {
  if (typeof value !== "string" || !ID.test(value)) throw new OutcomeConfigError(`${what} must be an engine id`);
  return value;
};

/** Read a record: closed keys, integers in range, closed words, engine-shaped ids. Whether an id exists in the engine is `checkOutcomeConfig`'s. */
export function readOutcomeConfig(value: unknown): OutcomeConfig {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new OutcomeConfigError("the probe-outcome record is not an object");
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  if (keys.length !== KEYS.length || keys.some((key, index) => key !== KEYS[index])) throw new OutcomeConfigError("the probe-outcome record must have exactly its keys");
  if (record.schema_version !== 1) throw new OutcomeConfigError("schema_version must be 1");
  if (typeof record.format_id !== "string" || record.format_id === "") throw new OutcomeConfigError("format_id must be a string");
  return {
    schema_version: 1,
    format_id: record.format_id,
    assumed_branch: closed(record.assumed_branch, ASSUMED_BRANCH_RULES, "assumed_branch"),
    max_branches: integer(record.max_branches, 2, 5, "max_branches"),
    volatiles: closed(record.volatiles, VOLATILES_MODES, "volatiles"),
    unknown_count_end: integer(record.unknown_count_end, 1, 9999, "unknown_count_end"),
    unknown_item: id(record.unknown_item, "unknown_item"),
    unknown_move: id(record.unknown_move, "unknown_move"),
    move_path_conditions: ids(record.move_path_conditions, "move_path_conditions"),
    unknown_volatile_limit: integer(record.unknown_volatile_limit, 0, 99, "unknown_volatile_limit"),
  };
}

interface ConditionLike {
  exists: boolean;
  effectType?: string;
  condition?: Record<string, unknown>;
  [handler: string]: unknown;
}

/** The handlers of a volatile that the engine's move path runs (`runMove`: OverrideAction, BeforeMove, the lock; the move step: ModifyMove, TryMove). */
const MOVE_PATH_HANDLERS = ["onBeforeMove", "onLockMove", "onModifyMove", "onOverrideAction", "onTryMove"];
const NOT_VOLATILES = new Set(["Status", "Weather", "Terrain", "Rule", "Format", "Ruleset"]);

/** The condition ids of the mod with a move-path handler (a condition of a move, ability or item is its `condition` object), sorted: derived from the engine's data, no id named. */
export function movePathConditions(dex: { data: Parameters<typeof conditionIds>[0]["data"]; conditions: { get(id: string): ConditionLike } }): string[] {
  return conditionIds(dex).filter((id) => {
    const found = dex.conditions.get(id);
    if (found.effectType !== undefined && NOT_VOLATILES.has(found.effectType)) return false;
    const body = (found.condition ?? found) as Record<string, unknown>;
    return MOVE_PATH_HANDLERS.some((handler) => typeof body[handler] === "function" || typeof found[handler] === "function");
  });
}

/** The ids the record names that the pinned engine's data must hold (a problem per missing one; none means ok). */
export function checkOutcomeConfig(config: OutcomeConfig, dex: { items: { get(id: string): { exists: boolean } }; moves: { get(id: string): { exists: boolean } }; conditions?: { get(id: string): ConditionLike }; data?: Parameters<typeof conditionIds>[0]["data"] }): string[] {
  const problems: string[] = [];
  if (!dex.items.get(config.unknown_item).exists) problems.push("unknown_item is not an engine item");
  if (!dex.moves.get(config.unknown_move).exists) problems.push("unknown_move is not an engine move");
  if (dex.data !== undefined && dex.conditions !== undefined) {
    const derived = movePathConditions(dex as Required<Pick<typeof dex, "data" | "conditions">>);
    for (const id of derived) if (!config.move_path_conditions.includes(id)) problems.push(`the engine's ${id} reads the move path and is not in move_path_conditions`);
    for (const id of config.move_path_conditions) if (!derived.includes(id)) problems.push(`move_path_conditions names ${id}, which the engine's move path does not read`);
  } else problems.push("no engine data to check move_path_conditions against");
  return problems;
}

/** SHA-256 of the record's JCS bytes: the `outcome_config_hash` of the feature set's identity. */
export const outcomeConfigHash = (config: OutcomeConfig): string => sha256Hex(jcs(config as unknown as Json));

/** Read `<dir>/<format_id>/probe-outcome.v1.json`. */
export function loadOutcomeConfig(formatId: string, dir: string = OUTCOME_CONFIG_DIR): OutcomeConfig {
  const config = readOutcomeConfig(JSON.parse(readFileSync(join(dir, formatId, "probe-outcome.v1.json"), "utf8")) as unknown);
  if (config.format_id !== formatId) throw new OutcomeConfigError(`probe-outcome.v1.json is the record of another format`);
  return config;
}
