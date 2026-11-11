// The reviewed record of the feature set `probe-outcome/1` (ADR-022 erratum E-22.5.3): config/features/<format>/probe-outcome.v1.json.
// Integers and closed words only; it chooses among values the engine allows and holds no game rule. A change of any value is a new
// file and a new `outcome_config_hash` (so a new feature set hash), never an edit.
//
// - `assumed_branch`: how a foe that has revealed no move gets its one assumed opposing-action branch. The only rule is
//   `assumed-set-filler`: the move the assumed set (ADR-018) holds in place of an unrevealed one (its filler), else its first move.
// - `volatiles`: whether the volatiles of the facts are declared through the engine's own `addVolatile` (`declared`) or left out (`omitted`).
//   Many start handlers refuse in a scenario that has none of the battle that led to them (Encore needs a last move, a counter needs its
//   source), and the engine's refusal makes the whole query unsupported; the committed record omits them (a measured finding of slice H3,
//   see CLAUDE.md), so a volatile's effect is the engine's default there. A way to write a volatile without its start handler is a later id.
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
  unknown_item: string;
  unknown_move: string;
}

const KEYS = ["assumed_branch", "format_id", "max_branches", "schema_version", "unknown_count_end", "unknown_item", "unknown_move", "volatiles"];
const ID = /^[a-z0-9]+$/;
const integer = (value: unknown, min: number, max: number, what: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max) throw new OutcomeConfigError(`${what} must be an integer from ${min} to ${max}`);
  return value;
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
  };
}

/** The ids the record names that the pinned engine's data must hold (a problem per missing one; none means ok). */
export function checkOutcomeConfig(config: OutcomeConfig, dex: { items: { get(id: string): { exists: boolean } }; moves: { get(id: string): { exists: boolean } } }): string[] {
  const problems: string[] = [];
  if (!dex.items.get(config.unknown_item).exists) problems.push("unknown_item is not an engine item");
  if (!dex.moves.get(config.unknown_move).exists) problems.push("unknown_move is not an engine move");
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
