// The reviewed world writers (ADR-026 Evidence 1, B2/N5, slice W part 2): the closed list of the state items
// a scenario world can hold, the class of each (E engine method, W reviewed writer, X resampled by the
// engine's own draw and counted, T turn-start reset, R refusal) and the ids admitted for the generic classes,
// as hashed reviewed data (`config/search/world-writers.v1.json`, the `WorldWriters` contract). The list of
// ITEMS is the closed part and lives here and in the schema's enum (a test holds them equal); what each item
// IS (its class, the engine ids a writer or a refusal row names) is data. No game rule is written here: every
// id the record names must resolve in the pinned engine (`readWorldWriters`).

import { readFileSync } from "node:fs";
import { ajvFor } from "@areneon/schema/ajv";
import worldWritersSchema from "@areneon/schema/contracts/v1/world-writers.schema.json" with { type: "json" };
import type { WorldWriters } from "@areneon/schema/types/v1/world-writers.ts";
import { type Json, jcs } from "./canonical-json.ts";
import type { Engine } from "./engine.ts";
import { sha256Hex } from "./rules.ts";

export type { WorldWriters };

export const STATE_CLASSES = ["E", "W", "X", "T", "R"] as const;
export type StateClass = (typeof STATE_CLASSES)[number];

/**
 * Every state item, with the classes this implementation can give it. A W item is written by the writer of
 * that name; give it class R and a world that holds the item is refused instead. `needs` says which extra
 * field the row carries: `condition` (the engine id of the condition the writer adds) or `ids` (the engine
 * ids, or protocol line kinds, whose presence is the item).
 */
export const STATE_ITEMS = {
  active_turns: { classes: ["W", "R"] },
  bench_status: { classes: ["W", "R"] },
  choice_lock: { classes: ["W", "R"], needs: "condition" },
  condition_duration: { classes: ["W", "R"] },
  fainted_last_turn: { classes: ["W", "R"] },
  item_removed: { classes: ["W", "R"] },
  last_move: { classes: ["W", "R"] },
  protect_counter: { classes: ["W", "R"], needs: "condition" },
  start_trigger_restore: { classes: ["W", "R"] },
  toxic_stage: { classes: ["W", "R"] },
  total_fainted: { classes: ["W", "R"] },
  turn_number: { classes: ["W", "R"] },
  volatile_duration: { classes: ["W", "R"] },
  bench_order: { classes: ["X", "R"] },
  confusion_duration: { classes: ["X", "R"], needs: "condition" },
  opponent_pp: { classes: ["X", "R"] },
  sleep_counter: { classes: ["X", "R"] },
  per_turn_fields: { classes: ["T"] },
  ability_changed: { classes: ["R"] },
  illusion_transform: { classes: ["R"], needs: "ids" },
  item_replaced: { classes: ["R"] },
  locked_moves: { classes: ["R"], needs: "ids" },
  slot_conditions: { classes: ["R"], needs: "ids" },
  substitute: { classes: ["R"], needs: "ids" },
  trapping: { classes: ["R"], needs: "ids" },
  truant_slow_start: { classes: ["R"], needs: "ids" },
  unadmitted_id: { classes: ["R"] },
  unavailable_mechanics: { classes: ["R"], needs: "lines" },
} as const satisfies Record<string, { classes: readonly StateClass[]; needs?: "condition" | "ids" | "lines" }>;
export type StateItemId = keyof typeof STATE_ITEMS;
export const STATE_ITEM_IDS = Object.keys(STATE_ITEMS).sort() as StateItemId[];

export const ADMITTED_KINDS = ["volatile", "start_trigger"] as const;
export type AdmittedKind = (typeof ADMITTED_KINDS)[number];
/** The classes each kind of admitted id can have. */
export const ADMITTED_CLASSES: Record<AdmittedKind, readonly StateClass[]> = {
  volatile: ["E", "W", "X"],
  start_trigger: ["W"],
};

export class WorldWritersError extends Error {
  override name = "WorldWritersError";
}

const validWriters = ajvFor("contract").compile(worldWritersSchema);

/** The engine ids the record's names are checked against: a condition, a move, an ability or an item. */
interface DexLookup {
  conditions: { get(id: string): { exists: boolean } };
  moves: { get(id: string): { exists: boolean } };
  abilities: { get(id: string): { exists: boolean } };
  items: { get(id: string): { exists: boolean } };
}

const LINE_KIND = /^-[a-z]+$/;

/**
 * Read a record: the schema (closed items, classes and kinds), then what it cannot say. Every state item is
 * classified exactly once, in ascending order, with a class this implementation gives it; a `condition` or
 * `ids` field exists exactly where the item takes it; admitted ids are sorted and unique by kind then id; and
 * every named id resolves in the pinned engine's data for the format's mod.
 */
export function readWorldWriters(value: unknown, engine: Engine): WorldWriters {
  if (!validWriters(value)) throw new WorldWritersError(`invalid WorldWriters: ${JSON.stringify(validWriters.errors)}`);
  const record = structuredClone(value) as unknown as WorldWriters;
  const format = engine.sim.Dex.formats.get(record.format_id);
  if (!format.exists) throw new WorldWritersError(`format ${record.format_id} is not an engine format`);
  const dex = engine.sim.Dex.mod(format.mod) as unknown as DexLookup;
  const known = (id: string, tables: readonly (keyof DexLookup)[]): boolean => tables.some((table) => dex[table].get(id).exists);

  if (record.items.length !== STATE_ITEM_IDS.length) throw new WorldWritersError("items: every state item is classified exactly once");
  record.items.forEach((item, index) => {
    if (item.id !== STATE_ITEM_IDS[index]) throw new WorldWritersError(`items: ${item.id} is repeated, out of order or missing a neighbour`);
    const spec = STATE_ITEMS[item.id] as { classes: readonly StateClass[]; needs?: string };
    if (!spec.classes.includes(item.class)) throw new WorldWritersError(`items: ${item.id} cannot have class ${item.class}`);
    const needs = spec.needs ?? "";
    if ((item.condition !== undefined) !== (needs === "condition")) throw new WorldWritersError(`items: ${item.id} ${needs === "condition" ? "needs" : "cannot have"} a condition`);
    if ((item.ids !== undefined) !== (needs === "ids" || needs === "lines")) throw new WorldWritersError(`items: ${item.id} ${needs === "ids" || needs === "lines" ? "needs" : "cannot have"} ids`);
    if (item.condition !== undefined && !known(item.condition, ["conditions"])) throw new WorldWritersError(`items: ${item.id}: ${item.condition} is not an engine condition`);
    item.ids?.forEach((id, k) => {
      if (k > 0 && !((item.ids as string[])[k - 1] as string < id)) throw new WorldWritersError(`items: ${item.id}: ${id} is repeated or out of order`);
      if (needs === "lines") {
        if (!LINE_KIND.test(id)) throw new WorldWritersError(`items: ${item.id}: ${id} is not a protocol line kind`);
      } else if (!known(id, ["conditions", "moves", "abilities", "items"])) throw new WorldWritersError(`items: ${item.id}: ${id} is not an engine condition, move, ability or item`);
    });
  });
  const seen = new Map<string, string>();
  record.items.forEach((item) => {
    for (const id of item.ids ?? []) {
      if (needs(item.id) === "ids") {
        const other = seen.get(id);
        if (other !== undefined) throw new WorldWritersError(`items: ${id} is listed under ${other} and ${item.id}`);
        seen.set(id, item.id);
      }
    }
  });
  record.admitted.forEach((entry, index) => {
    const before = record.admitted[index - 1];
    if (before !== undefined && !(before.kind < entry.kind || (before.kind === entry.kind && before.id < entry.id))) throw new WorldWritersError(`admitted: ${entry.kind} ${entry.id} is repeated or out of order`);
    if (!ADMITTED_CLASSES[entry.kind].includes(entry.class)) throw new WorldWritersError(`admitted: ${entry.kind} ${entry.id} cannot have class ${entry.class}`);
    const tables: readonly (keyof DexLookup)[] = entry.kind === "volatile" ? ["conditions"] : ["abilities", "items"];
    if (!known(entry.id, tables)) throw new WorldWritersError(`admitted: ${entry.kind} ${entry.id} is not in the engine's data`);
    if (entry.kind === "volatile" && seen.has(entry.id)) throw new WorldWritersError(`admitted: ${entry.id} is also a refusal row of ${seen.get(entry.id)}`);
  });
  const confusion = record.items.find((item) => item.id === "confusion_duration");
  if (record.admitted.some((entry) => entry.kind === "volatile" && entry.id === confusion?.condition)) throw new WorldWritersError("admitted: the confusion condition is the item confusion_duration's");
  return record;
}

const needs = (id: StateItemId): string => (STATE_ITEMS[id] as { needs?: string }).needs ?? "";

/** The hash the search identity names the writers by: SHA-256 of the record's JCS bytes. */
export const worldWritersHash = (record: WorldWriters): string => sha256Hex(jcs(record as unknown as Json));

/** The committed record for a format (`config/search/world-writers.v1.json`), read through the reader. */
export function loadWorldWriters(engine: Engine, formatId: string, path: URL = new URL("../../../config/search/world-writers.v1.json", import.meta.url)): WorldWriters {
  const record = readWorldWriters(JSON.parse(readFileSync(path, "utf8")), engine);
  if (record.format_id !== formatId) throw new WorldWritersError(`the record is for ${record.format_id}, not ${formatId}`);
  return record;
}
