// The battle-state tensor schema (ADR-030 step 5 groundwork, ADR-028 section 5): the reviewed, versioned list of
// the fields a batched battle-state tensor holds for one format (`config/engine-state/<format>/battle-state-spec.v1.json`,
// the `BattleStateSpec` contract). Each field names its dtype, shape, value range or engine-data vocabulary, its
// source (a path into the pinned engine's `Battle.toJSON()` or a named derivation) and its visibility class. This
// module reads the record and states the projection rules; it builds no engine and produces no tensor (that is the
// next step). No game rule is written here: vocabulary sizes come from the engine-data table, counts from the rule
// table and the team vocabulary (`bindingFor`), and both are checked, never assumed.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import specSchema from "@areneon/schema/contracts/v1/battle-state-spec.schema.json" with { type: "json" };
import type { BattleStateSpec, Field } from "@areneon/schema/types/v1/battle-state-spec.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { type Engine, STAT_IDS } from "./engine.ts";
import { GOLDEN_DIR } from "./golden.ts";
import { sha256Hex } from "./rules.ts";
import { STATE_ITEM_IDS } from "./world-writers.ts";

export type { BattleStateSpec, Field };

export class BattleStateSpecError extends Error {
  override name = "BattleStateSpecError";
}

/** The groups, in the record's order (every group exactly once). */
export const GROUPS = ["active", "battle", "field", "roster", "seat_active", "seat_field", "seat_roster", "seat_side", "side"] as const;
export type GroupId = (typeof GROUPS)[number];

/** The engine's stat-stage ids, the shape of `pokemon.boosts` (held equal to real states by the coverage test). */
export const BOOST_IDS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"] as const;

/** The levels of `Battle.toJSON()` whose keys the coverage test classifies. */
export const ENGINE_LEVELS = ["battle", "side", "pokemon", "field", "move_slot", "effect"] as const;
export type EngineLevel = (typeof ENGINE_LEVELS)[number];

const DTYPE_RANGE = { int8: [-128, 127], int16: [-32768, 32767], int32: [-2147483648, 2147483647] } as const;

/** The engine roots a group's fields may read (a roster field reads its Pokemon, an active field the slot's Pokemon or the side's slot). */
const ROOTS: Record<GroupId, readonly string[]> = {
  battle: ["battle"],
  side: ["side"],
  roster: ["pokemon"],
  active: ["active", "side"],
  field: ["field"],
  seat_active: [],
  seat_field: [],
  seat_roster: [],
  seat_side: [],
};

/** The lists of the engine-data version 2 table a vocabulary may name. */
export interface VocabularyTable {
  readonly [list: string]: unknown;
}

/** What a spec is checked against: the engine-data v2 table, the team vocabulary's hash and the counts of the engine and rule table. */
export interface BattleStateBinding {
  engine_data: VocabularyTable;
  vocabulary_hash: string;
  counts: Record<string, number>;
}

const validSpec = ajvFor("contract").compile(specSchema);

const isFlag = (field: Field): boolean => field.visibility === "seat" && field.value.kind === "range" && field.value.min === 0 && field.value.max === 1;

const ordered = <T>(rows: readonly T[], key: (row: T) => string, what: string): void => {
  rows.forEach((row, index) => {
    const before = rows[index - 1];
    if (before !== undefined && !(key(before) < key(row))) throw new BattleStateSpecError(`${what}: ${key(row)} is repeated or out of order`);
  });
};

/** Every id of the union of the named lists of the engine-data table (sorted, unique). */
export function vocabularyIds(table: VocabularyTable, lists: readonly string[]): string[] {
  const ids = new Set<string>();
  for (const name of lists) {
    const list = table[name];
    if (!Array.isArray(list)) throw new BattleStateSpecError(`the engine-data table has no list ${name}`);
    for (const entry of list as { id: string }[]) ids.add(entry.id);
  }
  return [...ids].sort();
}

/**
 * The (level, key) pairs of `Battle.toJSON()` an engine path reads: the key after the root is at the root's level
 * (an active Pokemon is a Pokemon); after `moveSlots` it is a move slot's; after an effect-state key (`statusState`,
 * a `volatiles` or condition map value) it is an effect state's. Deeper keys are leaves and are not classified.
 */
export function pathKeys(path: string): [EngineLevel, string][] {
  const segments = path.match(/^[a-z]+|\.[A-Za-z]+|\[\]|\{\}/g) as string[];
  const root = segments[0] as string;
  let level: EngineLevel | null = root === "active" ? "pokemon" : (root as EngineLevel);
  const out: [EngineLevel, string][] = [];
  for (const segment of segments.slice(1)) {
    if (segment === "[]" || segment === "{}") continue;
    if (level === null) break;
    const key = segment.slice(1);
    out.push([level, key]);
    if (key === "moveSlots" || key === "baseMoveSlots") level = "move_slot";
    else if (key.endsWith("State") || ["sideConditions", "volatiles", "pseudoWeather", "slotConditions"].includes(key)) level = "effect";
    else level = null;
  }
  return out;
}

/** The engine keys the spec's fields read, as `level.key`. */
export function coveredKeys(spec: BattleStateSpec): Set<string> {
  const out = new Set<string>();
  for (const field of spec.fields) if (field.source.kind === "engine") for (const [level, key] of pathKeys(field.source.path)) out.add(`${level}.${key}`);
  return out;
}

/** Who reads a field: whether its side is the reader's own, and whether a seat field's seat index is the reader's. */
export interface Viewer {
  owner: boolean;
  seat: boolean;
}

/**
 * What a seat sees of a field (the projection rules, by visibility class alone). `owner` is whether the field's
 * side is the reader's own (a field without a side is nobody's: it is read as another seat's); `seat` is whether a
 * seat field is read at the reader's own seat index. `seat` fields are exact only there; `hidden` is never
 * projected; `own-exact` is exact for the owner and its `projection` otherwise (`none`: nothing); `revealed-only`
 * is the value where the reader's own reveal mask is set and the unknown marker elsewhere (`masked`). A group's
 * `gate` (a roster entry the reader has not seen) masks every field of the entry for another seat in addition.
 */
export function seatView(field: Field, viewer: Viewer): "exact" | "percent" | "masked" | "none" {
  switch (field.visibility) {
    case "seat":
      return viewer.seat ? "exact" : "none";
    case "hidden":
      return "none";
    case "public":
      return "exact";
    case "revealed-only":
      return "masked";
    case "own-exact":
      return viewer.owner ? "exact" : field.projection === "hp_percent" ? "percent" : "none";
  }
}

/** Read a record: the schema, then what the profile cannot say; with `binding`, also the engine-data, vocabulary and count bindings. */
export function readBattleStateSpec(value: unknown, binding?: BattleStateBinding): BattleStateSpec {
  if (!validSpec(value)) throw new BattleStateSpecError(`invalid BattleStateSpec: ${JSON.stringify(validSpec.errors)}`);
  const spec = structuredClone(value) as unknown as BattleStateSpec;

  ordered(spec.dims, (d) => d.symbol, "dims");
  const dims = new Map(spec.dims.map((d) => [d.symbol, d]));
  const used = new Set<string>();
  const shape = (symbols: readonly string[], what: string): void => {
    for (const s of symbols) {
      if (!dims.has(s)) throw new BattleStateSpecError(`${what}: ${s} is not a dimension`);
      used.add(s);
    }
  };
  for (const dim of spec.dims) if (dim.kind === "vocabulary") ordered(dim.lists, (l) => l, `dims: ${dim.symbol}`);
  if (spec.groups.map((g) => g.id).join() !== GROUPS.join()) throw new BattleStateSpecError("groups: every group exactly once, in order");
  for (const group of spec.groups) shape(group.shape, `groups: ${group.id}`);

  ordered(spec.fields, (f) => f.id, "fields");
  const fields = new Map(spec.fields.map((f) => [f.id, f]));
  for (const field of spec.fields) {
    const where = `fields: ${field.id}`;
    if (field.id.split(".")[0] !== field.group) throw new BattleStateSpecError(`${where}: the id does not start with its group`);
    shape(field.extra, where);
    const [lo, hi] = DTYPE_RANGE[field.dtype];
    if (field.value.kind === "range") {
      if (field.value.min > field.value.max) throw new BattleStateSpecError(`${where}: min exceeds max`);
      if (field.value.min < lo || field.value.max > hi) throw new BattleStateSpecError(`${where}: the range does not fit ${field.dtype}`);
    } else ordered(field.value.lists, (l) => l, `${where}: lists`);
    const seat = field.group.startsWith("seat_");
    if (field.source.kind === "engine") {
      if (seat) throw new BattleStateSpecError(`${where}: a seat field is a derivation; the engine's state holds nothing a seat was shown`);
      const root = field.source.path.split(/[.[{]/)[0] as string;
      if (!ROOTS[field.group as GroupId].includes(root)) throw new BattleStateSpecError(`${where}: a ${field.group} field cannot read the engine root ${root}`);
    }
    if (seat !== (field.visibility === "seat")) throw new BattleStateSpecError(`${where}: the class seat is exactly for a seat field (its own seat's derivation)`);
    if (field.masks !== undefined && !seat) throw new BattleStateSpecError(`${where}: only a seat field masks another`);
    if ((field.projection !== undefined) !== (field.visibility === "own-exact")) throw new BattleStateSpecError(`${where}: a projection exactly for an own-exact field`);
    if ((field.reveal !== undefined) !== (field.visibility === "revealed-only")) throw new BattleStateSpecError(`${where}: a reveal mask exactly for a revealed-only field`);
  }
  for (const field of spec.fields) {
    if (field.reveal === undefined) continue;
    const mask = fields.get(field.reveal);
    if (mask === undefined) throw new BattleStateSpecError(`fields: ${field.id}: its mask ${field.reveal} is not a field`);
    if (!isFlag(mask) || mask.group !== `seat_${field.group}` || mask.extra.join() !== field.extra.join()) {
      throw new BattleStateSpecError(`fields: ${field.id}: its mask must be a 0/1 seat field of the same entity and extra dimensions`);
    }
    if (mask.masks !== field.id) throw new BattleStateSpecError(`fields: ${field.id}: its mask ${mask.id} masks ${mask.masks ?? "nothing"}, not this field`);
  }
  for (const group of spec.groups) {
    if (group.gate === undefined) continue;
    const gate = fields.get(group.gate);
    if (group.id.startsWith("seat_") || gate === undefined || !isFlag(gate) || gate.group !== `seat_${group.id}` || gate.extra.length > 0) {
      throw new BattleStateSpecError(`groups: ${group.id}: its gate must be a 0/1 seat field of the group's entity`);
    }
  }
  for (const dim of spec.dims) if (!used.has(dim.symbol)) throw new BattleStateSpecError(`dims: ${dim.symbol} is not used`);

  ordered(spec.omitted, (o) => `${o.level}.${o.key}`, "omitted");
  const covered = coveredKeys(spec);
  for (const o of spec.omitted) if (covered.has(`${o.level}.${o.key}`)) throw new BattleStateSpecError(`omitted: ${o.level}.${o.key} is read by a field`);

  // A mapping is read by one seat at its own seat index: the own half as the owner, the rest as another seat.
  const refs = (list: readonly string[], what: string, opponent: boolean): void => {
    ordered(list, (r) => r, what);
    for (const ref of list) {
      const field = fields.get(ref);
      if (field === undefined) throw new BattleStateSpecError(`${what}: ${ref} is not a field`);
      if (field.visibility === "hidden") throw new BattleStateSpecError(`${what}: ${ref} is hidden: no seat may read it`);
      if (seatView(field, { owner: !opponent, seat: true }) === "none") throw new BattleStateSpecError(`${what}: ${ref} is not projectable to another seat`);
    }
  };
  ordered(spec.encoder, (c) => `${c.group}:${c.column}`, "encoder");
  for (const c of spec.encoder) {
    const what = `encoder: ${c.group} ${c.column}`;
    refs(c.own, `${what} own`, false);
    refs(c.opponent, `${what} opponent`, true);
    refs(c.shared, `${what} shared`, true);
  }
  ordered(spec.reducer, (r) => r.key, "reducer");
  for (const r of spec.reducer) {
    refs(r.own, `reducer: ${r.key} own`, false);
    refs(r.opponent, `reducer: ${r.key} opponent`, true);
    refs(r.shared, `reducer: ${r.key} shared`, true);
  }
  ordered(spec.out_of_scope, (o) => o.option, "out_of_scope");

  if (spec.world_items.map((w) => w.id).join() !== STATE_ITEM_IDS.join()) throw new BattleStateSpecError("world_items: every state item of the world writers exactly once, in order");
  for (const item of spec.world_items) {
    if (item.kind === "held") {
      ordered(item.fields, (f) => f, `world_items: ${item.id}`);
      for (const ref of item.fields) if (!fields.has(ref)) throw new BattleStateSpecError(`world_items: ${item.id}: ${ref} is not a field`);
    }
  }

  if (binding !== undefined) checkBinding(spec, binding);
  return spec;
}

function checkBinding(spec: BattleStateSpec, binding: BattleStateBinding): void {
  if (sha256Hex(jcs(binding.engine_data as unknown as Json)) !== spec.engine_data_hash) throw new BattleStateSpecError("engine_data_hash is not the table's");
  if (binding.vocabulary_hash !== spec.vocabulary_hash) throw new BattleStateSpecError("vocabulary_hash is not the team vocabulary's");
  for (const dim of spec.dims) {
    if (dim.kind === "count") {
      if (binding.counts[dim.source] !== dim.size) throw new BattleStateSpecError(`dims: ${dim.symbol} has size ${dim.size}, the ${dim.source} is ${binding.counts[dim.source]}`);
    } else if (vocabularyIds(binding.engine_data, dim.lists).length === 0) throw new BattleStateSpecError(`dims: ${dim.symbol} has no id`);
  }
  for (const field of spec.fields) {
    if (field.value.kind !== "vocabulary") continue;
    const size = 2 + vocabularyIds(binding.engine_data, field.value.lists).length;
    if (size - 1 > DTYPE_RANGE[field.dtype][1]) throw new BattleStateSpecError(`fields: ${field.id}: ${size} indices do not fit ${field.dtype}`);
  }
}

/** The hash a later engine build and its tensors name the spec by: SHA-256 of the record's JCS bytes. */
export const battleStateSpecHash = (spec: BattleStateSpec): string => sha256Hex(jcs(spec as unknown as Json));

/** The binding of the pinned engine: the golden engine-data v2 table and team vocabulary, and the counts read from the engine. */
export function bindingFor(engine: Engine, formatId: string, dir: string = GOLDEN_DIR): BattleStateBinding {
  const format = engine.sim.Dex.formats.get(formatId);
  const rules = engine.sim.Dex.formats.getRuleTable(format);
  if (rules.maxMoveCount === undefined) throw new BattleStateSpecError(`the rule table of ${formatId} has no move count`);
  const battle = new engine.sim.Battle({ formatid: formatId, seed: `sodium,${"0".repeat(31)}1` }) as unknown as { activePerHalf: number; sides: unknown[]; prng: { getSeed(): string } };
  const seed = battle.prng.getSeed().split(",")[1] as string;
  const vocabulary = JSON.parse(readFileSync(join(dir, "team-vocabulary.summary.json"), "utf8")) as { vocabulary_hash: string };
  return {
    engine_data: JSON.parse(readFileSync(join(dir, "engine-data-2.json"), "utf8")) as VocabularyTable,
    vocabulary_hash: vocabulary.vocabulary_hash,
    counts: {
      players: battle.sides.length,
      max_team_size: rules.maxTeamSize,
      picked_team_size: rules.pickedTeamSize ?? rules.maxTeamSize,
      active_per_side: battle.activePerHalf,
      max_move_count: rules.maxMoveCount,
      stored_stat_ids: STAT_IDS.length - 1,
      boost_ids: BOOST_IDS.length,
      prng_words: seed.length / 8,
    },
  };
}

/** The committed record for a format, read through the reader against the pinned engine. */
export function loadBattleStateSpec(engine: Engine, formatId: string, path: URL = new URL(`../../../config/engine-state/${formatId}/battle-state-spec.v1.json`, import.meta.url)): BattleStateSpec {
  const spec = readBattleStateSpec(JSON.parse(readFileSync(path, "utf8")), bindingFor(engine, formatId));
  if (spec.format_id !== formatId) throw new BattleStateSpecError(`the record is for ${spec.format_id}, not ${formatId}`);
  return spec;
}

/**
 * The parents an engine path's last segment is read from, and whether that segment exists in each: the drift test's
 * check that a field's source resolves on real states. `active` walks each side's Pokemon in play (`isActive`; the
 * engine serialises `side.active` as references).
 */
export function resolvePath(state: Record<string, unknown>, path: string): { parents: number; found: number } {
  const segments = path.match(/^[a-z]+|\.[A-Za-z]+|\[\]|\{\}/g) as string[];
  const root = segments[0] as string;
  const sides = (state.sides ?? []) as Record<string, unknown>[];
  let values: unknown[] =
    root === "battle"
      ? [state]
      : root === "field"
        ? [state.field]
        : root === "side"
          ? sides
          : root === "pokemon"
            ? sides.flatMap((s) => s.pokemon as unknown[])
            : sides.flatMap((s) => (s.pokemon as Record<string, unknown>[]).filter((m) => m.isActive === true));
  const rest = segments.slice(1);
  rest.forEach((segment, index) => {
    if (index === rest.length - 1) return;
    values = values.flatMap((v) => step(v, segment));
  });
  const last = rest.at(-1) as string;
  const found = values.filter((v) => (last === "[]" ? Array.isArray(v) : last === "{}" ? isObject(v) && !Array.isArray(v) : isObject(v) && last.slice(1) in v)).length;
  return { parents: values.length, found };
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;

function step(value: unknown, segment: string): unknown[] {
  if (segment === "[]") return Array.isArray(value) ? value : [];
  if (segment === "{}") return isObject(value) && !Array.isArray(value) ? Object.values(value) : [];
  return isObject(value) && segment.slice(1) in value ? [value[segment.slice(1)]] : [];
}
