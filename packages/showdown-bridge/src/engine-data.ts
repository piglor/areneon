// The game vocabularies and their numeric attributes, exported from the
// pinned engine's data (ADR-010 section 3). Nothing here is a game rule:
// every value is read from the engine's own tables for the format's mod, in a
// canonical order (each list sorted by id), and refuses any shape it does not
// know. Its JCS hash goes into the encoder version, so a pin bump that changes
// any engine data changes the hash and never silently remaps a vocabulary.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import engineDataSchema from "@areneon/schema/contracts/v1/engine-data.schema.json" with { type: "json" };
import type { EngineData } from "@areneon/schema/types/v1/engine-data.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type Engine, EngineError } from "./engine.ts";

export type { EngineData };

export class EngineDataError extends Error {
  override name = "EngineDataError";
}

/** The golden export, beside the other pinned-engine fixtures. */
export const ENGINE_DATA_FILE = "engine-data.json";
/** The table's hash and entry counts, small enough to review in a diff. */
export const ENGINE_DATA_SUMMARY = "engine-data.summary.json";

const validTable = ajvFor("contract").compile(engineDataSchema);

type Raw = Record<string, unknown>;

/** The slice of a Showdown `ModdedDex` the export reads. */
interface RawDex {
  species: { all(): unknown[] };
  moves: { all(): unknown[] };
  items: { all(): unknown[] };
  abilities: { all(): unknown[] };
  types: { all(): unknown[] };
  conditions: { get(id: string): unknown };
  data: { Conditions: Record<string, unknown> };
}

interface RawSim {
  Dex: {
    formats: { get(id: string): { exists: boolean; mod: string } };
    mod(name: string): RawDex;
  };
}

const bad = (message: string): never => {
  throw new EngineDataError(message);
};

function rec(value: unknown, what: string): Raw {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad(`${what}: not an object`);
  return value as Raw;
}
function str(value: unknown, what: string): string {
  if (typeof value !== "string" || value === "") bad(`${what}: not a non-empty string`);
  return value as string;
}
function int(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) bad(`${what}: not an integer`);
  return value as number;
}
function list(value: unknown, what: string): unknown[] {
  if (!Array.isArray(value)) bad(`${what}: not an array`);
  return value as unknown[];
}

const byId = <T extends { id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : 1);

/** Entries the format's mod plays with: the engine marks the rest nonstandard. */
const standard = (all: unknown[], what: string): Raw[] =>
  all.map((e) => rec(e, what)).filter((e) => e.exists === true && e.isNonstandard === null);

function named(entry: Raw, what: string): { id: string; name: string } {
  return { id: str(entry.id, `${what} id`), name: str(entry.name, `${what} name`) };
}

/** Resolve a reference by name to an id, refusing a name outside its vocabulary. */
function resolver(entries: { id: string; name: string }[], what: string): (name: unknown) => string {
  const ids = new Map(entries.map((e) => [e.name, e.id]));
  return (name) => ids.get(str(name, what)) ?? bad(`${what}: ${String(name)} is not in the vocabulary`);
}

const SLOTS = ["0", "1", "H", "S"];

/**
 * Export the vocabularies of `formatId`'s mod from the pinned engine. Refuses
 * any value of a shape it does not know, and (through `readEngineData`) any
 * table whose references do not close.
 */
export function exportEngineData(engine: Engine, formatId: string): EngineData {
  const sim = engine.sim as unknown as RawSim;
  const format = sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  const dex = sim.Dex.mod(format.mod);

  const typeEntries = standard(dex.types.all(), "type")
    .map((t) => ({ ...named(t, "type"), damageTaken: rec(t.damageTaken, "type damageTaken") }))
    .sort(byId);
  const typeOf = resolver(typeEntries, "type");
  // A standard forme may name a base species the engine marks nonstandard;
  // the vocabulary holds that one too, so no reference dangles.
  const bases = new Set(standard(dex.species.all(), "species").map((s) => str(s.baseSpecies, "species baseSpecies")));
  const speciesEntries = dex.species
    .all()
    .map((s) => rec(s, "species"))
    .filter((s) => s.exists === true && (s.isNonstandard === null || bases.has(str(s.name, "species name"))));
  // A standard species may list an ability the engine marks nonstandard (the
  // validator refuses it on a set); the vocabulary holds it so that no
  // reference dangles and the encoder never meets an unlisted ability.
  const listed = new Set(speciesEntries.flatMap((s) => Object.values(rec(s.abilities, "species abilities")).map((a) => str(a, "ability name"))));
  const abilityEntries = dex.abilities
    .all()
    .map((a) => rec(a, "ability"))
    .filter((a) => a.exists === true && (a.isNonstandard === null || listed.has(str(a.name, "ability name"))));
  const abilities = abilityEntries.map((a) => ({ ...named(a, "ability"), num: int(a.num, "ability num") })).sort(byId);
  const abilityOf = resolver(abilities, "ability");
  const speciesOf = resolver(speciesEntries.map((s) => named(s, "species")), "species");

  // The engine keys a type's damageTaken by the attacking type's name.
  const types = typeEntries.map((t) => ({
    id: t.id,
    name: t.name,
    damage_taken: typeEntries.map((attacker) => int(t.damageTaken[attacker.name], `${t.name} damageTaken ${attacker.name}`)),
  }));
  const species = speciesEntries
    .map((s) => {
      const stats = rec(s.baseStats, "species baseStats");
      const slots = rec(s.abilities, "species abilities");
      for (const key of Object.keys(slots)) if (!SLOTS.includes(key)) bad(`species abilities: unknown slot ${key}`);
      return {
        ...named(s, "species"),
        num: int(s.num, "species num"),
        base_species: speciesOf(s.baseSpecies),
        types: list(s.types, "species types").map(typeOf),
        base_stats: {
          hp: int(stats.hp, "hp"),
          atk: int(stats.atk, "atk"),
          def: int(stats.def, "def"),
          spa: int(stats.spa, "spa"),
          spd: int(stats.spd, "spd"),
          spe: int(stats.spe, "spe"),
        },
        weight_hg: int(s.weighthg, "species weighthg"),
        abilities: SLOTS.filter((slot) => slot in slots).map((slot) => abilityOf(slots[slot])),
      };
    })
    .sort(byId);

  const moveEntries = standard(dex.moves.all(), "move");
  const moves = moveEntries
    .map((m) => {
      const accuracy = m.accuracy === true ? 0 : int(m.accuracy, "move accuracy");
      return {
        ...named(m, "move"),
        num: int(m.num, "move num"),
        type: typeOf(m.type),
        category: str(m.category, "move category") as "Physical" | "Special" | "Status",
        base_power: int(m.basePower, "move basePower"),
        accuracy,
        always_hits: m.accuracy === true,
        priority: int(m.priority, "move priority"),
        pp: int(m.pp, "move pp"),
        crit_ratio: int(m.critRatio, "move critRatio"),
        target: str(m.target, "move target"),
        flags: Object.keys(rec(m.flags, "move flags")).sort(),
      };
    })
    .sort(byId);
  const unique = (values: string[]) => [...new Set(values)].sort();

  const items = standard(dex.items.all(), "item")
    .map((i) => ({
      ...named(i, "item"),
      num: int(i.num, "item num"),
      is_berry: i.isBerry === true,
      fling_power: i.fling === undefined ? 0 : int(rec(i.fling, "item fling").basePower, "item fling basePower"),
    }))
    .sort(byId);

  // The raw table is keyed by id; the condition object carries the name.
  const conditions = Object.keys(dex.data.Conditions).map((id) => rec(dex.conditions.get(id), "condition"));
  const ofKind = (effectType: string) =>
    conditions.filter((c) => c.effectType === effectType).map((c) => named(c, effectType)).sort(byId);
  const terrains = unique(moveEntries.filter((m) => m.terrain !== undefined).map((m) => str(m.terrain, "move terrain")))
    .map((id) => {
      const condition = rec(dex.conditions.get(id), "terrain condition");
      return condition.exists === true ? named(condition, "terrain") : bad(`terrain ${id}: the engine has no such condition`);
    });

  return readEngineData({
    schema_version: 1,
    showdown_commit: engine.commit,
    format_id: formatId,
    mod: format.mod,
    types,
    move_targets: unique(moves.map((m) => m.target)),
    move_flags: unique(moves.flatMap((m) => m.flags)),
    species,
    moves,
    items,
    abilities,
    statuses: ofKind("Status"),
    weathers: ofKind("Weather"),
    terrains,
  });
}

/** Strictly ascending ids: sorted by code unit, no repeat. */
function ordered(ids: string[], what: string): void {
  ids.forEach((id, k) => {
    if (k > 0 && !((ids[k - 1] as string) < id)) bad(`${what}: ${id} is repeated or out of order`);
  });
}

/**
 * Parse a stored table: the schema, then what the contract profile cannot say
 * (sorted and unique ids, references that resolve). Refuses anything else.
 */
export function readEngineData(value: unknown): EngineData {
  if (!validTable(value)) return bad(`invalid engine data: ${JSON.stringify(validTable.errors)}`);
  const table = value as unknown as EngineData;
  for (const key of ["types", "species", "moves", "items", "abilities", "statuses", "weathers", "terrains"] as const) {
    ordered(table[key].map((e) => e.id), key);
  }
  ordered(table.move_targets, "move_targets");
  ordered(table.move_flags, "move_flags");
  const ids = (entries: { id: string }[]) => new Set(entries.map((e) => e.id));
  const typeIds = ids(table.types);
  const abilityIds = ids(table.abilities);
  const speciesIds = ids(table.species);
  const check = (ok: boolean, message: string) => {
    if (!ok) bad(message);
  };
  for (const t of table.types) check(t.damage_taken.length === table.types.length, `type ${t.id}: damage_taken is not one entry per type`);
  for (const s of table.species) {
    check(s.types.every((t) => typeIds.has(t)), `species ${s.id}: unknown type`);
    check(s.abilities.every((a) => abilityIds.has(a)), `species ${s.id}: unknown ability`);
    check(speciesIds.has(s.base_species), `species ${s.id}: unknown base species`);
  }
  for (const m of table.moves) {
    check(typeIds.has(m.type), `move ${m.id}: unknown type`);
    check(table.move_targets.includes(m.target), `move ${m.id}: unknown target`);
    ordered(m.flags, `move ${m.id} flags`);
    check(m.flags.every((f) => table.move_flags.includes(f)), `move ${m.id}: unknown flag`);
  }
  return table;
}

/** SHA-256 of the table's JCS bytes: the engine-data hash of the encoder version. */
export function engineDataHash(table: EngineData): string {
  return createHash("sha256").update(jcs(table as unknown as Json), "utf8").digest("hex");
}

/** The reviewable summary committed beside the golden table. */
export function engineDataSummary(table: EngineData): {
  engine_data_hash: string;
  showdown_commit: string;
  counts: Record<string, number>;
} {
  const counts: Record<string, number> = {};
  for (const key of ["types", "species", "moves", "items", "abilities", "statuses", "weathers", "terrains"] as const) {
    counts[key] = table[key].length;
  }
  return { engine_data_hash: engineDataHash(table), showdown_commit: table.showdown_commit, counts };
}
