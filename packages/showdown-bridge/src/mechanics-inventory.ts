// The mechanics coverage inventory (ADR-030 groundwork G5, #1465): every move,
// ability, item and condition the format's roster can reach, with the engine
// handlers and the data flags each one carries, read from the pinned engine.
// A batched engine or a damage kernel admits content per entity (ADR-028 section
// 5, ADR-030 rule 2); this is the list the build order is read from.
//
// Nothing here is a game rule. An entity's handlers are the function-valued keys
// of its data object in the format's mod, found by reflection (so a handler name
// the engine adds appears with no code change); its data flags are the other keys
// that have a non-default value; the conditions it starts are the string fields
// that name one (`volatileStatus`, `sideCondition`, `weather`, ...) and the
// literals of the add/set calls in its handlers' own source text. Ids the
// simulation code tests directly (`hasAbility('x')`, `.id === 'x'`, a switch on an
// effect id) are scanned from `sim/*.ts` and the mod chain's `scripts.ts` and
// `conditions.ts`, so an effect implemented outside a handler (Tera Shell, #1166)
// is listed for its entity instead of missed. The only knowledge in code is how
// to walk the object, the call names the scan looks for, and the NAMES of the
// damage-relevant handlers (ADR-030's list), which only label a handler.
//
// Limits: a literal is found only when it is written as a string in one of the
// patterns below (an id held in a variable is not); a handler body is read from
// the function's source text, so a start made through a helper is not seen; a
// condition no roster entity starts is listed with an empty `started_by`. The
// table is derived from the pinned Showdown data files (MIT, not CC0;
// NOTICE-engine-data.txt beside it keeps the upstream notice).

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { jcs, type Json } from "./canonical-json.ts";
import { type EngineDataV2, engineDataHash } from "./engine-data.ts";
import { DEFAULT_SHOWDOWN_DIR, type Engine, EngineError } from "./engine.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";

export class MechanicsInventoryError extends Error {
  override name = "MechanicsInventoryError";
}

export const INVENTORY_FORMAT = "mechanics-inventory/1";
/** The golden export, beside the engine-data table. */
export const INVENTORY_FILE = "mechanics-inventory.json";
/** The inventory's hash and counts, small enough to review in a diff. */
export const INVENTORY_SUMMARY = "mechanics-inventory.summary.json";

export const KINDS = ["move", "ability", "item", "condition"] as const;
export type Kind = (typeof KINDS)[number];

/**
 * The handler names ADR-030 calls damage-relevant (its R-2 count of 382 definitions). They only
 * label a handler found by reflection: the set of handlers an entity has is never read from here.
 */
export const DAMAGE_HANDLER_NAMES: readonly string[] = [
  "basePowerCallback",
  "onAllyBasePower",
  "onAnyBasePower",
  "onAnyModifyDamage",
  "onBasePower",
  "onEffectiveness",
  "onModifyAtk",
  "onModifyCritRatio",
  "onModifyDamage",
  "onModifyDef",
  "onModifyMove",
  "onModifySpA",
  "onModifySpD",
  "onModifyType",
  "onSourceModifyAtk",
  "onSourceModifyDamage",
  "onSourceModifySpA",
];

/** Keys of an engine data object that only describe it (names, texts, numbers of the index). */
const DESCRIPTIVE = new Set(["name", "fullname", "id", "num", "gen", "exists", "isNonstandard", "effectType", "contestType", "spritenum", "rating", "desc", "shortDesc", "type", "category", "target"]);
/** Fields whose string value names a condition the entity starts. */
const START_FIELDS = new Set(["volatileStatus", "sideCondition", "slotCondition", "weather", "terrain", "pseudoWeather", "status"]);
const ORDER_KEY = /^on\w+(?:Priority|Order|SubOrder)$/;
const MAX_DEPTH = 4;

export interface InventoryEntity {
  id: string;
  /** Function-valued keys, dotted for a nested block (`condition.onStart`, `secondaries.onHit`), sorted. */
  handlers: string[];
  /** The subset of `handlers` whose last name is in `damage_handler_names`. */
  damage_handlers: string[];
  /** Numeric priority, order and sub-order siblings of the handlers, as `key=value`. */
  orders: string[];
  /** Non-default data keys (`key=value` for a string, boolean or stat change, else the key). */
  data_flags: string[];
  /** Conditions the entity starts: `<field or call>:<condition id>`. */
  starts: string[];
  /** Ids the entity's handler bodies test: `<call>:<id>`. */
  body_refs: string[];
  /** Simulation files that test the entity's id: `<file> <call>`. */
  sim_refs: string[];
  /** A move's category and target kind. */
  category?: string;
  target?: string;
  /** A condition's starters (`<kind>:<id>` of roster entities). */
  started_by?: string[];
  /** Where a condition is defined: the mod's condition table, or the `condition` block of the move, ability or item of that id. */
  defined_in?: "table" | Kind;
}

export interface InventorySpecies {
  id: string;
  abilities: string[];
  move_count: number;
  handlers: string[];
}

export interface MechanicsInventory {
  schema_version: 1;
  inventory_format: string;
  showdown_commit: string;
  format_id: string;
  mod: string;
  engine_data_hash: string;
  team_vocabulary_hash: string;
  damage_handler_names: string[];
  /** Source files the id scan read (relative to the checkout) and their SHA-256. */
  sources: { path: string; sha256: string }[];
  species: InventorySpecies[];
  moves: InventoryEntity[];
  abilities: InventoryEntity[];
  items: InventoryEntity[];
  conditions: InventoryEntity[];
  /** Condition ids an entity names that the engine has no condition for: `<kind>:<id> <call>:<id>`. */
  unresolved_starts: string[];
  /** Damage-handler definitions over EVERY entry of the mod, roster or not (the ADR counted a grep). */
  unfiltered: { abilities: number; conditions: number; items: number; moves: number };
}

const bad = (message: string): never => {
  throw new MechanicsInventoryError(message);
};
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (values: Iterable<string>): string[] => [...new Set(values)].sort(cmp);
export const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

// ---- reflection ----------------------------------------------------------------------------

interface Acc {
  handlers: Set<string>;
  orders: Set<string>;
  flags: Set<string>;
  starts: Set<string>;
  bodies: string[];
}

const isDefault = (value: unknown): boolean =>
  value === undefined || value === null || value === false || value === "" || value === 0 || (Array.isArray(value) && value.length === 0) || (isObject(value) && Object.keys(value).length === 0);

function walk(object: Record<string, unknown>, path: string, depth: number, acc: Acc): void {
  for (const [key, value] of Object.entries(object)) {
    if (DESCRIPTIVE.has(key)) continue;
    const here = path + key;
    if (typeof value === "function") {
      acc.handlers.add(here);
      acc.bodies.push(Function.prototype.toString.call(value));
    } else if (typeof value === "number" && ORDER_KEY.test(key)) {
      acc.orders.add(`${here}=${value}`);
    } else if (isDefault(value)) {
      continue;
    } else if (typeof value === "object") {
      acc.flags.add(here);
      if (depth >= MAX_DEPTH) continue;
      if (Array.isArray(value)) value.forEach((item, k) => isObject(item) && walk(item, `${here}[${k}].`, depth + 1, acc));
      else walk(value as Record<string, unknown>, `${here}.`, depth + 1, acc);
    } else {
      if (START_FIELDS.has(key) && typeof value === "string") acc.starts.add(`${here}:${toId(value)}`);
      const valued = typeof value !== "number" || path.endsWith("boosts.");
      acc.flags.add(valued ? `${here}=${String(value)}` : here);
    }
  }
}

// ---- id scan ---------------------------------------------------------------------------------

interface Pattern {
  via: string;
  /** The kind the literal names, or `any` when the call does not say (`.id === 'x'`). */
  kind: Kind | "any";
  re: RegExp;
  /** The call starts the named condition. */
  start?: boolean;
}

const QUOTED = String.raw`(\[[^\]]*\]|'[^']*'|"[^"]*")`;
const pattern = (via: string, kind: Pattern["kind"], source: string, start = false): Pattern => ({ via, kind, re: new RegExp(source.replace("Q", QUOTED), "g"), start });
const call = (name: string, kind: Kind, start = false): Pattern => pattern(name, kind, String.raw`\b${name}\(\s*Q`, start);

const PATTERNS: Pattern[] = [
  call("hasAbility", "ability"),
  call("hasItem", "item"),
  call("hasMove", "move"),
  call("hasVolatile", "condition"),
  call("getVolatile", "condition"),
  call("isWeather", "condition"),
  call("isTerrain", "condition"),
  call("addVolatile", "condition", true),
  call("setStatus", "condition", true),
  call("trySetStatus", "condition", true),
  call("setWeather", "condition", true),
  call("setTerrain", "condition", true),
  call("addSideCondition", "condition", true),
  call("addPseudoWeather", "condition", true),
  call("addSlotCondition", "condition", true),
  pattern("abilityIs", "ability", String.raw`\.(?:ability|baseAbility)\s*[!=]==?\s*Q`),
  pattern("itemIs", "item", String.raw`\.(?:item|lastItem)\s*[!=]==?\s*Q`),
  pattern("statusIs", "condition", String.raw`\.status\s*[!=]==?\s*Q`),
  pattern("volatileKey", "condition", String.raw`volatiles\[\s*Q\s*\]`),
  pattern("idIs", "any", String.raw`\.id\s*[!=]==?\s*Q`),
];

const SWITCH = /switch\s*\(([^)]*)\)\s*\{/g;
const CASE = new RegExp(String.raw`\bcase\s+${QUOTED}\s*:`, "g");

/** The quoted ids inside a captured argument (a string or an array of strings). */
const literals = (captured: string): string[] => [...captured.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => toId((m[1] ?? m[2]) as string)).filter((id) => id !== "");

export interface Ref {
  via: string;
  kind: Kind | "any";
  id: string;
  start: boolean;
}

/** The ids a piece of source text tests or starts, in the patterns above and in `case` labels of a switch on an id. */
export function scanRefs(text: string): Ref[] {
  const found: Ref[] = [];
  for (const p of PATTERNS) for (const m of text.matchAll(p.re)) for (const id of literals(m[1] as string)) found.push({ via: p.via, kind: p.kind, id, start: p.start === true });
  for (const m of text.matchAll(SWITCH)) {
    if (!/\.(?:id|name)\b/.test(m[1] as string)) continue;
    let depth = 1;
    let end = (m.index as number) + m[0].length;
    for (; end < text.length && depth > 0; end++) depth += text[end] === "{" ? 1 : text[end] === "}" ? -1 : 0;
    for (const c of text.slice((m.index as number) + m[0].length, end).matchAll(CASE)) for (const id of literals(c[1] as string)) found.push({ via: "switchId", kind: "any", id, start: false });
  }
  return found;
}

interface SimDex {
  parentMod: string;
  moves: { get(id: string): unknown; all(): unknown[] };
  abilities: { get(id: string): unknown; all(): unknown[] };
  items: { get(id: string): unknown; all(): unknown[] };
  conditions: { get(id: string): unknown };
  species: { get(id: string): unknown };
  data: { Conditions: Record<string, unknown> };
}
interface SimRoot {
  Dex: { formats: { get(id: string): { exists: boolean; mod: string } }; mod(name: string): SimDex };
}

/** The files the id scan reads: `sim/*.ts` and the `scripts.ts` and `conditions.ts` of the mod and its parents. */
export function scanFiles(dir: string, dex: (name: string) => SimDex, mod: string): string[] {
  const paths = readdirSync(join(dir, "sim")).filter((f) => f.endsWith(".ts")).sort(cmp).map((f) => `sim/${f}`);
  const seen = new Set<string>();
  for (let name = mod; name !== "" && name !== "base" && !seen.has(name); name = dex(name).parentMod) {
    seen.add(name);
    for (const file of ["scripts.ts", "conditions.ts"]) if (existsSync(join(dir, "data/mods", name, file))) paths.push(`data/mods/${name}/${file}`);
  }
  return paths;
}

// ---- export ----------------------------------------------------------------------------------

const damageName = (handler: string): boolean => DAMAGE_HANDLER_NAMES.includes(handler.slice(handler.lastIndexOf(".") + 1));
const countDamage = (entries: unknown[]): number =>
  entries.reduce<number>((total, entry) => {
    const acc: Acc = { handlers: new Set(), orders: new Set(), flags: new Set(), starts: new Set(), bodies: [] };
    if (isObject(entry) && entry.exists === true) walk(entry, "", 0, acc);
    return total + [...acc.handlers].filter(damageName).length;
  }, 0);

/**
 * Export the inventory of `formatId` from the pinned engine for the roster `vocabulary` and the
 * engine-data version 2 table `table`. `dir` is the checkout the id scan reads (default the pinned one).
 */
export function exportMechanicsInventory(engine: Engine, formatId: string, vocabulary: TeamVocabulary, table: EngineDataV2, dir: string = DEFAULT_SHOWDOWN_DIR): MechanicsInventory {
  const sim = engine.sim as unknown as SimRoot;
  const format = sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  if (vocabulary.format_id !== formatId || vocabulary.showdown_commit !== engine.commit) bad("the team vocabulary is for another format or pin");
  if (table.format_id !== formatId || table.showdown_commit !== engine.commit) bad("the engine-data table is for another format or pin");
  const dex = sim.Dex.mod(format.mod);

  const ids: Record<Kind, string[]> = {
    move: sorted(vocabulary.roster.flatMap((s) => s.moves)),
    ability: sorted(vocabulary.roster.flatMap((s) => s.abilities)),
    item: sorted(vocabulary.items),
    condition: [],
  };
  const resolves = (id: string): boolean => isObject(dex.conditions.get(id)) && (dex.conditions.get(id) as Record<string, unknown>).exists === true;
  const getters: Record<Kind, (id: string) => unknown> = {
    move: (id) => dex.moves.get(id),
    ability: (id) => dex.abilities.get(id),
    item: (id) => dex.items.get(id),
    condition: (id) => dex.conditions.get(id),
  };

  // The id scan over the simulation code, attributed to the entities whose ids it names.
  const files = scanFiles(dir, (name) => sim.Dex.mod(name), format.mod);
  const sources = files.map((path) => ({ path, text: readFileSync(join(dir, path), "utf8") }));
  const simRefs = new Map<string, Set<string>>();
  const note = (map: Map<string, Set<string>>, key: string, value: string): void => void map.set(key, (map.get(key) ?? new Set()).add(value));

  // Entities of the first three kinds, then the conditions they start (and the engine's own list).
  const records = new Map<string, InventoryEntity>();
  const started = new Map<string, Set<string>>();
  const unresolved = new Set<string>();
  const build = (kind: Kind, id: string): InventoryEntity => {
    const raw = getters[kind](id);
    if (!isObject(raw) || raw.exists !== true) bad(`${kind} ${id}: the engine has no such entry`);
    const acc: Acc = { handlers: new Set(), orders: new Set(), flags: new Set(), starts: new Set(), bodies: [] };
    walk(raw as Record<string, unknown>, "", 0, acc);
    const refs = new Set<string>();
    for (const body of acc.bodies) {
      for (const r of scanRefs(body)) {
        if (r.start) acc.starts.add(`${r.via}:${r.id}`);
        else refs.add(`${r.via}:${r.id}`);
      }
    }
    if (kind !== "condition" && (raw as Record<string, unknown>).condition !== undefined) acc.starts.add(`condition:${id}`);
    const entity: InventoryEntity = {
      id,
      handlers: sorted(acc.handlers),
      damage_handlers: sorted([...acc.handlers].filter(damageName)),
      orders: sorted(acc.orders),
      data_flags: sorted(acc.flags),
      starts: sorted(acc.starts),
      body_refs: sorted(refs),
      sim_refs: [],
    };
    if (kind === "move") {
      entity.category = String((raw as Record<string, unknown>).category);
      entity.target = String((raw as Record<string, unknown>).target);
    }
    records.set(`${kind}:${id}`, entity);
    return entity;
  };
  for (const kind of ["move", "ability", "item"] as const) for (const id of ids[kind]) build(kind, id);
  for (const [key, entity] of records) {
    entity.starts = entity.starts.filter((start) => {
      const id = start.slice(start.lastIndexOf(":") + 1);
      if (!resolves(id)) {
        unresolved.add(`${key} ${start}`);
        return false;
      }
      note(started, id, key);
      return true;
    });
  }
  const statusIds = [...table.statuses, ...table.weathers, ...table.terrains].map((e) => e.id);
  ids.condition = sorted([...table.conditions.map((c) => c.id), ...statusIds, ...started.keys()]);
  for (const id of ids.condition) {
    const entity = build("condition", id);
    entity.started_by = sorted(started.get(id) ?? []);
    entity.defined_in = Object.hasOwn(dex.data.Conditions, id) ? "table" : (["move", "ability", "item"] as const).find((k) => records.has(`${k}:${id}`)) ?? "table";
  }
  // What a condition's own handlers start must resolve too (they are not counted as starters of the roster).
  for (const kind of ["condition"] as const) {
    for (const id of ids[kind]) {
      const entity = records.get(`${kind}:${id}`) as InventoryEntity;
      entity.starts = entity.starts.filter((start) => resolves(start.slice(start.lastIndexOf(":") + 1)) || !unresolved.add(`${kind}:${id} ${start}`));
    }
  }

  for (const file of sources) {
    for (const r of scanRefs(file.text)) {
      for (const kind of r.kind === "any" ? KINDS : [r.kind]) {
        if (ids[kind].includes(r.id)) note(simRefs, `${kind}:${r.id}`, `${file.path} ${r.via}`);
      }
    }
  }
  for (const [key, entity] of records) entity.sim_refs = sorted(simRefs.get(key) ?? []);

  const roster = vocabulary.roster.map((s): InventorySpecies => {
    const acc: Acc = { handlers: new Set(), orders: new Set(), flags: new Set(), starts: new Set(), bodies: [] };
    const raw = dex.species.get(s.id);
    if (isObject(raw)) walk(raw, "", 0, acc);
    return { id: s.id, abilities: s.abilities, move_count: s.moves.length, handlers: sorted(acc.handlers) };
  });
  const list = (kind: Kind): InventoryEntity[] => ids[kind].map((id) => records.get(`${kind}:${id}`) as InventoryEntity);
  return {
    schema_version: 1,
    inventory_format: INVENTORY_FORMAT,
    showdown_commit: engine.commit,
    format_id: formatId,
    mod: format.mod,
    engine_data_hash: engineDataHash(table),
    team_vocabulary_hash: teamVocabularyHash(vocabulary),
    damage_handler_names: [...DAMAGE_HANDLER_NAMES],
    sources: sources.map((s) => ({ path: s.path, sha256: sha256(s.text) })),
    species: roster,
    moves: list("move"),
    abilities: list("ability"),
    items: list("item"),
    conditions: list("condition"),
    unresolved_starts: sorted(unresolved),
    unfiltered: {
      moves: countDamage(dex.moves.all()),
      abilities: countDamage(dex.abilities.all()),
      items: countDamage(dex.items.all()),
      conditions: countDamage(Object.keys(dex.data.Conditions).map((id) => dex.conditions.get(id))),
    },
  };
}

// ---- reading and rendering --------------------------------------------------------------------

const LIST_KEYS = ["handlers", "damage_handlers", "orders", "data_flags", "starts", "body_refs", "sim_refs"] as const;

function strings(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) return bad(`${what}: not a list of strings`);
  const list = value as string[];
  list.forEach((v, k) => k > 0 && cmp(list[k - 1] as string, v) >= 0 && bad(`${what}: ${v} is repeated or out of order`));
  return list;
}

/**
 * Parse a stored inventory: its shape, sorted unique ids and lists, damage handlers a subset of
 * the handlers, and, given the roster and the engine-data table, that the move, ability, item and
 * condition ids are exactly theirs (conditions: the table's conditions, statuses, weathers and
 * terrains, and the ids the entities start).
 */
export function readMechanicsInventory(value: unknown, vocabulary?: TeamVocabulary, table?: EngineDataV2): MechanicsInventory {
  if (!isObject(value) || value.schema_version !== 1 || value.inventory_format !== INVENTORY_FORMAT) return bad("not a mechanics inventory of this format");
  const inv = value as unknown as MechanicsInventory;
  const kinds: Record<Kind, InventoryEntity[]> = { move: inv.moves, ability: inv.abilities, item: inv.items, condition: inv.conditions };
  for (const kind of KINDS) {
    if (!Array.isArray(kinds[kind])) bad(`${kind}s: not a list`);
    strings(
      kinds[kind].map((e) => e.id),
      `${kind} ids`,
    );
    for (const entity of kinds[kind]) {
      for (const key of LIST_KEYS) strings(entity[key], `${kind} ${entity.id} ${key}`);
      for (const h of entity.damage_handlers) if (!entity.handlers.includes(h) || !damageName(h)) bad(`${kind} ${entity.id}: ${h} is not a damage handler of its handlers`);
      if (kind === "condition" && (!Array.isArray(entity.started_by) || entity.defined_in === undefined)) bad(`condition ${entity.id}: no started_by or defined_in`);
    }
  }
  strings(inv.damage_handler_names, "damage_handler_names");
  if (vocabulary !== undefined && table !== undefined) {
    if (inv.team_vocabulary_hash !== teamVocabularyHash(vocabulary) || inv.engine_data_hash !== engineDataHash(table)) bad("the inventory was made against another vocabulary or engine-data table");
    const expected: Record<Kind, string[]> = {
      move: sorted(vocabulary.roster.flatMap((s) => s.moves)),
      ability: sorted(vocabulary.roster.flatMap((s) => s.abilities)),
      item: sorted(vocabulary.items),
      condition: [...table.conditions, ...table.statuses, ...table.weathers, ...table.terrains].map((e) => e.id),
    };
    for (const kind of KINDS) {
      const have = new Set(kinds[kind].map((e) => e.id));
      for (const id of expected[kind]) if (!have.has(id)) bad(`${kind} ${id} of the vocabulary or table is missing`);
      if (kind !== "condition" && have.size !== expected[kind].length) bad(`${kind}s: not exactly the vocabulary's`);
    }
  }
  return inv;
}

/** The golden file's bytes: pretty at the top, one entity per line. */
export function inventoryBytes(inv: MechanicsInventory): string {
  const entityList = (list: unknown[]): string => `[\n${list.map((e) => `    ${JSON.stringify(e)}`).join(",\n")}\n  ]`;
  const body = Object.entries(inv)
    .map(([key, value]) => `  ${JSON.stringify(key)}: ${Array.isArray(value) && value.length > 0 && typeof value[0] === "object" ? entityList(value) : JSON.stringify(value)}`)
    .join(",\n");
  return `{\n${body}\n}\n`;
}

/** SHA-256 of the inventory's JCS bytes. */
export const inventoryHash = (inv: MechanicsInventory): string => sha256(jcs(inv as unknown as Json));

export interface InventorySummary {
  inventory_hash: string;
  engine_data_hash: string;
  showdown_commit: string;
  counts: Record<string, number>;
}

/** The reviewable summary committed beside the golden file. */
export function inventorySummary(inv: MechanicsInventory): InventorySummary {
  const counts: Record<string, number> = { species: inv.species.length };
  for (const kind of KINDS) {
    const list = ({ move: inv.moves, ability: inv.abilities, item: inv.items, condition: inv.conditions } as Record<Kind, InventoryEntity[]>)[kind];
    const plural = `${kind === "ability" ? "abilitie" : kind}s`;
    counts[plural] = list.length;
    counts[`${plural}_with_handlers`] = list.filter((e) => e.handlers.length > 0).length;
    counts[`${plural}_with_damage_handlers`] = list.filter((e) => e.damage_handlers.length > 0).length;
    counts[`${kind}_damage_handler_definitions`] = list.reduce((n, e) => n + (e.defined_in === undefined || e.defined_in === "table" ? e.damage_handlers.length : 0), 0);
  }
  counts.sim_referenced = [...inv.moves, ...inv.abilities, ...inv.items, ...inv.conditions].filter((e) => e.sim_refs.length > 0).length;
  return { inventory_hash: inventoryHash(inv), engine_data_hash: inv.engine_data_hash, showdown_commit: inv.showdown_commit, counts };
}
