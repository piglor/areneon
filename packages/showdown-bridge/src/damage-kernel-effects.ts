// The damage kernel's effects record (ADR-030 step 3, #1457): reviewed, hashed data naming the abilities, items and moves whose
// hooks reach the probe's damage path and that the kernel implements, and the weathers, terrains and side and field conditions it
// implements, each by an effect kind and the engine's own parameters (a type, a move flag, a move list, a 4096-based modifier).
// The kernel holds the effect KINDS (the mechanism: how each step of `getDamage` applies one); the ids and values are here, never
// in code (owner rule R7). An id that is not listed and has a hook the kernel's inert list does not hold makes the kernel abstain;
// a condition that is not listed is not implemented. `checkKernelEffects` holds each entry to the engine's data: its kind must
// explain every damage-path hook the entity has (a `no_damage_effect` entry names exactly the hooks it hides), or the record is
// refused. The record is bound to the pinned engine commit and its format; a pin bump starts it again.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import type { KernelData } from "./damage-kernel-data.ts";

const TYPE = /^[A-Z][a-z]+$/;
const ID = /^[a-z0-9]+$/;

/** One effect an ability or item has on a `damage` answer. Modifiers are the engine's 4096-based integers. */
export type KernelEffect =
  | { kind: "pinch"; type: string } // the holder's Atk / Sp. Atk 1.5 for a move of the type at a third of its HP or less
  | { kind: "base_power_flag"; flag: string; modifier: number } // base power, a move with the flag
  | { kind: "stab"; modifier: number } // the same-type modifier
  | { kind: "no_crit" } // the holder is never hit critically
  | { kind: "guts" } // Atk 1.5 with any status, and a burn does not halve
  | { kind: "attack_multiplier"; stat: "atk" | "spa"; modifier: number; weather?: string } // the holder's attacking stat
  | { kind: "source_type_attack"; types: string[]; modifier: number } // the attacker's attacking stat, a move of the types into the holder
  | { kind: "technician"; threshold: number; modifier: number } // base power at most the threshold
  | { kind: "super_effective_taken"; modifier: number } // final damage into the holder of a super-effective move
  | { kind: "full_hp_taken"; modifier: number } // final damage into the holder at full HP
  | { kind: "crit_damage"; modifier: number } // final damage of the holder's critical hit
  | { kind: "defense_multiplier"; stat: "def" | "spd"; modifier: number; terrain?: string } // the holder's defending stat
  | { kind: "ground_immunity" } // the holder is not grounded
  | { kind: "suppress_weather" } // while in play, no weather has an effect
  | { kind: "mold_breaker" } // the holder's moves ignore a breakable ability of the target
  | { kind: "absorb"; type: string } // a move of the type into the holder deals nothing
  | { kind: "flag_immunity"; flag: string } // a move with the flag into the holder deals nothing
  | { kind: "infiltrator" } // the holder's moves ignore the target's screens
  | { kind: "no_damage_effect"; hides: string[] } // the hooks (named) it has on the damage path never change a damaging move's answer
  | { kind: "type_boost"; type: string; modifier: number } // base power, a move of the type
  | { kind: "resist_berry"; type: string; super_effective: boolean } // final damage halved into the holder (eaten)
  | { kind: "final_modifier"; modifier: number } // final damage of the holder's move
  | { kind: "super_effective_dealt"; modifier: number } // final damage of the holder's super-effective move
  | { kind: "category_power"; category: "Physical" | "Special"; modifier: number } // base power, a move of the category
  | { kind: "crit_ratio"; add: number } // the holder's crit ratio
  | { kind: "status_power"; modifier: number; burn_exempt: boolean } // a move's base power while its user has a status (not sleep)
  | { kind: "priority_block" } // a priority move into the holder's side fails (the probe answers `unsupported`)
  | { kind: "ate"; type: string; modifier: number; exceptions: string[] } // the holder's Normal moves become the type, with a base power boost
  | { kind: "sheer_force"; modifier: number } // a move with secondary effects loses them and gains base power
  | { kind: "unnerve" } // the holder's foes cannot eat berries
  | { kind: "needs_sleep" } // a move that works only while its user sleeps (else the probe answers `unsupported`)
  | { kind: "constant_power"; multiplier: number } // a move whose base power callback returns its base power times this in the probe's scenario (an empty queue, no switch this turn)
  | { kind: "boost_power"; per: number } // base power plus `per` for each positive stat stage of the user
  | { kind: "fails" }; // a move the probe's move step never lets reach the damage loop (`unsupported`)

/** A weather, terrain or side or field condition the kernel implements, by the engine's id. */
export type KernelCondition =
  | { kind: "weather_damage"; types: { type: string; modifier: number }[]; exceptions: string[] } // WeatherModifyDamage of a move of a type (abstain on an exception)
  | { kind: "weather_stat"; stat: "def" | "spd"; type: string; modifier: number } // the defender's stat of a holder of the type, modified at once (not chained)
  | { kind: "terrain"; boost: { type: string; modifier: number } | null; weakens: { moves: string[]; modifier: number } | null; halves: { type: string; modifier: number } | null; blocks_priority: boolean }
  | { kind: "screen"; category: "Physical" | "Special" | null; yields_to: string[]; modifier: number } // the defender's side, not on a crit; `null` category: both
  | { kind: "no_damage_effect"; hides: string[] }; // Trick Room, Tailwind

export interface KernelEffects {
  schema_version: 1;
  data_class: "synthetic";
  format_id: string;
  showdown_commit: string;
  description: string;
  abilities: Record<string, KernelEffect>;
  items: Record<string, KernelEffect>;
  /** Moves whose own hooks the kernel implements or knows change no damage answer. */
  moves: Record<string, KernelEffect>;
  /** The weathers, terrains and side and field conditions the kernel implements. */
  conditions: Record<string, KernelCondition>;
}

export class KernelEffectsError extends Error {
  override name = "KernelEffectsError";
}

/** A record whose entry does not explain its entity's hooks (a defect of the record, not a pin mismatch). */
export class KernelRecordError extends Error {
  override name = "KernelRecordError";
}

const fail = (message: string): never => {
  throw new KernelEffectsError(message);
};
const isModifier = (v: unknown): boolean => Number.isSafeInteger(v) && (v as number) > 0 && (v as number) <= 65536;

/** The fields each kind takes, and how each is checked. */
const FIELDS: Record<KernelEffect["kind"], Record<string, (v: unknown) => boolean>> = {
  pinch: { type: (v) => typeof v === "string" && TYPE.test(v) },
  base_power_flag: { flag: (v) => typeof v === "string" && ID.test(v), modifier: isModifier },
  stab: { modifier: isModifier },
  no_crit: {},
  guts: {},
  attack_multiplier: { stat: (v) => v === "atk" || v === "spa", modifier: isModifier, weather: (v) => v === undefined || (typeof v === "string" && ID.test(v)) },
  source_type_attack: { types: (v) => Array.isArray(v) && v.length > 0 && v.every((t) => typeof t === "string" && TYPE.test(t)), modifier: isModifier },
  technician: { threshold: (v) => Number.isSafeInteger(v) && (v as number) > 0, modifier: isModifier },
  super_effective_taken: { modifier: isModifier },
  full_hp_taken: { modifier: isModifier },
  crit_damage: { modifier: isModifier },
  defense_multiplier: { stat: (v) => v === "def" || v === "spd", modifier: isModifier, terrain: (v) => v === undefined || (typeof v === "string" && ID.test(v)) },
  ground_immunity: {},
  suppress_weather: {},
  mold_breaker: {},
  absorb: { type: (v) => typeof v === "string" && TYPE.test(v) },
  flag_immunity: { flag: (v) => typeof v === "string" && ID.test(v) },
  infiltrator: {},
  no_damage_effect: { hides: (v) => Array.isArray(v) && v.length > 0 && v.every((h, i) => typeof h === "string" && (i === 0 || h > (v[i - 1] as string))) },
  type_boost: { type: (v) => typeof v === "string" && TYPE.test(v), modifier: isModifier },
  resist_berry: { type: (v) => typeof v === "string" && TYPE.test(v), super_effective: (v) => typeof v === "boolean" },
  final_modifier: { modifier: isModifier },
  super_effective_dealt: { modifier: isModifier },
  category_power: { category: (v) => v === "Physical" || v === "Special", modifier: isModifier },
  crit_ratio: { add: (v) => Number.isSafeInteger(v) && (v as number) >= 1 && (v as number) <= 4 },
  status_power: { modifier: isModifier, burn_exempt: (v) => typeof v === "boolean" },
  priority_block: {},
  ate: { type: (v) => typeof v === "string" && TYPE.test(v), modifier: isModifier, exceptions: (v) => Array.isArray(v) && v.every((m) => typeof m === "string" && ID.test(m)) },
  sheer_force: { modifier: isModifier },
  unnerve: {},
  needs_sleep: {},
  fails: {},
  constant_power: { multiplier: (v) => Number.isSafeInteger(v) && (v as number) >= 1 && (v as number) <= 4 },
  boost_power: { per: (v) => Number.isSafeInteger(v) && (v as number) > 0 },
};
/** Kinds that only an ability has, and kinds that only an item has. */
const ITEM_KINDS = new Set(["type_boost", "resist_berry", "final_modifier", "super_effective_dealt", "category_power", "crit_ratio", "no_damage_effect"]);
const MOVE_KINDS = new Set(["no_damage_effect", "status_power", "needs_sleep", "fails", "constant_power", "boost_power"]);
const ABILITY_KINDS = new Set(Object.keys(FIELDS).filter((k) => (!ITEM_KINDS.has(k) && !MOVE_KINDS.has(k)) || k === "no_damage_effect"));

function readTable(value: unknown, kinds: Set<string>, what: string): Record<string, KernelEffect> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(`the ${what} table is an object`);
  const out: Record<string, KernelEffect> = {};
  const ids = Object.keys(value as object);
  if (ids.some((id, i) => !ID.test(id) || (i > 0 && id <= (ids[i - 1] as string)))) fail(`the ${what} ids are engine ids, sorted and unique`);
  for (const id of ids) {
    const effect = (value as Record<string, unknown>)[id] as Record<string, unknown>;
    if (effect === null || typeof effect !== "object" || typeof effect.kind !== "string" || !kinds.has(effect.kind)) fail(`an effect of the ${what} table names a kind it may hold`);
    const fields = FIELDS[effect.kind as KernelEffect["kind"]];
    for (const key of Object.keys(effect)) if (key !== "kind" && !(key in fields)) fail(`an effect has a field its kind does not take`);
    for (const [key, check] of Object.entries(fields)) if (!check(effect[key])) fail(`an effect's field does not have its kind's shape`);
    out[id] = effect as KernelEffect;
  }
  return out;
}

const isType = (v: unknown): boolean => typeof v === "string" && TYPE.test(v);
const isIds = (v: unknown): boolean => Array.isArray(v) && v.every((m, i) => typeof m === "string" && ID.test(m) && (i === 0 || m > (v[i - 1] as string)));
const typed = (v: unknown): boolean => v !== null && typeof v === "object" && !Array.isArray(v) && Object.keys(v).sort().join() === "modifier,type" && isType((v as { type: unknown }).type) && isModifier((v as { modifier: unknown }).modifier);
const CONDITION_FIELDS: Record<KernelCondition["kind"], Record<string, (v: unknown) => boolean>> = {
  weather_damage: { types: (v) => Array.isArray(v) && v.length > 0 && v.every(typed), exceptions: isIds },
  weather_stat: { stat: (v) => v === "def" || v === "spd", type: isType, modifier: isModifier },
  terrain: {
    boost: (v) => v === null || typed(v),
    weakens: (v) => v === null || (v !== null && typeof v === "object" && Object.keys(v).sort().join() === "modifier,moves" && isIds((v as { moves: unknown }).moves) && isModifier((v as { modifier: unknown }).modifier)),
    halves: (v) => v === null || typed(v),
    blocks_priority: (v) => typeof v === "boolean",
  },
  screen: { category: (v) => v === null || v === "Physical" || v === "Special", yields_to: isIds, modifier: isModifier },
  no_damage_effect: FIELDS.no_damage_effect,
};

function readConditions(value: unknown): Record<string, KernelCondition> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail("the conditions table is an object");
  const ids = Object.keys(value as object);
  if (ids.some((id, i) => !ID.test(id) || (i > 0 && id <= (ids[i - 1] as string)))) fail("the conditions ids are engine ids, sorted and unique");
  const out: Record<string, KernelCondition> = {};
  for (const id of ids) {
    const c = (value as Record<string, unknown>)[id] as Record<string, unknown>;
    const fields = c !== null && typeof c === "object" && typeof c.kind === "string" ? CONDITION_FIELDS[c.kind as KernelCondition["kind"]] : undefined;
    if (fields === undefined) fail("a condition of the record names a kind it may hold");
    const known = fields as Record<string, (v: unknown) => boolean>;
    for (const key of Object.keys(c)) if (key !== "kind" && !(key in known)) fail("a condition has a field its kind does not take");
    for (const [key, check] of Object.entries(known)) if (!check(c[key])) fail("a condition's field does not have its kind's shape");
    out[id] = c as KernelCondition;
  }
  return out;
}

/** The damage-path hooks each kind explains (an entity's blocking hooks must all be among them; `no_damage_effect` names its own). */
const EXPLAINS: Record<Exclude<KernelEffect["kind"], "no_damage_effect">, string[]> = {
  pinch: ["onModifyAtk", "onModifySpA"],
  base_power_flag: ["onBasePower"],
  stab: ["onModifySTAB"],
  no_crit: ["onCriticalHit"],
  guts: ["onModifyAtk", "sim_literal"],
  attack_multiplier: ["onModifyAtk", "onModifySpA"],
  source_type_attack: ["onSourceModifyAtk", "onSourceModifySpA"],
  technician: ["onBasePower"],
  super_effective_taken: ["onSourceModifyDamage"],
  full_hp_taken: ["onSourceModifyDamage"],
  crit_damage: ["onModifyDamage"],
  defense_multiplier: ["onModifyDef", "onModifySpD"],
  ground_immunity: ["sim_literal"],
  suppress_weather: ["key:suppressWeather", "onSwitchIn"],
  mold_breaker: ["onModifyMove"],
  absorb: ["onAllyTryHitSide", "onAnyRedirectTarget", "onTryHit"],
  flag_immunity: ["onAllyTryHitSide", "onTryHit"],
  infiltrator: ["onModifyMove"],
  type_boost: ["onBasePower"],
  resist_berry: ["onSourceModifyDamage"],
  final_modifier: ["onAfterMoveSecondarySelf", "onModifyDamage"],
  super_effective_dealt: ["onModifyDamage"],
  category_power: ["onBasePower"],
  crit_ratio: ["onModifyCritRatio"],
  status_power: ["onBasePower", "sim_literal"],
  priority_block: ["onFoeTryMove"],
  ate: ["onBasePower", "onModifyType"],
  sheer_force: ["onBasePower", "onModifyMove", "sim_literal"],
  unnerve: ["onFoeTryEatItem", "onSwitchIn"],
  needs_sleep: ["onTry"],
  constant_power: ["basePowerCallback", "onTry"],
  boost_power: ["basePowerCallback"],
  fails: ["onTry", "onTryMove", "sim_literal"],
};

/**
 * Hold every entry to the engine's data: the entity exists, and its kind explains every hook (and simulator by-name test) it has
 * on the damage path; a `no_damage_effect` entry names exactly those it hides. A mis-kinded entry is a `KernelRecordError`.
 */
export function checkKernelEffects(effects: KernelEffects, data: KernelData): void {
  const check = (what: string, id: string, effect: KernelEffect, blockers: readonly string[] | null): void => {
    if (blockers === null) throw new KernelRecordError(`the effects record names ${what} ${id}, which the engine does not have`);
    if (effect.kind === "no_damage_effect") {
      if (effect.hides.join() !== blockers.join()) throw new KernelRecordError(`the effects record's ${what} ${id} hides other hooks than it has: ${blockers.join(",")}`);
      return;
    }
    const explained = new Set(EXPLAINS[effect.kind]);
    const unexplained = blockers.filter((b) => !explained.has(b));
    if (unexplained.length > 0) throw new KernelRecordError(`the effects record's ${what} ${id} is kind ${effect.kind}, which does not explain ${unexplained.join(",")}`);
  };
  for (const [id, effect] of Object.entries(effects.abilities)) check("ability", id, effect, data.abilityBlockers(id, false));
  for (const [id, effect] of Object.entries(effects.items)) check("item", id, effect, data.itemBlockers(id, false));
  for (const [id, effect] of Object.entries(effects.moves)) check("move", id, effect, data.move(id)?.blockers ?? null);
  for (const id of Object.keys(effects.conditions)) if (data.conditionType(id) === null) throw new KernelRecordError(`the effects record names condition ${id}, which the engine does not have`);
}

/** Read the record: closed keys, the effect kinds and their fields, sorted ids. */
export function readKernelEffects(value: unknown): KernelEffects {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail("the effects record is an object");
  const v = value as Record<string, unknown>;
  const keys = ["abilities", "conditions", "data_class", "description", "format_id", "items", "moves", "schema_version", "showdown_commit"];
  if (jcs(Object.keys(v).sort() as unknown as Json) !== jcs(keys as unknown as Json)) fail("the effects record has exactly its keys");
  if (v.schema_version !== 1 || v.data_class !== "synthetic") fail("the effects record is schema version 1, synthetic");
  if (typeof v.format_id !== "string" || typeof v.description !== "string" || typeof v.showdown_commit !== "string" || !/^[0-9a-f]{40}$/.test(v.showdown_commit)) fail("the effects record names its format and commit");
  return { ...(v as unknown as KernelEffects), abilities: readTable(v.abilities, ABILITY_KINDS, "abilities"), items: readTable(v.items, ITEM_KINDS, "items"), moves: readTable(v.moves, MOVE_KINDS, "moves"), conditions: readConditions(v.conditions) };
}

/** SHA-256 of the record's JCS bytes (part of the kernel's identity). */
export const kernelEffectsHash = (effects: KernelEffects): string => sha256Hex(jcs(effects as unknown as Json));

/** The committed record of a format, bound to the engine's pin (another pin's record is refused: the kernel abstains on everything). */
export function loadKernelEffects(formatId: string, commit: string): KernelEffects {
  const path = fileURLToPath(new URL(`../../../config/backends/${formatId}.damage-kernel.effects.json`, import.meta.url));
  const effects = readKernelEffects(JSON.parse(readFileSync(path, "utf8")));
  if (effects.format_id !== formatId) fail("the effects record is for another format");
  if (effects.showdown_commit !== commit) fail("the effects record is for another engine pin");
  return effects;
}
