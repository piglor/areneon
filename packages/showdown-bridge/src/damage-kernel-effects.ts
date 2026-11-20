// The damage kernel's effects record (ADR-030 step 3, #1457): reviewed, hashed data naming the abilities and items whose hooks
// reach the probe's damage path and that the kernel implements, each by an effect kind and the engine's own parameters (a type,
// a move flag, a 4096-based modifier). The kernel holds the effect KINDS (how each step of `getDamage` applies one); the ids and
// values are here, never in code. An id that is not listed and has a hook the kernel's inert list does not hold makes the kernel
// abstain. The record is bound to the pinned engine commit and its format; a pin bump starts it again.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";

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
  | { kind: "no_damage_effect" } // hooks on the damage path that never change a damaging move's answer
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
}

export class KernelEffectsError extends Error {
  override name = "KernelEffectsError";
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
  no_damage_effect: {},
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

/** Read the record: closed keys, the effect kinds and their fields, sorted ids. */
export function readKernelEffects(value: unknown): KernelEffects {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail("the effects record is an object");
  const v = value as Record<string, unknown>;
  const keys = ["abilities", "data_class", "description", "format_id", "items", "moves", "schema_version", "showdown_commit"];
  if (jcs(Object.keys(v).sort() as unknown as Json) !== jcs(keys as unknown as Json)) fail("the effects record has exactly its keys");
  if (v.schema_version !== 1 || v.data_class !== "synthetic") fail("the effects record is schema version 1, synthetic");
  if (typeof v.format_id !== "string" || typeof v.description !== "string" || typeof v.showdown_commit !== "string" || !/^[0-9a-f]{40}$/.test(v.showdown_commit)) fail("the effects record names its format and commit");
  return { ...(v as unknown as KernelEffects), abilities: readTable(v.abilities, ABILITY_KINDS, "abilities"), items: readTable(v.items, ITEM_KINDS, "items"), moves: readTable(v.moves, MOVE_KINDS, "moves") };
}

/** SHA-256 of the record's JCS bytes (part of the kernel's identity). */
export const kernelEffectsHash = (effects: KernelEffects): string => sha256Hex(jcs(effects as unknown as Json));

/** The committed record of a format, bound to the engine's pin (another pin's record is refused: the kernel abstains on everything). */
export function loadKernelEffects(formatId: string, commit: string): KernelEffects {
  const path = fileURLToPath(new URL(`../../../config/mechanics/${formatId}.damage-kernel.json`, import.meta.url));
  const effects = readKernelEffects(JSON.parse(readFileSync(path, "utf8")));
  if (effects.format_id !== formatId) fail("the effects record is for another format");
  if (effects.showdown_commit !== commit) fail("the effects record is for another engine pin");
  return effects;
}
