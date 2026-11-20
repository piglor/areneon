// The data the damage kernel reads (ADR-030 step 3, #1457): the pinned engine's DATA for one format's mod, read once into
// plain tables (species base stats and types, the type chart, natures, moves' damage fields, Mega Stones), and which entities
// the kernel can treat as having no effect on a `damage` answer. No game value is written here: every number comes from the
// engine's data objects, and an entity's inertness is read from its own data by reflection.
//
// Inertness (the kernel's refusal rule): an ability, item or move is INERT for a `damage` query when none of its data keys is a
// hook the probe's damage path can reach. A hook is any key that starts with `on` or ends with `Callback` (a function, or a
// constant such as Battle Armor's `onCriticalHit: false`), also inside `condition`, `secondary`, `secondaries`, `self` and
// `fling`. `INERT_HOOKS` is the reviewed list of hook names the probe's path cannot reach or whose effect the probe overwrites
// (the scenario start is followed by the declared stages, items, field and HP; nothing after `getDamage` is read). Every other
// hook makes the entity non-inert, and the kernel abstains on a query that names it unless the kernel implements it. An id the
// pinned simulator code tests by a string literal (`hasItem('ironball')`, `hasAbility('levitate')`, `move.id === 'facade'`) is
// non-inert too, whatever its hooks: the simulator reads it outside any hook. That scan reads the pinned sources named below.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { DEFAULT_SHOWDOWN_DIR, type Engine } from "./engine.ts";

/** The six stats, in the order the tables keep them. */
export const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
export type Stat = (typeof STATS)[number];

/**
 * Hook names that cannot change a `damage` answer of the probe (ADR-016 section 2's flow: start the scenario, write the declared
 * state absolutely, run the engine's move step with the accuracy step passing and the hit loop replaced by `getDamage`):
 * after-hit, residual, switching, speed, accuracy, PP, healing, item-taking, trapping, boost-application and volatile-adding
 * hooks (the declared stages are written last and directly), and a switch-in's `onStart` (its stages, item and field are
 * overwritten by the declaration; the starts that do not stay overwritten are hooks of their own: `onSwitchIn`, `onUpdate`,
 * `onWeatherChange`, `onTerrainChange`, `onAnySwitchIn`, none of which is here). `onSetStatus` and its kin are inert only while no
 * status is declared on the holder (`STATUS_HOOKS`). A hook not listed is assumed to matter.
 */
export const INERT_HOOKS: ReadonlySet<string> = new Set([
  "onAfterBoost",
  "onAfterEachBoost",
  "onAfterHit",
  "onAfterMove",
  "onAfterMoveSecondary",
  "onAfterMoveSecondarySelf",
  "onAfterSubDamage",
  "onAfterUseItem",
  "onBeforeMove",
  "onAllyAfterUseItem",
  "onAllyFaint",
  "onAllyTryAddVolatile",
  "onAllyTryBoost",
  "onAnyAccuracy",
  "onAnyDamage",
  "onAnyFaint",
  "onChangeBoost",
  "onDamage",
  "onDamagingHit",
  "onDeductPP",
  "onDisableMove",
  "onDragOut",
  "onEat",
  "onEatItem",
  "onEmergencyExit",
  "onEnd",
  "onFaint",
  "onFlinch",
  "onFoeAfterBoost",
  "onFoeMaybeTrapPokemon",
  "onFoeTrapPokemon",
  "onFractionalPriority",
  "onHit",
  "onHitProtect",
  "onImmunity",
  "onMaybeTrapPokemon",
  "onModifyAccuracy",
  "onModifyPriority",
  "onModifySecondaries",
  "onModifySpe",
  "onModifyWeight",
  "onMoveFail",
  "onResidual",
  "onSourceAfterFaint",
  "onSourceDamagingHit",
  "onSourceModifyAccuracy",
  "onSourceModifySecondaries",
  "onSourceTryHeal",
  "onStart",
  "onSwitchOut",
  "onTakeItem",
  "onTrapPokemon",
  "onTryAddVolatile",
  "onTryBoost",
  "onTryHeal",
  "onUpdate",
  "onUseMoveMessage",
  "onWeather",
]);

/** Hooks that act only when a status is set: inert unless the query declares a status on the holder. */
export const STATUS_HOOKS: ReadonlySet<string> = new Set(["onSetStatus", "onAllySetStatus", "onAfterSetStatus"]);

/** Hooks of a move that act only while its own condition or secondary effect is active, or after it hit: not on the damage path. */
const MOVE_NESTED_INERT = ["condition.", "secondary.", "secondaries[", "self."];

/** Data keys of an ability that change what the simulator does outside a hook (Air Lock's weather suppression). */
const ABILITY_FLAGS = ["suppressWeather"];

/** The pinned simulator sources scanned for string literals that name an entity. */
export const SIM_SOURCES = ["sim/battle.ts", "sim/battle-actions.ts", "sim/pokemon.ts", "sim/field.ts", "sim/side.ts", "sim/dex.ts", "sim/dex-moves.ts"];

/** Narrow views of the engine's data objects. */
interface DexSpecies {
  exists: boolean;
  id: string;
  name: string;
  baseSpecies: string;
  types: string[];
  baseStats: Record<Stat, number>;
  abilities: Record<string, string>;
  isMega?: boolean;
  forme?: string;
  battleOnly?: unknown;
  requiredItem?: string;
}
interface DexMove {
  exists: boolean;
  id: string;
  name: string;
  type: string;
  category: string;
  basePower: number;
  target: string;
  priority: number;
  critRatio?: number;
  willCrit?: boolean;
  flags: Record<string, unknown>;
  [key: string]: unknown;
}
interface DexEntity {
  exists: boolean;
  id: string;
  [key: string]: unknown;
}
interface DexItem extends DexEntity {
  megaStone?: Record<string, string>;
}
interface DexNature {
  exists: boolean;
  plus?: string;
  minus?: string;
}
interface DexType {
  exists: boolean;
  name: string;
  damageTaken: Record<string, number>;
}
interface KernelDex {
  species: { get(name: string): DexSpecies };
  moves: { get(name: string): DexMove; all(): DexMove[] };
  abilities: { get(name: string): DexEntity; all(): DexEntity[] };
  items: { get(name: string): DexItem; all(): DexItem[] };
  natures: { get(name: string): DexNature };
  types: { get(name: string): DexType; names(): string[] };
  conditions: { get(name: string): { exists: boolean; id: string; effectType: string } };
}

/** A species forme as the kernel reads it. */
export interface KernelSpecies {
  id: string;
  name: string;
  base_species: string;
  types: readonly string[];
  /** HP, Atk, Def, SpA, SpD, Spe. */
  base: readonly number[];
  /** The forme's first listed ability (a Mega Evolution's ability). */
  first_ability: string;
  is_mega: boolean;
  /** A forme that exists only in battle or needs an item (a Mega forme, a Primal): never a set's own species for the kernel. */
  battle_only: boolean;
}

/** A move's fields that reach `getDamage`. */
export interface KernelMove {
  id: string;
  type: string;
  category: "Physical" | "Special" | "Status";
  base_power: number;
  target: string;
  priority: number;
  crit_ratio: number;
  will_crit: boolean;
  /** `overrideOffensiveStat` (Body Press: `def`), `overrideOffensivePokemon` (Foul Play: `target`), `overrideDefensiveStat` (Psyshock: `def`). */
  offensive_stat: string | null;
  offensive_target: boolean;
  defensive_stat: string | null;
  ignore_defensive: boolean;
  /** The probe answers these `unsupported` (`query_kind`) before any engine step: a multi-hit, fixed-damage or OHKO move. */
  outside_scope: boolean;
  /** `ignoreImmunity`: true, or the types it ignores. */
  ignore_immunity: true | readonly string[] | null;
  /** It has secondary effects (Sheer Force removes them), and the data key that boosts it under Sheer Force anyway. */
  has_secondaries: boolean;
  sheer_force_boost: boolean;
  /** The move's hooks and data keys the kernel does not implement (empty: it can answer). */
  blockers: readonly string[];
  flags: ReadonlySet<string>;
}

/** Everything the kernel reads of one format's mod. */
export interface KernelData {
  showdown_commit: string;
  format_id: string;
  mod: string;
  /** Type names in the engine's order, and `chart[defender][attacker]` = the engine's damage-taken code (0 normal, 1 super effective, 2 resisted, 3 immune). */
  types: readonly string[];
  chart: ReadonlyMap<string, ReadonlyMap<string, number>>;
  species(name: string): KernelSpecies | null;
  move(id: string): KernelMove | null;
  /** The nature's raised and lowered stats, or null for an unknown nature. */
  nature(name: string): { plus: Stat | null; minus: Stat | null } | null;
  /** The Mega forme an item makes of a species (by the species' name, then its base species), or null. */
  megaForme(item: string, species: KernelSpecies): string | null;
  /** The non-inert hooks of an ability or item, given whether the holder has a declared status (empty: inert). */
  abilityBlockers(id: string, hasStatus: boolean): readonly string[] | null;
  itemBlockers(id: string, hasStatus: boolean): readonly string[] | null;
  /** The engine's condition of an id: its effect type, or null. */
  conditionType(id: string): string | null;
  /** Every hook of an entity (`ability`, `item`, `condition`), for the handler order of an event. */
  hooks(kind: "ability" | "item" | "condition", id: string): ReadonlySet<string>;
  /** A hook's `Priority`, `Order` and `SubOrder` siblings (0, false and 0 when absent), as the engine's `resolvePriority` reads them. */
  hookOrder(kind: "ability" | "item" | "condition", id: string, hook: string): { order: number | false; priority: number; sub_order: number };
  /** A move's own hook's ordering siblings, or null when the move has no such hook. */
  moveHookOrder(id: string, hook: string): { order: number | false; priority: number; sub_order: number } | null;
  /** Whether an ability is `breakable` (a Mold Breaker attacker ignores it). */
  breakable(id: string): boolean;
  /** String literals (as engine ids) the pinned simulator sources contain. */
  sim_literals: ReadonlySet<string>;
}

export class KernelDataError extends Error {
  override name = "KernelDataError";
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

const isHook = (key: string): boolean => /^on[A-Z]/.test(key) || /Callback$/.test(key);

/** Whether a key is the ordering sibling of another key of the object (`onModifyAtkPriority` of `onModifyAtk`). */
const isSibling = (value: Record<string, unknown>, key: string): boolean => {
  const base = /^(.*?)(SubOrder|Priority|Order)$/.exec(key);
  // A number under such a name orders a hook (its hook may be absent: Taunt's `onResidualOrder` with an `onEnd` instead).
  return base !== null && ((base[1] as string) in value || typeof value[key] === "number");
};

/** Every hook of a data object, with its path (`onBasePower`, `condition.onStart`, `secondaries[0].onHit`). */
export function hooksOf(entity: Record<string, unknown>): string[] {
  const out: string[] = [];
  const walk = (value: Record<string, unknown>, prefix: string): void => {
    for (const key of Object.keys(value)) {
      // The engine's data objects hold some hook keys with no value (`onPlate` on every item): no value is no hook.
      if (isHook(key) && !isSibling(value, key) && value[key] !== undefined && value[key] !== null) out.push(prefix + key);
    }
  };
  walk(entity, "");
  for (const nested of ["condition", "secondary", "self", "fling"]) {
    const value = entity[nested];
    if (value !== null && typeof value === "object" && !Array.isArray(value)) walk(value as Record<string, unknown>, `${nested}.`);
  }
  const list = entity.secondaries;
  if (Array.isArray(list)) list.forEach((value, i) => value !== null && typeof value === "object" && walk(value as Record<string, unknown>, `secondaries[${i}].`));
  return [...new Set(out)].sort();
}

/** The hooks of an ability or item that are not inert (given a declared status on the holder). */
export function blockingHooks(hooks: readonly string[], hasStatus: boolean): string[] {
  return hooks.filter((hook) => !(INERT_HOOKS.has(hook) || (!hasStatus && STATUS_HOOKS.has(hook))));
}

/**
 * The ids the pinned simulator sources test by name: the quoted arguments of `hasAbility(...)` and `hasItem(...)`, a literal
 * compared with an `id`, `item`, `ability`, `status`, `weather` or `terrain` (`=== 'x'`, `!== 'x'`), a volatile looked up or
 * changed by name (`volatiles['x']`, `'x' in ...volatiles`, `addVolatile('x')`, `removeVolatile('x')`, `'x' in ...pseudoWeather`),
 * and the quoted members of a list tested with `.includes(...)`. As engine ids (`'Ability Shield'` is `abilityshield`).
 */
export function simLiterals(dir: string, mod: string, read: (path: string) => string = (path) => readFileSync(path, "utf8")): Set<string> {
  const out = new Set<string>();
  const quoted = (text: string): string[] => [...text.matchAll(/'([^'\n]{1,40})'|"([^"\n]{1,40})"/g)].map((m) => toId((m[1] ?? m[2]) as string));
  const patterns = [
    /has(?:Ability|Item)\(([^)]*)\)/g,
    /\b(?:id|item|ability|status|weather|terrain|moveid)\s*[!=]==?\s*('[^']*'|"[^"]*")/g,
    /('[^']*'|"[^"]*")\s*[!=]==?\s*[\w.]*\b(?:id|item|ability|status)\b/g,
    /volatiles\[\s*('[^']*'|"[^"]*")\s*\]/g,
    /('[^']*'|"[^"]*")\s+in\s+[\w.]*(?:volatiles|pseudoWeather|sideConditions|slotConditions[^\s)]*)/g,
    /(?:add|remove|get)Volatile\(\s*('[^']*'|"[^"]*")/g,
    /\[([^\]]*)\]\.includes\(/g,
  ];
  for (const file of [...SIM_SOURCES, `data/mods/${mod}/scripts.ts`]) {
    let text: string;
    try {
      text = read(join(dir, file));
    } catch {
      throw new KernelDataError("a pinned simulator source the kernel scans is missing");
    }
    for (const pattern of patterns) for (const m of text.matchAll(pattern)) for (const id of quoted(m[1] as string)) out.add(id);
  }
  out.delete("");
  return out;
}

/** Move keys the kernel reads itself or that do not reach `getDamage` (they act after the hit, or are data of other generations). */
const MOVE_SAFE_KEYS = new Set([
  "exists", "id", "name", "fullname", "num", "gen", "effectType", "sourceEffect", "placeholderFor", "isMax", "isNonstandard", "chloroblastRecoil", "kind", "desc", "shortDesc", "isNonstandard", "realMove", "baseMove",
  "type", "category", "basePower", "target", "priority", "critRatio", "willCrit", "flags", "accuracy", "pp", "noPPBoosts", "baseMoveType",
  "contestType", "isZ", "zMove", "isMax", "maxMove", "secondary", "secondaries", "self", "boosts", "drain", "recoil", "heal", "selfSwitch",
  "forceSwitch", "condition", "volatileStatus", "sideCondition", "status", "hasCrashDamage", "mindBlownRecoil", "ignoreEvasion",
  "selfBoost", "thawsTarget", "tags", "sleepUsable", "stallingMove", "tracksTarget", "breaksProtect", "hasSheerForceBoost", "struggleRecoil",
  "overrideOffensiveStat", "overrideOffensivePokemon", "overrideDefensiveStat", "ignoreDefensive", "ignoreImmunity",
  "multihit", "damage", "ohko", "damageCallback", "spreadModifier", "nonGhostTarget", "pressureTarget", "ignoreAbility", "ignoreNegativeOffensive",
  "ignorePositiveDefensive", "ignoreOffensive", "forceSTAB", "multiaccuracy", "smartTarget", "callsMove", "selfdestruct", "slotCondition",
  "pseudoWeather", "terrain", "weather", "stab", "infiltrates", "basePowerCallback",
]);
/** Move keys whose effect on `getDamage` the kernel does not implement (it abstains on such a move). */
const MOVE_BLOCKING_KEYS = ["ignoreAbility", "ignoreNegativeOffensive", "ignorePositiveDefensive", "ignoreOffensive", "forceSTAB", "multiaccuracy", "smartTarget", "callsMove", "selfdestruct", "infiltrates", "spreadModifier", "stab", "nonGhostTarget"];

function readMove(dex: KernelDex, id: string, literals: ReadonlySet<string>): KernelMove | null {
  const move = dex.moves.get(id);
  if (!move.exists || move.id !== id) return null;
  const blockers: string[] = [];
  for (const hook of hooksOf(move as unknown as Record<string, unknown>)) {
    if (MOVE_NESTED_INERT.some((prefix) => hook.startsWith(prefix))) continue;
    if (INERT_HOOKS.has(hook)) continue;
    // A multi-hit, fixed-damage or OHKO move is answered before any hook; `damageCallback` is one of those.
    if (hook === "damageCallback") continue;
    blockers.push(hook);
  }
  for (const key of Object.keys(move)) {
    if (isHook(key)) continue;
    // The engine's Move object holds every key with a default; an absent value is no data.
    if (move[key] === undefined || move[key] === null || move[key] === false) continue;
    if (!MOVE_SAFE_KEYS.has(key)) blockers.push(`key:${key}`);
    else if (MOVE_BLOCKING_KEYS.includes(key) && move[key] !== undefined && move[key] !== false) blockers.push(`key:${key}`);
  }
  if (literals.has(id)) blockers.push("sim_literal");
  const category = move.category as KernelMove["category"];
  if (category !== "Physical" && category !== "Special" && category !== "Status") blockers.push("category");
  const ignore = move.ignoreImmunity;
  const outside = category === "Status" || move.multihit !== undefined || move.damage !== undefined || move.damageCallback !== undefined || move.ohko !== undefined;
  return {
    id,
    type: move.type,
    category,
    base_power: move.basePower,
    target: move.target,
    priority: move.priority,
    crit_ratio: typeof move.critRatio === "number" ? move.critRatio : 0,
    will_crit: move.willCrit === true,
    offensive_stat: typeof move.overrideOffensiveStat === "string" ? move.overrideOffensiveStat : null,
    offensive_target: move.overrideOffensivePokemon === "target",
    defensive_stat: typeof move.overrideDefensiveStat === "string" ? move.overrideDefensiveStat : null,
    ignore_defensive: move.ignoreDefensive === true,
    outside_scope: outside,
    ignore_immunity: ignore === true ? true : ignore !== null && typeof ignore === "object" ? Object.keys(ignore as object).filter((t) => (ignore as Record<string, unknown>)[t] === true) : null,
    has_secondaries: Array.isArray(move.secondaries) && (move.secondaries as unknown[]).length > 0,
    sheer_force_boost: move.hasSheerForceBoost === true,
    blockers: blockers.sort(),
    flags: new Set(Object.keys(move.flags ?? {}).filter((f) => move.flags[f])),
  };
}

const memo = new WeakMap<Engine, Map<string, KernelData>>();

/** The kernel's data of one format, read from the pinned engine (once per engine and format). */
export function kernelData(engine: Engine, formatId: string, dir: string = DEFAULT_SHOWDOWN_DIR): KernelData {
  const table = memo.get(engine) ?? new Map<string, KernelData>();
  memo.set(engine, table);
  const hit = table.get(formatId);
  if (hit !== undefined) return hit;
  const format = engine.sim.Dex.formats.get(formatId);
  if (!format.exists) throw new KernelDataError("the format is not the engine's");
  const dex = engine.sim.Dex.mod(format.mod) as unknown as KernelDex;
  const literals = simLiterals(dir, format.mod);
  const types = dex.types.names();
  const chart = new Map<string, Map<string, number>>();
  for (const defender of types) {
    const row = new Map<string, number>();
    const data = dex.types.get(defender);
    for (const attacker of types) row.set(attacker, data.damageTaken[attacker] ?? 0);
    chart.set(defender, row);
  }
  const entityOf = (kind: "ability" | "item" | "condition", id: string): DexEntity =>
    (kind === "ability" ? dex.abilities.get(id) : kind === "item" ? dex.items.get(id) : dex.conditions.get(id)) as unknown as DexEntity;
  const species = new Map<string, KernelSpecies | null>();
  const moves = new Map<string, KernelMove | null>();
  const blockersOf = (kind: "ability" | "item", id: string, hasStatus: boolean): string[] | null => {
    const entity = kind === "ability" ? dex.abilities.get(id) : dex.items.get(id);
    if (!entity.exists || entity.id !== id) return null;
    const out = blockingHooks(hooksOf(entity), hasStatus);
    if (kind === "ability") for (const flag of ABILITY_FLAGS) if (entity[flag] !== undefined && entity[flag] !== false) out.push(`key:${flag}`);
    if (literals.has(id)) out.push("sim_literal");
    return out.sort();
  };
  const data: KernelData = {
    showdown_commit: engine.commit,
    format_id: formatId,
    mod: format.mod,
    types,
    chart,
    sim_literals: literals,
    species(name) {
      const id = toId(name);
      if (!species.has(id)) {
        const s = dex.species.get(name);
        species.set(
          id,
          s.exists
            ? {
                id: s.id,
                name: s.name,
                base_species: s.baseSpecies,
                types: [...s.types],
                base: STATS.map((stat) => s.baseStats[stat]),
                first_ability: toId(s.abilities["0"] ?? ""),
                is_mega: s.isMega === true,
                battle_only: s.battleOnly !== undefined || s.requiredItem !== undefined,
              }
            : null,
        );
      }
      return species.get(id) ?? null;
    },
    move(id) {
      if (!moves.has(id)) moves.set(id, readMove(dex, id, literals));
      return moves.get(id) ?? null;
    },
    nature(name) {
      const nature = dex.natures.get(name);
      if (!nature.exists) return null;
      const stat = (value: string | undefined): Stat | null => ((STATS as readonly string[]).includes(value ?? "") ? (value as Stat) : null);
      return { plus: stat(nature.plus), minus: stat(nature.minus) };
    },
    megaForme(item, s) {
      const stone = dex.items.get(item).megaStone;
      return stone?.[s.name] ?? stone?.[s.base_species] ?? null;
    },
    abilityBlockers: (id, hasStatus) => blockersOf("ability", id, hasStatus),
    itemBlockers: (id, hasStatus) => blockersOf("item", id, hasStatus),
    conditionType(id) {
      const condition = dex.conditions.get(id);
      return condition.exists && condition.id === id ? condition.effectType : null;
    },
    hooks(kind, id) {
      return new Set(hooksOf(entityOf(kind, id)));
    },
    hookOrder(kind, id, hook) {
      const e = entityOf(kind, id);
      const order = e[`${hook}Order`];
      const priority = e[`${hook}Priority`];
      const sub = e[`${hook}SubOrder`];
      return { order: typeof order === "number" && order !== 0 ? order : false, priority: typeof priority === "number" ? priority : 0, sub_order: typeof sub === "number" ? sub : 0 };
    },
    moveHookOrder(id, hook) {
      const m = dex.moves.get(id) as unknown as DexEntity;
      if (m[hook] === undefined || m[hook] === null) return null;
      const order = m[`${hook}Order`];
      const priority = m[`${hook}Priority`];
      const sub = m[`${hook}SubOrder`];
      return { order: typeof order === "number" && order !== 0 ? order : false, priority: typeof priority === "number" ? priority : 0, sub_order: typeof sub === "number" ? sub : 0 };
    },
    breakable(id) {
      const flags = dex.abilities.get(id).flags as Record<string, unknown> | undefined;
      return flags?.breakable !== undefined && flags.breakable !== 0 && flags.breakable !== false;
    },
  };
  table.set(formatId, data);
  return data;
}
