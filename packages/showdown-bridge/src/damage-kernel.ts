// The damage kernel (ADR-030 step 3, #1457): a clean-room integer implementation of the Champions damage computation, the CPU
// reference. It answers a `damage` mechanics query with the bytes the pinned probe gives (16 no-crit rolls, 16 crit rolls, the
// crit chance, the defender's HP and the KO counts), for TRAINING games only and only for content it was admitted on by
// byte-equality with the probe (the differential harness, `packages/differential`). It never guesses: a query that names an
// entity or a state it does not implement is ABSTAINED on (`KernelAbstain`, naming the entity), and the engine probe answers it.
//
// Clean room. Written from the mechanics (the generation 9 damage formula as the pinned engine runs it: base damage, then the
// spread, weather, crit, roll, same-type, type and burn steps, then the final modifiers, each with its 4096-based rounding; the
// event chains with their handler order), from the pinned engine's DATA (species, moves, natures, the type chart, Mega Stones,
// hook priorities: `damage-kernel-data.ts`), from the reviewed effects record (`damage-kernel-effects.ts`: which ability or item
// has which effect, with the engine's parameters) and from the project's hand-worked fixtures. The engine's MIT sources were read
// for behaviour; no AGPL code (the Showdown client, a damage calculator) was used, and `@smogon/calc` is never imported.
//
// Integers only: every value is an integer and every division an exact integer division (`div`). A batch is decoded into
// struct-of-arrays columns (`Int32Array`) and the 32 rolls of every query are computed in one loop over them (`computeBatch`).
//
// What KERNEL_FORMAT 1 covers, abstaining on everything else:
// - attacker and defender: the set's species forme (a Mega or battle-only forme as a set's own species is abstained on), or the
//   Mega forme its stone makes when the query declares a Mega Evolution (its first ability then); Champions Stat Point stats;
//   absolute stages; declared HP (at most the maximum); a declared status the engine knows;
// - abilities and items with no hook the probe's damage path can reach (`INERT_HOOKS`) and not tested by name in the simulator,
//   and those the effects record implements;
// - a Physical or Special move with a fixed base power, a single (`normal`, `any`, `adjacentFoe`) or spread (`allAdjacentFoes`,
//   `allAdjacent`) target and no hook of its own on the damage path; `willCrit`, `critRatio`, the stat overrides (Body Press,
//   Foul Play, Psyshock), `ignoreDefensive` and `ignoreImmunity` read from its data; a status, multi-hit, fixed-damage or OHKO
//   move is answered `unsupported` (`query_kind`), as the probe answers it;
// - type immunity (Ground by groundedness, the chart's immune cells), absorbing and flag immunities, Psychic Terrain against a
//   priority move: every roll 0;
// - rain, sun, sand and snow (unless an ability in play suppresses weather), the four terrains, Reflect, Light Screen and Aurora
//   Veil on the defender's side, Trick Room and Tailwind (no damage effect), burn;
// - a declared history whose volatiles are all inert and that declares no pending slot condition.
// Event chains: the handlers an event finds are ordered as the engine orders them (order, priority, then speed, then sub-order);
// where two Pokemon's handlers tie on order and priority (their speeds decide, or a draw), or two side or field conditions tie,
// the kernel abstains.
import type { PokemonSet } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { koBlocked, koCounts } from "./mechanics-ko.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { Outcome } from "./mechanics-result.ts";
import { type KernelData, type KernelSpecies, STATS, type Stat } from "./damage-kernel-data.ts";
import type { KernelEffect, KernelEffects } from "./damage-kernel-effects.ts";

/** Bumped whenever the kernel computes anything differently, or covers more (its admission starts again). */
export const KERNEL_FORMAT = 1;

type DamageQuery = Extract<MechanicsQuery["query"], { kind: "damage" }>;
type Combatant = DamageQuery["attacker"];
type DamageAnswer = Extract<Extract<Outcome, { kind: "computed" }>["answer"], { kind: "damage" }>;

/** The kernel cannot answer a query: the entity or state it does not implement (the engine probe answers it). */
export class KernelAbstain extends Error {
  override name = "KernelAbstain";
  /** `<kind>:<id>` of what it does not implement (`ability:intimidate`, `move:knockoff`, `state:handler_tie`, ...). */
  readonly entity: string;
  /** A fixed word for why (a hook name, `sim_literal`, `not_implemented`, ...). */
  readonly why: string;
  constructor(entity: string, why: string) {
    super(`the damage kernel does not cover ${entity} (${why})`);
    this.entity = entity;
    this.why = why;
  }
}

/** The weathers and terrains the kernel implements, by the engine's ids, and the terrain boost's type. */
const WEATHERS = new Set(["raindance", "sunnyday", "sandstorm", "snowscape"]);
const TERRAINS = new Set(["electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"]);
const TERRAIN_TYPE: Record<string, string> = { electricterrain: "Electric", grassyterrain: "Grass", psychicterrain: "Psychic" };
/** The moves Grassy Terrain halves into a grounded defender (the pinned terrain handler's own list). */
const GRASSY_WEAKENED = new Set(["earthquake", "bulldoze", "magnitude"]);
const SINGLE_TARGETS = new Set(["normal", "any", "adjacentFoe", "randomNormal"]);
const SPREAD_TARGETS = new Set(["allAdjacentFoes", "allAdjacent"]);
/** Side conditions by the query's names and the engine's ids. */
const SIDE_IDS: [keyof DamageQuery["state"]["sides"][number], string][] = [
  ["aurora_veil", "auroraveil"],
  ["light_screen", "lightscreen"],
  ["reflect", "reflect"],
  ["tailwind", "tailwind"],
];

/** The engine's 4096-based modifiers. */
const M_ONE = 4096;
const M_SPREAD = 3072;
const M_HALF = 2048;
const M_ONE_AND_HALF = 6144;
const M_TERRAIN = 5325;
const M_SCREEN = 2732; // two active Pokemon a side
const CRIT_MULT = [0, 24, 8, 2, 1];
/** The sub-order the engine gives a handler with none of its own, by holder kind (`resolvePriority`). */
const SUB_ORDER = { volatile: 2, side: 4, field: 5, ability: 7, item: 8, status: 0 } as const;
/** The events whose handler order the kernel checks: those of `getDamage` and of the move step before it. */
const ORDERED_EVENTS = ["BasePower", "ModifyAtk", "ModifySpA", "ModifyDef", "ModifySpD", "ModifyDamage", "ModifySTAB", "CriticalHit", "ModifyCritRatio", "Effectiveness", "ModifyMove", "ModifyType", "TryMove", "PrepareHit", "NegateImmunity", "ModifyBoost", "AfterMoveSecondarySelf"];
/** Events whose target is the defender (the others are the attacker's). */
const DEFENDER_EVENTS = new Set(["ModifyDef", "ModifySpD", "CriticalHit", "Effectiveness", "NegateImmunity"]);

/** Integer division of non-negative integers (exact: `a - a % b` is a multiple of `b`). */
const div = (a: number, b: number): number => (a - (a % b)) / b;
/** The engine's `modify(value, m / 4096)`. */
const modify = (value: number, m: number): number => div(value * m + 2047, 4096);
/** The engine's `chainModify`: the running modifier times the next, rounded half up in 1/4096. */
const chain = (m: number, next: number): number => (m * next + 2048) >> 12;

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** One combatant as the kernel computes with it. */
interface Fighter {
  role: "attacker" | "defender";
  species: KernelSpecies;
  types: readonly string[];
  ability: string;
  item: string | null;
  status: string | null;
  volatiles: readonly string[];
  /** HP, Atk, Def, SpA, SpD, Spe as the engine stores them (before stages). */
  stats: number[];
  boosts: Combatant["boosts"];
  hp: number;
  ability_effect: KernelEffect | null;
  item_effect: KernelEffect | null;
}

/** The Champions stat: base + Stat Points + 75 for HP, base + Stat Points + 20 then the nature for the others. */
export function championsStat(base: number, points: number, stat: Stat, nature: { plus: Stat | null; minus: Stat | null }): number {
  if (stat === "hp") return base + points + 75;
  const value = base + points + 20;
  // The engine's 16-bit truncation of `stat * 110` never bites at these sizes.
  if (nature.plus === stat) return div(value * 110, 100);
  if (nature.minus === stat) return div(value * 90, 100);
  return value;
}

/** A stat after its stage (the engine's boost table: `floor(stat * (2 + b) / 2)` up, `floor(stat * 2 / (2 - b))` down). */
export const staged = (stat: number, boost: number): number => (boost >= 0 ? div(stat * (2 + boost), 2) : div(stat * 2, 2 - boost));

/** The kernel's inputs that the probe's envelope does not give. */
export interface KernelContext {
  data: KernelData;
  effects: KernelEffects;
  ko_blockers: MechanicsConfig["ko_blockers"];
  /** The scenario format's `adjustLevel`, or null when the set's own level stands. */
  level: number | null;
}

/** Every value of a history fact: a known value, or each end of an unknown one. */
const valuesOf = <T>(fact: unknown): T[] => {
  if (fact === undefined || fact === null) return [];
  const f = fact as { kind: string; value?: T; ends?: T[] };
  return f.kind === "known" ? [f.value as T] : (f.ends ?? []);
};

/** Hooks that can refuse a volatile the history declares (the engine's TryAddVolatile family and its status immunity). */
const VOLATILE_REFUSERS = ["onTryAddVolatile", "onAllyTryAddVolatile", "onImmunity"];

function declaredVolatiles(ctx: KernelContext, combatant: Combatant): string[] {
  const ids = new Set<string>();
  for (const list of valuesOf<{ id: string }[]>((combatant.history as { volatiles?: unknown } | undefined)?.volatiles)) for (const v of list) ids.add(v.id);
  for (const id of ids) {
    // Only a reviewed volatile: its start cannot be refused and it changes no damage answer.
    if (ctx.effects.volatiles[id] === undefined) throw new KernelAbstain(`volatile:${id}`, "not_implemented");
    const blockers = ctx.data.volatileBlockers(id);
    if (blockers === null) throw new KernelAbstain(`volatile:${id}`, "unknown");
    if (blockers.some((b) => b !== "sim_literal" && b !== "onStart")) throw new KernelAbstain(`volatile:${id}`, blockers[0] as string);
  }
  return [...ids].sort();
}

/** A declared volatile can be refused by an ability, item or terrain that blocks volatiles: abstain when one is present. */
function checkVolatileRefusers(ctx: KernelContext, fighters: readonly Fighter[], field: readonly string[]): void {
  if (fighters.every((f) => f.volatiles.length === 0)) return;
  const holders: ["ability" | "item" | "condition", string][] = [...fighters.flatMap((f) => [["ability", f.ability], ...(f.item === null ? [] : [["item", f.item]])] as ["ability" | "item", string][]), ...field.map((id) => ["condition", id] as ["condition", string])];
  for (const [kind, id] of holders) {
    const hooks = ctx.data.hooks(kind, id);
    if (VOLATILE_REFUSERS.some((hook) => hooks.has(hook))) throw new KernelAbstain(`${kind}:${id}`, "volatile_refuser");
  }
}

function entityEffect(ctx: KernelContext, kind: "ability" | "item", id: string, hasStatus: boolean): KernelEffect | null {
  const blockers = kind === "ability" ? ctx.data.abilityBlockers(id, hasStatus) : ctx.data.itemBlockers(id, hasStatus);
  if (blockers === null) throw new KernelAbstain(`${kind}:${id}`, "unknown");
  const effect = (kind === "ability" ? ctx.effects.abilities[id] : ctx.effects.items[id]) ?? null;
  if (blockers.length > 0 && effect === null) throw new KernelAbstain(`${kind}:${id}`, blockers[0] as string);
  return effect;
}

function fighter(ctx: KernelContext, role: Fighter["role"], set: PokemonSet, combatant: Combatant): Fighter {
  const { data } = ctx;
  const own = data.species(set.species);
  if (own === null) throw new KernelAbstain(`species:${toId(set.species)}`, "unknown");
  if (own.is_mega || own.battle_only) throw new KernelAbstain(`species:${own.id}`, "battle_only");
  if (ctx.level !== null && set.level !== ctx.level) throw new KernelAbstain("state:level", "not_implemented");
  const item = set.item === "" ? null : toId(set.item);
  let species = own;
  let ability = toId(set.ability);
  if (combatant.mega) {
    const forme = item === null ? null : data.megaForme(item, own);
    const mega = forme === null ? null : data.species(forme);
    if (mega === null || mega.id === own.id) throw new KernelAbstain(`species:${own.id}`, "mega");
    species = mega;
    ability = mega.first_ability;
  }
  let status: string | null = null;
  if (combatant.status !== null) {
    if (data.conditionType(combatant.status) !== "Status") throw new KernelAbstain(`status:${combatant.status}`, "not_implemented");
    status = combatant.status;
  }
  const ability_effect = entityEffect(ctx, "ability", ability, status !== null);
  const item_effect = item === null ? null : entityEffect(ctx, "item", item, status !== null);
  const volatiles = declaredVolatiles(ctx, combatant);
  const nature = data.nature(set.nature);
  if (nature === null) throw new KernelAbstain("state:nature", "unknown");
  // A Mega Evolution keeps the HP it had: the engine sets `maxhp` once, from the forme the Pokemon started in.
  const stats = STATS.map((stat, i) => championsStat((stat === "hp" ? own : species).base[i] as number, set.evs[stat] ?? 0, stat, nature));
  const maxhp = stats[0] as number;
  if (combatant.hp !== null && combatant.hp > maxhp) throw new KernelAbstain("state:hp", "above_maximum");
  return { role, species, types: species.types, ability, item, status, volatiles, stats, boosts: combatant.boosts, hp: combatant.hp ?? maxhp, ability_effect, item_effect };
}

/** A pending slot condition in the history: abstain (no slot condition is implemented). */
function checkSlots(query: DamageQuery): void {
  for (const side of query.history?.sides ?? []) {
    if (valuesOf<unknown[]>(side.pending_slots).some((list) => list.length > 0)) throw new KernelAbstain("state:history_slot", "not_implemented");
  }
}

/** One handler an event finds: who holds it and its ordering keys. */
interface Handler {
  holder: "attacker" | "defender" | "field" | "side0" | "side1";
  kind: "ability" | "item" | "condition";
  id: string;
  order: number;
  priority: number;
  pokemon: boolean;
  sub_order: number;
}

/**
 * The handlers an event finds, in the engine's order. `target` is the Pokemon the event is about, `source` the other one. A tie
 * the kernel cannot order is an abstention: two Pokemon's handlers on equal order and priority (the engine compares their speeds,
 * a draw on equal ones), or two side or field conditions on equal keys (a draw).
 */
function handlers(ctx: KernelContext, event: string, target: Fighter, source: Fighter, sides: readonly string[][], field: readonly string[]): Handler[] {
  const found: Handler[] = [];
  const add = (holder: Handler["holder"], kind: Handler["kind"], id: string, names: string[], sub: number, pokemon: boolean): void => {
    const hooks = ctx.data.hooks(kind, id);
    for (const name of names) {
      if (!hooks.has(name)) continue;
      const o = ctx.data.hookOrder(kind, id, name);
      found.push({ holder, kind, id, order: o.order === false ? 4294967296 : o.order, priority: o.priority, pokemon, sub_order: o.sub_order || sub });
    }
  };
  for (const [p, names] of [
    [target, [`on${event}`, `onAlly${event}`, `onAny${event}`]],
    [source, [`onSource${event}`, `onFoe${event}`, `onAny${event}`]],
  ] as [Fighter, string[]][]) {
    add(p.role, "ability", p.ability, names, SUB_ORDER.ability, true);
    if (p.item !== null) add(p.role, "item", p.item, names, SUB_ORDER.item, true);
    if (p.status !== null) add(p.role, "condition", p.status, names, SUB_ORDER.status, true);
    for (const v of p.volatiles) add(p.role, "condition", v, names, SUB_ORDER.volatile, true);
  }
  const targetSide = target.role === "attacker" ? 0 : 1;
  sides.forEach((ids, side) => {
    for (const id of ids) add(side === 0 ? "side0" : "side1", "condition", id, side === targetSide ? [`on${event}`, `onAny${event}`] : [`onFoe${event}`, `onAny${event}`], SUB_ORDER.side, false);
  });
  for (const id of field) add("field", "condition", id, [`on${event}`], SUB_ORDER.field, false);
  for (let i = 0; i < found.length; i++) {
    for (let j = i + 1; j < found.length; j++) {
      const a = found[i] as Handler;
      const b = found[j] as Handler;
      if (a.holder === b.holder) continue;
      const keysTie = a.order === b.order && a.priority === b.priority;
      if (keysTie && ((a.pokemon && b.pokemon) || (!a.pokemon && !b.pokemon && a.sub_order === b.sub_order))) throw new KernelAbstain("state:handler_tie", event);
    }
  }
  // order (ascending), priority (descending), speed (a Pokemon's before a condition's), sub-order (ascending).
  const key = (h: Handler): number[] => [h.order, -h.priority, h.pokemon ? 0 : 1, h.sub_order];
  return found.sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    for (let k = 0; k < ka.length; k++) if (ka[k] !== kb[k]) return (ka[k] as number) - (kb[k] as number);
    return 0;
  });
}

/** One query, decoded: what the roll loop needs. */
interface Decoded {
  /** Base damage before `modifyDamage`: no crit, crit. */
  base: [number, number];
  spread: boolean;
  /** WeatherModifyDamage as a 4096-based modifier. */
  weather: number;
  /** The same-type modifier (4096: none). */
  stab: number;
  type_mod: number;
  burn: boolean;
  /** The ModifyDamage chain: no crit, crit. */
  final: [number, number];
  /** The crit run is a crit (false when the defender is never hit critically: its rolls are the no-crit ones). */
  crit_applies: boolean;
  crit_chance: DamageAnswer["crit_chance"];
  defender: Fighter;
}

export type Prepared = { kind: "rolls"; decoded: Decoded } | { kind: "outcome"; outcome: Outcome };

function assemble(ctx: KernelContext, defender: Fighter, noCrit: number[], crit: number[], critChance: DamageAnswer["crit_chance"]): Outcome {
  return {
    kind: "computed",
    answer: {
      kind: "damage",
      no_crit: noCrit as DamageAnswer["no_crit"],
      crit: crit as DamageAnswer["crit"],
      crit_chance: critChance,
      hp_current: defender.hp,
      hp_max: defender.stats[0] as number,
      ko: koCounts(noCrit, crit, defender.hp, koBlocked(ctx.ko_blockers, { item: defender.item ?? "", ability: defender.ability })),
    },
  };
}

const blocked = (ctx: KernelContext, defender: Fighter): Prepared => {
  const zeros = new Array<number>(16).fill(0);
  return { kind: "outcome", outcome: assemble(ctx, defender, zeros, [...zeros], { kind: "never" }) };
};

/** The effect of a handler's entity, or null for a status or volatile (none is implemented). */
const effectOf = (f: Fighter, h: Handler): KernelEffect | null => (h.kind === "ability" ? f.ability_effect : h.kind === "item" ? f.item_effect : null);

/** STUB (tests-first): decodes every query to zero inputs, so every covered answer differs from the probe. */
export function prepareQuery(ctx: KernelContext, _query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): Prepared {
  const defender = fighter(ctx, "defender", defenderSet, _query.defender);
  void fighter(ctx, "attacker", attackerSet, _query.attacker);
  void handlers;
  void checkSlots;
  void checkVolatileRefusers;
  void blocked;
  void effectOf;
  void [WEATHERS, TERRAINS, TERRAIN_TYPE, GRASSY_WEAKENED, SINGLE_TARGETS, SPREAD_TARGETS, SIDE_IDS, M_ONE_AND_HALF, M_TERRAIN, M_SCREEN, CRIT_MULT, ORDERED_EVENTS, DEFENDER_EVENTS, chain];
  return { kind: "rolls", decoded: { base: [0, 0], spread: false, weather: M_ONE, stab: M_ONE, type_mod: 0, burn: false, final: [M_ONE, M_ONE], crit_applies: true, crit_chance: { kind: "never" }, defender } };
}

/** The columns of a decoded batch: one row per query, the numbers `modifyDamage` reads. */
interface Columns {
  n: number;
  base: Int32Array; // [2n]: no crit, crit
  final: Int32Array; // [2n]: no crit, crit
  flags: Int32Array; // bit 0 spread, 1 burn, 2 the crit run is a crit
  weather: Int32Array;
  stab: Int32Array;
  type_mod: Int32Array;
}

function columnsOf(rows: readonly Decoded[]): Columns {
  const n = rows.length;
  const c: Columns = { n, base: new Int32Array(2 * n), final: new Int32Array(2 * n), flags: new Int32Array(n), weather: new Int32Array(n), stab: new Int32Array(n), type_mod: new Int32Array(n) };
  rows.forEach((r, i) => {
    c.base[2 * i] = r.base[0];
    c.base[2 * i + 1] = r.base[1];
    c.final[2 * i] = r.final[0];
    c.final[2 * i + 1] = r.final[1];
    c.flags[i] = (r.spread ? 1 : 0) | (r.burn ? 2 : 0) | (r.crit_applies ? 4 : 0);
    c.weather[i] = r.weather;
    c.stab[i] = r.stab;
    c.type_mod[i] = r.type_mod;
  });
  return c;
}

/** `modifyDamage` for every query, crit run and roll of a batch: `out[(2 * i + crit) * 16 + roll]`. */
export function computeBatch(c: Columns): Int32Array {
  const out = new Int32Array(c.n * 32);
  if (out.length >= 0) return out; // STUB (tests-first): every roll 0
  for (let i = 0; i < c.n; i++) {
    const flags = c.flags[i] as number;
    const typeMod = c.type_mod[i] as number;
    for (let crit = 0; crit < 2; crit++) {
      const isCrit = crit === 1 && (flags & 4) !== 0;
      let pre = (c.base[2 * i + crit] as number) + 2;
      if (flags & 1) pre = modify(pre, M_SPREAD);
      pre = modify(pre, c.weather[i] as number);
      if (isCrit) pre = div(pre * 3, 2);
      const final = c.final[2 * i + crit] as number;
      for (let roll = 0; roll < 16; roll++) {
        let d = div(pre * (100 - roll), 100);
        d = modify(d, c.stab[i] as number);
        if (typeMod > 0) d = d * (1 << typeMod);
        else for (let k = 0; k > typeMod; k--) d = div(d, 2);
        if (flags & 2) d = modify(d, M_HALF);
        d = modify(d, final);
        if (d === 0) d = 1;
        out[(2 * i + crit) * 16 + roll] = d % 65536;
      }
    }
  }
  return out;
}

/** The outcomes of decoded rows, in order. */
export function rollOutcomes(ctx: KernelContext, rows: readonly Decoded[]): Outcome[] {
  const out = computeBatch(columnsOf(rows));
  return rows.map((r, i) => assemble(ctx, r.defender, Array.from(out.subarray(32 * i, 32 * i + 16)), Array.from(out.subarray(32 * i + 16, 32 * i + 32)), r.crit_chance));
}
