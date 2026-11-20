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
// What KERNEL_FORMAT 4 covers, abstaining on everything else. Format 1's fresh corpus found two classes it got wrong (an
// after-move hook that draws; a switch-in terrain refusing a declared status), now abstained; format 3 takes every condition's
// ids and values from the effects record, models a weather's direct stat modify, and makes `onUpdate` blocking; format 4 reads
// the pinch, Guts and resist-berry modifiers and the pinch HP threshold from the record too (the same answers as format 3):
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
// - the weathers, terrains and side and field conditions the effects record lists (`conditions`: weather damage modifiers by type,
//   a weather's direct stat modify, terrain boosts, weakened moves and halved types, the priority block, screens), unless an
//   ability in play suppresses weather; burn;
// - a declared history with no volatile, no pending slot condition and facts the engine accepts.
// Event chains: the handlers an event finds are ordered as the engine orders them (order, priority, then speed, then sub-order);
// where two Pokemon's handlers tie on order and priority (their speeds decide, or a draw), or two side or field conditions tie,
// the kernel abstains.
import type { PokemonSet } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { koBlocked, koCounts } from "./mechanics-ko.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { Outcome } from "./mechanics-result.ts";
import { type KernelData, type KernelSpecies, STATS, type Stat } from "./damage-kernel-data.ts";
import type { KernelCondition, KernelEffect, KernelEffects } from "./damage-kernel-effects.ts";

/** Bumped whenever the kernel computes anything differently, or covers more (its admission starts again). */
export const KERNEL_FORMAT = 4;

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

const SINGLE_TARGETS = new Set(["normal", "any", "adjacentFoe", "randomNormal"]);
const SPREAD_TARGETS = new Set(["allAdjacentFoes", "allAdjacent"]);
/** Side conditions by the query's names and the engine's ids. */
const SIDE_IDS: [keyof DamageQuery["state"]["sides"][number], string][] = [
  ["aurora_veil", "auroraveil"],
  ["light_screen", "lightscreen"],
  ["reflect", "reflect"],
  ["tailwind", "tailwind"],
];

/**
 * The engine's 4096-based modifiers of the damage FORMULA itself (`getDamage` and `modifyDamage`: the spread reduction, the
 * same-type bonus, the burn halving, the critical-hit multiplier and crit ratios). They are the formula, not the rule of any one
 * ability, item, move or condition; every per-entity value (a pinch ability's modifier and HP threshold, Guts, a resist berry, a
 * weather, terrain or screen) is read from the reviewed effects record (owner rule R7).
 */
const M_ONE = 4096;
const M_SPREAD = 3072;
const M_HALF = 2048;
const M_ONE_AND_HALF = 6144;
const CRIT_MULT = [0, 24, 8, 2, 1];
/** The sub-order the engine gives a handler with none of its own, by holder kind (`resolvePriority`). */
const SUB_ORDER = { side: 4, field: 5, ability: 7, item: 8, status: 0 } as const;
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

/** One handler of a stat event: a hook that modifies the stat at once (`this.modify`: a weather's), or one that chains (`chainModify`). */
export type StatHandler = { direct: number } | { chain: number };

/**
 * A stat event as the engine runs it: each direct hook modifies the value in handler order, the chained modifiers accumulate and
 * modify the result once at the end (the engine's `runEvent` and `finalModify`), so the two do not commute.
 */
export function applyStatHandlers(value: number, list: readonly StatHandler[]): number {
  let v = value;
  let m = M_ONE;
  for (const h of list) {
    if ("direct" in h) v = modify(v, h.direct);
    else m = chain(m, h.chain);
  }
  return m === M_ONE ? v : modify(v, m);
}

/** One combatant as the kernel computes with it. */
interface Fighter {
  role: "attacker" | "defender";
  species: KernelSpecies;
  types: readonly string[];
  ability: string;
  item: string | null;
  status: string | null;
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

/** A declared volatile: abstain (none is implemented; the engine checks each against its own start and duration rules). */
function checkVolatiles(combatant: Combatant): void {
  for (const list of valuesOf<{ id: string }[]>((combatant.history as { volatiles?: unknown } | undefined)?.volatiles)) {
    const first = list[0];
    if (first !== undefined) throw new KernelAbstain(`volatile:${first.id}`, "not_implemented");
  }
}

function entityEffect(ctx: KernelContext, kind: "ability" | "item", id: string, hasStatus: boolean): KernelEffect | null {
  const blockers = kind === "ability" ? ctx.data.abilityBlockers(id, hasStatus) : ctx.data.itemBlockers(id, hasStatus);
  if (blockers === null) throw new KernelAbstain(`${kind}:${id}`, "unknown");
  const effect = (kind === "ability" ? ctx.effects.abilities[id] : ctx.effects.items[id]) ?? null;
  if (blockers.length > 0 && effect === null) throw new KernelAbstain(`${kind}:${id}`, blockers[0] as string);
  return effect;
}

/** A combatant before the other is known: its forme, ability and status. */
interface Base {
  role: Fighter["role"];
  own: KernelSpecies;
  species: KernelSpecies;
  ability: string;
  item: string | null;
  status: string | null;
  set: PokemonSet;
  combatant: Combatant;
  ability_effect: KernelEffect | null;
}

function fighter(ctx: KernelContext, role: Fighter["role"], set: PokemonSet, combatant: Combatant): Base {
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
  return { role, own, species, ability, item, status, set, combatant, ability_effect };
}

/** The rest of a combatant once both are known: a declared status is set before the field, so a switch-in effect of either can refuse it. */
function finish(ctx: KernelContext, base: Base, other: Base): Fighter {
  const { data } = ctx;
  const { role, species, ability, item, status, set, combatant, own, ability_effect } = base;
  // The engine starts the scenario (every switch-in's start effect, a Surge's terrain among them) before the declared status, so a
  // status can be refused by a condition an ability started (Electric and Misty Terrain refuse statuses): abstain then.
  if (status !== null) {
    for (const id of [ability, other.ability]) {
      const starts = data.abilityStarts(id);
      const refuses = starts === null ? data.hooks("ability", id).has("onStart") : starts.some((c) => data.hooks("condition", c).has("onSetStatus"));
      if (refuses) throw new KernelAbstain(`ability:${id}`, "status_start");
    }
  }
  const item_effect = item === null ? null : entityEffect(ctx, "item", item, status !== null);
  checkVolatiles(combatant);
  const nature = data.nature(set.nature);
  if (nature === null) throw new KernelAbstain("state:nature", "unknown");
  // A Mega Evolution keeps the HP it had: the engine sets `maxhp` once, from the forme the Pokemon started in.
  const stats = STATS.map((stat, i) => championsStat((stat === "hp" ? own : species).base[i] as number, set.evs[stat] ?? 0, stat, nature));
  const maxhp = stats[0] as number;
  if (combatant.hp !== null && combatant.hp > maxhp) throw new KernelAbstain("state:hp", "above_maximum");
  return { role, species, types: species.types, ability, item, status, stats, boosts: combatant.boosts, hp: combatant.hp ?? maxhp, ability_effect, item_effect };
}

/**
 * The declared history the kernel can answer with: no pending slot condition, no attacker list or stall counter (the engine
 * checks both against its own state), and facts the engine would accept as declared (a used move of the Pokemon's own set, an
 * engine move or item). The facts left change no answer the kernel computes; a declaration the engine would refuse abstains,
 * since the probe throws there.
 */
function checkHistory(ctx: KernelContext, query: DamageQuery, sets: readonly PokemonSet[]): void {
  const fail = (why: string): never => {
    throw new KernelAbstain("state:history", why);
  };
  const isMove = (id: string | null): boolean => id === null || ctx.data.move(id) !== null;
  for (const side of query.history?.sides ?? []) {
    if (valuesOf<unknown[]>(side.pending_slots).some((list) => list.length > 0)) throw new KernelAbstain("state:history_slot", "not_implemented");
  }
  if (!valuesOf<string | null>(query.history?.field_last_move).every(isMove)) fail("field_last_move");
  [query.attacker, query.defender].forEach((combatant, i) => {
    const facts = (combatant.history ?? {}) as Record<string, unknown>;
    const own = new Set((sets[i] as PokemonSet).moves.map(toId));
    if (valuesOf<string[]>(facts.moves_used).some((list) => list.some((id) => !own.has(id)))) fail("moves_used");
    if (!valuesOf<string | null>(facts.last_move).every(isMove)) fail("last_move");
    if (!valuesOf<string | null>(facts.last_item).every((id) => id === null || ctx.data.itemBlockers(id, false) !== null)) fail("last_item");
    if (valuesOf<number>(facts.stall_counter).some((n) => n !== 0)) fail("stall_counter");
    if (valuesOf<unknown[]>(facts.attacked_by).some((list) => list.length > 0)) fail("attacked_by");
  });
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
      const keysTie = a.order === b.order && a.priority === b.priority;
      // Two Pokemon (their speeds decide, or the engine draws); two conditions on equal sub-orders, the same side's included (a draw).
      const pokemonTie = a.pokemon && b.pokemon && a.holder !== b.holder;
      const conditionTie = !a.pokemon && !b.pokemon && a.sub_order === b.sub_order;
      if (keysTie && (pokemonTie || conditionTie)) throw new KernelAbstain("state:handler_tie", event);
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

/** Decode one query (the sets are the probe envelope's validated ones). Throws `KernelAbstain` for what the kernel does not cover. */
export function prepareQuery(ctx: KernelContext, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): Prepared {
  const { data } = ctx;
  checkHistory(ctx, query, [attackerSet, defenderSet]);
  const { state } = query;
  // Every declared condition must be one the reviewed record implements, of the right kind.
  const conditionOf = (id: string, kinds: KernelCondition["kind"][], what: string): KernelCondition => {
    const c = ctx.effects.conditions[id];
    if (c === undefined || !kinds.includes(c.kind)) throw new KernelAbstain(`${what}:${id}`, "not_implemented");
    return c;
  };
  if (state.weather !== null) conditionOf(state.weather, ["weather_damage", "weather_stat"], "weather");
  if (state.terrain !== null) conditionOf(state.terrain, ["terrain"], "terrain");
  if (state.trick_room) conditionOf("trickroom", ["no_damage_effect"], "field");
  const a = fighter(ctx, "attacker", attackerSet, query.attacker);
  const d = fighter(ctx, "defender", defenderSet, query.defender);
  const attacker = finish(ctx, a, d);
  const defender = finish(ctx, d, a);
  const move = data.move(query.move);
  if (move === null) throw new KernelAbstain(`move:${query.move}`, "unknown");
  // Outside the probe's v1 scope: answered before any hook of the move runs.
  if (move.outside_scope) return { kind: "outcome", outcome: { kind: "unsupported", reason: "query_kind", position: null } };
  const moveEffect = ctx.effects.moves[move.id] ?? null;
  if (move.blockers.length > 0 && moveEffect === null) throw new KernelAbstain(`move:${move.id}`, move.blockers[0] as string);
  const unsupported: Prepared = { kind: "outcome", outcome: { kind: "unsupported", reason: "query_kind", position: null } };
  if (moveEffect?.kind === "fails") return unsupported;
  if (moveEffect?.kind === "needs_sleep" && query.attacker.status !== "slp") return unsupported;
  const spread = SPREAD_TARGETS.has(move.target);
  if (!spread && !SINGLE_TARGETS.has(move.target)) throw new KernelAbstain(`move:${move.id}`, "target");
  if (move.base_power <= 0) throw new KernelAbstain(`move:${move.id}`, "base_power");
  if (move.flags.has("powder")) throw new KernelAbstain(`move:${move.id}`, "powder");
  const weatherCondition = state.weather === null ? null : conditionOf(state.weather, ["weather_damage", "weather_stat"], "weather");
  if (weatherCondition?.kind === "weather_damage" && weatherCondition.exceptions.includes(move.id)) throw new KernelAbstain(`move:${move.id}`, "weather_rule");
  const terrainCondition = state.terrain === null ? null : (conditionOf(state.terrain, ["terrain"], "terrain") as Extract<KernelCondition, { kind: "terrain" }>);

  const sides = state.sides.map((side) => SIDE_IDS.filter(([key]) => side[key]).map(([, id]) => id));
  for (const id of sides.flat()) conditionOf(id, ["screen", "no_damage_effect"], "side");
  const suppressed = [attacker, defender].some((f) => f.ability_effect?.kind === "suppress_weather");
  const weatherId = suppressed ? null : state.weather;
  const field = [...(state.weather === null ? [] : [state.weather]), ...(state.terrain === null ? [] : [state.terrain]), ...(state.trick_room ? ["trickroom"] : [])];
  const moldBreaker = attacker.ability_effect?.kind === "mold_breaker";
  /** The effect of a Pokemon's ability as it acts (a breakable ability of the defender is ignored by a Mold Breaker attacker). */
  const abilityOf = (f: Fighter): KernelEffect | null => (f.role === "defender" && moldBreaker && data.breakable(f.ability) ? null : f.ability_effect);
  /** Whether a handler's ability is one a Mold Breaker attacker ignores (the engine still finds the handler, then skips it). */
  const broken = (h: Handler): boolean => h.kind === "ability" && h.holder === "defender" && moldBreaker && data.breakable(h.id);
  // Every event of the move step and of `getDamage` must order its handlers without a tie the kernel cannot settle.
  const order: Record<string, Handler[]> = {};
  for (const event of ORDERED_EVENTS) {
    const target = DEFENDER_EVENTS.has(event) ? defender : attacker;
    order[event] = handlers(ctx, event, target, target === attacker ? defender : attacker, sides, field);
  }
  const grounded = (f: Fighter): boolean => !f.types.includes("Flying") && abilityOf(f)?.kind !== "ground_immunity";

  // A priority move into a side that blocks priority fails at the move step (before any hit step).
  if (move.priority > 0 && abilityOf(defender)?.kind === "priority_block") return unsupported;
  // The move's type after the attacker's type-changing ability (ModifyType).
  const ate = attacker.ability_effect?.kind === "ate" && move.type === "Normal" && !attacker.ability_effect.exceptions.includes(move.id) ? attacker.ability_effect : null;
  const moveType = ate?.kind === "ate" ? ate.type : move.type;

  // The immunity steps (each deals nothing): an absorbing or flag immunity of the defender, Psychic Terrain against a priority
  // move, the type chart (a Ground move by groundedness).
  const defenderAbility = abilityOf(defender);
  if (defenderAbility?.kind === "absorb" && defenderAbility.type === moveType) return blocked(ctx, defender);
  if (defenderAbility?.kind === "flag_immunity" && move.flags.has(defenderAbility.flag)) return blocked(ctx, defender);
  if (terrainCondition?.blocks_priority === true && move.priority > 0 && grounded(defender)) return blocked(ctx, defender);
  const ignores = move.ignore_immunity === true || (move.ignore_immunity !== null && move.ignore_immunity.includes(moveType));
  if (!ignores) {
    const immune = moveType === "Ground" ? !grounded(defender) : defender.types.some((t) => data.chart.get(t)?.get(moveType) === 3);
    if (immune) return blocked(ctx, defender);
  }

  // Handlers of the events below that the kernel does not compute abstain; those it does are applied in the engine's order.
  for (const event of ["ModifyCritRatio", "CriticalHit", "Effectiveness", "ModifyType", "NegateImmunity", "ModifyBoost"]) {
    for (const h of order[event] as Handler[]) {
      if (broken(h)) continue;
      const f = h.holder === "attacker" ? attacker : h.holder === "defender" ? defender : null;
      const e = f === null ? null : effectOf(f, h);
      const known =
        (event === "ModifyCritRatio" && h.holder === "attacker" && e?.kind === "crit_ratio") ||
        (event === "CriticalHit" && h.holder === "defender" && e?.kind === "no_crit") ||
        (event === "ModifyType" && h.holder === "attacker" && e?.kind === "ate");
      if (!known) throw new KernelAbstain(`${h.kind}:${h.id}`, event);
    }
  }

  // The crit: the move's data, the attacker's crit-ratio item, the defender's crit immunity.
  let ratio = move.crit_ratio;
  if (attacker.item_effect?.kind === "crit_ratio") ratio += attacker.item_effect.add;
  ratio = Math.max(0, Math.min(4, ratio));
  const critImmune = defenderAbility?.kind === "no_crit";
  const critChance: DamageAnswer["crit_chance"] = critImmune ? { kind: "never" } : move.will_crit ? { kind: "always" } : ratio === 0 ? { kind: "never" } : { kind: "chance", numerator: 1, denominator: CRIT_MULT[ratio] as number };

  // BasePower (an event on the attacker).
  // The move's base power before the BasePower event: its data, or what its callback returns.
  const boosts = query.attacker.boosts;
  const ownBasePower =
    moveEffect?.kind === "boost_power"
      ? move.base_power + moveEffect.per * [boosts.atk, boosts.def, boosts.spa, boosts.spd, boosts.spe].reduce((sum, b) => sum + Math.max(0, b), 0)
      : moveEffect?.kind === "constant_power"
        ? move.base_power * moveEffect.multiplier
        : move.base_power;
  let bpChain = M_ONE;
  // The move's own BasePower hook runs with the attacker's handlers (the engine adds it first, then sorts: its sub-order is 0).
  const moveBasePower = ctx.data.moveHookOrder(move.id, "onBasePower");
  const basePowerHandlers = [...(order.BasePower as Handler[])];
  if (moveBasePower !== null) {
    const own: Handler = { holder: "attacker", kind: "condition", id: `move:${move.id}`, order: moveBasePower.order === false ? 4294967296 : moveBasePower.order, priority: moveBasePower.priority, pokemon: true, sub_order: moveBasePower.sub_order };
    for (const h of basePowerHandlers) if (h.pokemon && h.holder === "defender" && h.order === own.order && h.priority === own.priority) throw new KernelAbstain("state:handler_tie", "BasePower");
    const at = basePowerHandlers.findIndex((h) => h.order > own.order || (h.order === own.order && (h.priority < own.priority || (h.priority === own.priority && (!h.pokemon || h.sub_order > own.sub_order)))));
    basePowerHandlers.splice(at < 0 ? basePowerHandlers.length : at, 0, own);
  }
  for (const h of basePowerHandlers) {
    if (h.id === `move:${move.id}`) {
      if (moveEffect?.kind !== "status_power") throw new KernelAbstain(`move:${move.id}`, "onBasePower");
      if (attacker.status !== null && attacker.status !== "slp") bpChain = chain(bpChain, moveEffect.modifier);
      continue;
    }
    if (broken(h)) continue;
    if (h.holder === "field") {
      const c = ctx.effects.conditions[h.id];
      if (c?.kind !== "terrain") throw new KernelAbstain(`condition:${h.id}`, "BasePower");
      if (c.boost !== null && c.boost.type === moveType && grounded(attacker)) bpChain = chain(bpChain, c.boost.modifier);
      if (c.weakens !== null && c.weakens.moves.includes(move.id) && grounded(defender)) bpChain = chain(bpChain, c.weakens.modifier);
      if (c.halves !== null && c.halves.type === moveType && grounded(defender)) bpChain = chain(bpChain, c.halves.modifier);
      continue;
    }
    const e = h.holder === "attacker" ? effectOf(attacker, h) : null;
    if (e?.kind === "base_power_flag") {
      if (move.flags.has(e.flag)) bpChain = chain(bpChain, e.modifier);
    } else if (e?.kind === "technician") {
      if (modify(ownBasePower, bpChain) <= e.threshold) bpChain = chain(bpChain, e.modifier);
    } else if (e?.kind === "type_boost") {
      if (moveType === e.type) bpChain = chain(bpChain, e.modifier);
    } else if (e?.kind === "category_power") {
      if (move.category === e.category) bpChain = chain(bpChain, e.modifier);
    } else if (e?.kind === "ate") {
      if (ate !== null) bpChain = chain(bpChain, e.modifier);
    } else if (e?.kind === "sheer_force") {
      if ((move.has_secondaries && !move.sheer_force_boost) || move.sheer_force_boost) bpChain = chain(bpChain, e.modifier);
    } else throw new KernelAbstain(`${h.kind}:${h.id}`, "BasePower");
  }
  const basePower = Math.max(1, modify(ownBasePower, bpChain));

  const physical = move.category === "Physical";
  const offensive = move.offensive_target ? defender : attacker;
  const atkStat = move.offensive_stat ?? (physical ? "atk" : "spa");
  const defStat = move.defensive_stat ?? (physical ? "def" : "spd");
  const atkEvent = physical ? "ModifyAtk" : "ModifySpA";
  const defEvent = defStat === "def" ? "ModifyDef" : "ModifySpD";

  // The attacking stat's chain (an event on the attacker, whose stat it is not under Foul Play), then the defending stat's.
  let atkChain = M_ONE;
  for (const h of order[atkEvent] as Handler[]) {
    if (broken(h)) continue;
    const f = h.holder === "attacker" ? attacker : h.holder === "defender" ? defender : null;
    const e = f === null ? null : effectOf(f, h);
    if (f === attacker && e?.kind === "pinch") {
      if (moveType === e.type && attacker.hp * e.threshold.denominator <= (attacker.stats[0] as number) * e.threshold.numerator) atkChain = chain(atkChain, e.modifier);
    } else if (f === attacker && e?.kind === "guts" && physical) {
      if (attacker.status !== null) atkChain = chain(atkChain, e.modifier);
    } else if (f === attacker && e?.kind === "attack_multiplier" && (e.stat === "atk") === physical) {
      if (e.weather === undefined || e.weather === weatherId) atkChain = chain(atkChain, e.modifier);
    } else if (f === defender && e?.kind === "source_type_attack") {
      if (e.types.includes(moveType)) atkChain = chain(atkChain, e.modifier);
    } else if (e?.kind === "no_damage_effect") continue;
    else throw new KernelAbstain(`${h.kind}:${h.id}`, atkEvent);
  }
  // The defending stat's handlers: a weather's stat hook modifies the stat at once (`modify`), the others chain to the end.
  const defHandlers: StatHandler[] = [];
  for (const h of order[defEvent] as Handler[]) {
    if (broken(h)) continue;
    if (h.holder === "field") {
      const c = ctx.effects.conditions[h.id];
      if (c?.kind !== "weather_stat") throw new KernelAbstain(`condition:${h.id}`, defEvent);
      if (h.id === weatherId && c.stat === defStat && defender.types.includes(c.type)) defHandlers.push({ direct: c.modifier });
      continue;
    }
    const e = h.holder === "defender" ? effectOf(defender, h) : null;
    if (e?.kind === "defense_multiplier" && e.stat === defStat) {
      if (e.terrain === undefined || e.terrain === state.terrain) defHandlers.push({ chain: e.modifier });
    } else throw new KernelAbstain(`${h.kind}:${h.id}`, defEvent);
  }

  const level = attackerSet.level;
  const baseDamage = (crit: boolean): number => {
    let atkBoost = offensive.boosts[atkStat as keyof Combatant["boosts"]];
    let defBoost = defender.boosts[defStat as keyof Combatant["boosts"]];
    if (crit && atkBoost < 0) atkBoost = 0;
    if ((crit && defBoost > 0) || move.ignore_defensive) defBoost = 0;
    const attack = modify(staged(offensive.stats[STATS.indexOf(atkStat as Stat)] as number, atkBoost), atkChain);
    const defense = applyStatHandlers(staged(defender.stats[STATS.indexOf(defStat as Stat)] as number, defBoost), defHandlers);
    const factor = div(2 * level, 5) + 2;
    return div(div(factor * basePower * attack, defense), 50);
  };

  // WeatherModifyDamage: the effective weather's modifier for the move's type (none while an ability in play suppresses weather).
  const effectiveWeather = weatherId === null ? null : ctx.effects.conditions[weatherId];
  const weather = effectiveWeather?.kind === "weather_damage" ? (effectiveWeather.types.find((t) => t.type === moveType)?.modifier ?? M_ONE) : M_ONE;

  let stab = attacker.types.includes(moveType) ? M_ONE_AND_HALF : M_ONE;
  for (const h of order.ModifySTAB as Handler[]) {
    const e = h.holder === "attacker" ? effectOf(attacker, h) : null;
    if (e?.kind !== "stab") throw new KernelAbstain(`${h.kind}:${h.id}`, "ModifySTAB");
    if (stab !== M_ONE) stab = e.modifier;
  }

  let typeMod = 0;
  for (const t of defender.types) {
    const code = data.chart.get(t)?.get(moveType) ?? 0;
    typeMod += code === 1 ? 1 : code === 2 ? -1 : 0;
  }
  typeMod = Math.max(-6, Math.min(6, typeMod));

  const infiltrates = attacker.ability_effect?.kind === "infiltrator";
  const finalChain = (crit: boolean): number => {
    let m = M_ONE;
    for (const h of order.ModifyDamage as Handler[]) {
      if (broken(h)) continue;
      if (h.holder === "side0" || h.holder === "side1") {
        const c = ctx.effects.conditions[h.id];
        if (c?.kind !== "screen") throw new KernelAbstain(`condition:${h.id}`, "ModifyDamage");
        const category = physical ? "Physical" : "Special";
        // A screen of both categories (Aurora Veil) yields to a screen of the move's own category on the same side.
        const yields = c.yields_to.some((y) => sides[1]?.includes(y) && (ctx.effects.conditions[y] as { category?: string } | undefined)?.category === category);
        const applies = c.category === null ? !yields : c.category === category;
        if (applies && h.holder === "side1" && !crit && !infiltrates) m = chain(m, c.modifier);
        continue;
      }
      const f = h.holder === "attacker" ? attacker : h.holder === "defender" ? defender : null;
      const e = f === null ? null : effectOf(f, h);
      if (f === attacker && e?.kind === "crit_damage") {
        if (crit) m = chain(m, e.modifier);
      } else if (f === attacker && e?.kind === "final_modifier") m = chain(m, e.modifier);
      else if (f === attacker && e?.kind === "super_effective_dealt") {
        if (typeMod > 0) m = chain(m, e.modifier);
      } else if (f === defender && e?.kind === "super_effective_taken") {
        if (typeMod > 0) m = chain(m, e.modifier);
      } else if (f === defender && e?.kind === "full_hp_taken") {
        if (defender.hp >= (defender.stats[0] as number)) m = chain(m, e.modifier);
      } else if (f === defender && e?.kind === "resist_berry") {
        // A berry is eaten only if the attacker does not stop its foes eating (Unnerve).
        if (moveType === e.type && (!e.super_effective || typeMod > 0) && attacker.ability_effect?.kind !== "unnerve") m = chain(m, e.modifier);
      } else throw new KernelAbstain(`${h.kind}:${h.id}`, "ModifyDamage");
    }
    return m;
  };

  return {
    kind: "rolls",
    decoded: {
      base: [baseDamage(false), baseDamage(!critImmune)],
      spread,
      weather,
      stab,
      type_mod: typeMod,
      burn: attacker.status === "brn" && physical && attacker.ability_effect?.kind !== "guts" && !(moveEffect?.kind === "status_power" && moveEffect.burn_exempt),
      final: [finalChain(false), finalChain(!critImmune)],
      crit_applies: !critImmune,
      crit_chance: critChance,
      defender,
    },
  };
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
