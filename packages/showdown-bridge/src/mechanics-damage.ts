// The `damage` probe (ADR-016 section 2, slice 3). Areneon computes no damage
// value: the engine's own `getDamage` does, on a clone of a scenario battle that
// holds the declared sets and the declared state. The probe runs the engine's
// own move step (`useMoveInner`: target resolution, `ModifyMove`, the immunity
// steps, the spread flag) on 33 clones with a recording random source
// (`mechanics-random.ts`): for each of the 16 damage rolls once with the crit
// forced off and once forced on, and once with the move's own crit rule to
// observe the crit chance from the engine's own draw. Only the accuracy step
// and the damage loop are replaced, so nothing is dealt and nothing hidden is
// read. The probe holds no crit table, no formula and no target rule.

import type { Engine, PokemonSet, SimBattle } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { koBlocked, koCounts } from "./mechanics-ko.ts";
import { acceptedDraws, critDraw, type Draw, type RecordingRandom, recordingRandom } from "./mechanics-random.ts";
import { scenarioPokemon, startScenario } from "./mechanics-scenario.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsResult } from "./mechanics-result.ts";

type DamageQuery = Extract<MechanicsQuery["query"], { kind: "damage" }>;
type Combatant = DamageQuery["attacker"];
type DamageAnswer = Extract<Extract<MechanicsResult, { kind: "computed" }>["answer"], { kind: "damage" }>;

/** What `probeDamage` returns: an answer, or the one closed reason the probe cannot answer. */
export type DamageOutcome =
  | { kind: "computed"; answer: DamageAnswer }
  | { kind: "unsupported"; reason: "query_kind" | "random_effect"; position: number | null };

/** A query the probe cannot put into the scenario (an unknown id, a state the engine refuses). */
export class DamageStateError extends Error {
  override name = "DamageStateError";
}

// The parts of the engine this module reaches, narrowed locally so the shared
// engine types stay as they are.
interface HitData {
  crit: boolean;
}
interface DmgMove {
  willCrit?: boolean;
  spreadHit?: boolean;
  category: string;
  multihit?: unknown;
  moveHitData?: unknown;
}
interface DmgPokemon {
  hp: number;
  maxhp: number;
  status: string;
  boosts: Record<string, number>;
  species: { name: string };
  item: string;
  ability: string;
  setStatus(status: string, source: null, sourceEffect: null, ignoreImmunities: boolean): unknown;
  getMoveHitData(move: DmgMove): HitData;
}
type HitStep = (targets: DmgPokemon[], pokemon: DmgPokemon, move: DmgMove) => (number | boolean | "" | undefined)[] | undefined;
interface DmgActions {
  runMegaEvo(pokemon: DmgPokemon): unknown;
  useMoveInner(move: string, pokemon: DmgPokemon, options: { target: DmgPokemon }): unknown;
  getDamage(source: DmgPokemon, target: DmgPokemon, move: DmgMove, suppressMessages: boolean): unknown;
  hitStepAccuracy: (targets: DmgPokemon[]) => boolean[];
  hitStepInvulnerabilityEvent: HitStep;
  hitStepTryHitEvent: HitStep;
  hitStepTypeImmunity: HitStep;
  hitStepTryImmunity: HitStep;
  hitStepMoveHitLoop: (targets: DmgPokemon[], pokemon: DmgPokemon, move: DmgMove) => boolean[];
}
interface DmgSide {
  pokemon: DmgPokemon[];
  addSideCondition(id: string, source: DmgPokemon): unknown;
  sideConditions: Record<string, unknown>;
}
interface DmgField {
  weather: string;
  terrain: string;
  clearWeather(): unknown;
  clearTerrain(): unknown;
  setWeather(id: string, source: DmgPokemon): unknown;
  setTerrain(id: string, source: DmgPokemon): unknown;
  addPseudoWeather(id: string, source: DmgPokemon): unknown;
  pseudoWeather: Record<string, unknown>;
}
interface DmgBattle {
  sides: DmgSide[];
  field: DmgField;
  actions: DmgActions;
  prng: unknown;
  dex: { conditions: { get(id: string): { exists: boolean; effectType: string } }; moves: { get(id: string): { exists: boolean; category: string; multihit?: unknown; id: string; damage?: unknown; damageCallback?: unknown; ohko?: unknown } } };
  toJSON(): object;
}

const asDmg = (battle: SimBattle): DmgBattle => battle as unknown as DmgBattle;

const STAT_KEYS = ["atk", "def", "spa", "spd", "spe"] as const;
const SIDE_CONDITIONS = [
  ["aurora_veil", "auroraveil"],
  ["light_screen", "lightscreen"],
  ["reflect", "reflect"],
  ["tailwind", "tailwind"],
] as const;

/**
 * A partner that adds nothing of its own: the Pokemon's species, no ability and
 * no item, so an ally ability (Friend Guard, Battery, Power Spot) of a copy
 * cannot change the declared Pokemon's damage. It exists so both slots of both
 * sides are occupied: a spread move then has two foes and the engine sets
 * `spreadHit` itself, and a screen sees more than one active Pokemon.
 */
const partnerOf = (set: PokemonSet): PokemonSet => ({ ...set, ability: "No Ability", item: "", name: set.species });

/** The engine's condition of that id and kind, or a refusal: an unknown id never reaches the engine's setters. */
function requireCondition(battle: DmgBattle, id: string, effectType: string): void {
  const condition = battle.dex.conditions.get(id);
  if (!condition.exists || condition.effectType !== effectType) throw new DamageStateError(`the engine has no ${effectType.toLowerCase()} of that id`);
}

/** Apply one combatant's declared state, after the battle started (absolute, replacing any switch-in effect). */
function declare(battle: DmgBattle, side: number, combatant: Combatant): DmgPokemon {
  const pokemon = battle.sides[side]?.pokemon[0] as DmgPokemon;
  if (combatant.mega) {
    const before = pokemon.species.name;
    battle.actions.runMegaEvo(pokemon);
    if (pokemon.species.name === before) throw new DamageStateError("the Pokemon cannot Mega Evolve");
  }
  if (combatant.hp !== null) {
    if (combatant.hp > pokemon.maxhp) throw new DamageStateError("hp is above the Pokemon's maximum HP");
    pokemon.hp = combatant.hp;
  }
  if (combatant.status !== null) {
    requireCondition(battle, combatant.status, "Status");
    // The status is an observed fact, so type immunities do not apply (Corrosion poisons a Steel type, Burn Up removes the Fire type of a burned user); an ability or the engine's own hooks can still refuse it.
    pokemon.setStatus(combatant.status, null, null, true);
    if (pokemon.status !== combatant.status) throw new DamageStateError("the engine does not take that status");
  }
  for (const key of STAT_KEYS) pokemon.boosts[key] = combatant.boosts[key];
  return pokemon;
}

/** At most this many started scenarios are kept per engine; a table past it is emptied (a started scenario is a pure function of its key, so forgetting only costs time). */
const STARTED_LIMIT = 256;
const startedTables = new WeakMap<Engine, Map<string, string>>();
/** How many scenario starts were skipped by the table (diagnostics, like `damagePathStats`). */
export const scenarioStartStats = { started: 0, reused: 0 };

/**
 * The started scenario battle of two sets, before any declared state. The engine starts it once per
 * (scenario format, attacker set, defender set) and the serialised start is kept; every caller, the first included,
 * gets its own battle from that serialisation, as a clone of the reference path does, so the result never depends on the table's contents.
 */
function startedScenario(engine: Engine, scenarioId: string, attackerSet: PokemonSet, defenderSet: PokemonSet): DmgBattle {
  const table = startedTables.get(engine) ?? new Map<string, string>();
  startedTables.set(engine, table);
  // The sets are the validated, canonical ones, which serialise the same way every time; a different order of keys would only miss.
  const key = JSON.stringify([scenarioId, attackerSet, defenderSet]);
  let json = table.get(key);
  if (json === undefined) {
    scenarioStartStats.started++;
    const battle = startScenario(engine, scenarioId, { p1: [attackerSet, partnerOf(attackerSet)], p2: [defenderSet, partnerOf(defenderSet)] });
    json = JSON.stringify(battle.toJSON());
    if (table.size >= STARTED_LIMIT) table.clear();
    table.set(key, json);
  } else scenarioStartStats.reused++;
  return asDmg(engine.sim.Battle.fromJSON(json));
}

/** The scenario battle with the whole declared state, ready to be cloned. */
function declaredBattle(engine: Engine, scenarioId: string, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet, reference = false): DmgBattle {
  // The reference path starts the scenario itself every time, so a test comparing the paths also compares the kept start with a fresh one.
  const battle = reference ? asDmg(startScenario(engine, scenarioId, { p1: [attackerSet, partnerOf(attackerSet)], p2: [defenderSet, partnerOf(defenderSet)] })) : startedScenario(engine, scenarioId, attackerSet, defenderSet);
  const attacker = declare(battle, 0, query.attacker);
  declare(battle, 1, query.defender);
  const { state } = query;
  // The field is exactly the declared one: a weather or terrain a switch-in set is cleared first.
  battle.field.clearWeather();
  battle.field.clearTerrain();
  if (state.weather !== null) {
    requireCondition(battle, state.weather, "Weather");
    battle.field.setWeather(state.weather, attacker);
    if (battle.field.weather !== state.weather) throw new DamageStateError("the engine does not take that weather");
  }
  if (state.terrain !== null) {
    requireCondition(battle, state.terrain, "Terrain");
    battle.field.setTerrain(state.terrain, attacker);
    if (battle.field.terrain !== state.terrain) throw new DamageStateError("the engine does not take that terrain");
  }
  if (state.trick_room) battle.field.addPseudoWeather("trickroom", attacker);
  state.sides.forEach((declared, index) => {
    const side = battle.sides[index] as DmgSide;
    for (const [key, id] of SIDE_CONDITIONS) {
      if (!declared[key]) continue;
      side.addSideCondition(id, side.pokemon[0] as DmgPokemon);
      if (!(id in side.sideConditions)) throw new DamageStateError(`the engine does not take ${id}`);
    }
  });
  return battle;
}

/** What one run on a clone saw. */
interface Run {
  draws: RecordingRandom["draws"];
  /** The damage loop was reached for the defender. */
  hit: boolean;
  /** The engine made the move multi-hit after `ModifyMove` (Parental Bond): one hit is not its damage. */
  multi: boolean;
  /** An immunity or protection step of the engine removed the defender: the move deals nothing. */
  blocked: boolean;
  damage: number;
  crit: boolean;
}

/**
 * One run of the engine's move step on a clone of the serialised battle `json` (the caller serialises once). `crit` sets the active move's
 * `willCrit` just before `getDamage` (the engine's own field, read before any
 * crit draw); undefined leaves the move's own rule. `forceChance` is what the
 * engine's `randomChance` answers.
 */
function run(engine: Engine, json: string, moveId: string, roll: number, crit: boolean | undefined, forceChance: boolean): Run {
  const battle = asDmg(engine.sim.Battle.fromJSON(json));
  const random = recordingRandom({ roll, chance: forceChance });
  battle.prng = random;
  const attacker = battle.sides[0]?.pokemon[0] as DmgPokemon;
  const defender = battle.sides[1]?.pokemon[0] as DmgPokemon;
  const seen: Run = { draws: random.draws, hit: false, blocked: false, multi: false, damage: 0, crit: false };
  // Nothing is dealt and no accuracy is rolled: the step that deals damage reads it instead.
  battle.actions.hitStepAccuracy = (targets) => targets.map(() => true);
  // The engine's own steps that can remove the defender (an immunity, Protect, Volt Absorb) are kept; the run notes it.
  for (const name of ["hitStepInvulnerabilityEvent", "hitStepTryHitEvent", "hitStepTypeImmunity", "hitStepTryImmunity"] as const) {
    const step = battle.actions[name];
    battle.actions[name] = function (this: unknown, targets, pokemon, move) {
      const results = step.call(this, targets, pokemon, move);
      const index = targets.indexOf(defender);
      if (results !== undefined && index >= 0 && !(results[index] || results[index] === 0)) seen.blocked = true;
      return results;
    };
  }
  battle.actions.hitStepMoveHitLoop = (targets, pokemon, move) => {
    seen.multi = move.multihit !== undefined && move.multihit !== null && move.multihit !== false;
    if (targets.includes(defender)) {
      if (crit !== undefined) move.willCrit = crit;
      const damage = battle.actions.getDamage(pokemon, defender, move, true);
      // `false`, `null` and `undefined` are the engine's "deals nothing" (an immunity, a zero base power).
      if (typeof damage === "number") seen.damage = damage;
      else if (damage !== false && damage !== null && damage !== undefined) throw new DamageStateError("getDamage returned a value the probe does not know");
      seen.hit = true;
      seen.crit = defender.getMoveHitData(move).crit;
    }
    return targets.map(() => true);
  };
  battle.actions.useMoveInner(moveId, attacker, { target: defender });
  return seen;
}

const ROLLS = Array.from({ length: 16 }, (_, index) => index);

/** Which implementation answers a `damage` query; both give byte-identical results (the one-clone path falls back to the per-roll one). */
export type DamagePath = "one-clone" | "per-roll";

/** The defender as the scenario holds it after the declared state, read before any move step. */
interface Facts {
  hp: number;
  maxhp: number;
  item: string;
  ability: string;
}
const factsOf = (base: DmgBattle): Facts => {
  const { hp, maxhp, item, ability } = scenarioPokemon(base as unknown as SimBattle, 1, 0) as unknown as DmgPokemon;
  return { hp, maxhp, item, ability };
};

/** What the two paths share once the runs are done: the crit chance, the defender's HP and the KO counts. */
function assemble(defender: Facts, observed: { hit: boolean; crit: boolean; draws: readonly Draw[] }, noCrit: number[], critVector: number[], koBlockers: MechanicsConfig["ko_blockers"]): DamageOutcome {
  // The chance is the engine's own: an exact `randomChance` pair, a crit it always makes, or none.
  const draw = critDraw(observed.draws);
  const crit_chance: DamageAnswer["crit_chance"] = observed.hit && observed.crit ? (draw === null ? { kind: "always" } : { kind: "chance", ...draw }) : { kind: "never" };
  return {
    kind: "computed",
    answer: {
      kind: "damage",
      no_crit: noCrit as DamageAnswer["no_crit"],
      crit: critVector as DamageAnswer["crit"],
      crit_chance,
      hp_current: defender.hp,
      hp_max: defender.maxhp,
      // The defender as the engine holds it after the declared state (a Mega Evolution changes its ability): a blocker on the config's list withholds the counts, not the vectors.
      ko: koCounts(noCrit, critVector, defender.hp, koBlocked(koBlockers, defender)),
    },
  };
}

/**
 * The engine gives a field effect's state its `target` (the field) the first time an event reaches it, so a battle serialised before the
 * first `getDamage` differs from one serialised after it in that key alone; it is the same on every call, and the comparison leaves it out.
 */
// The key names and the "[Field]" marker are the pinned engine's serialisation (sim/state.ts, sim/field.ts); a pin bump re-checks them.
function withoutLazyTargets(state: { field: object }): object {
  const field = JSON.parse(JSON.stringify(state.field)) as Record<string, unknown>;
  const states = [field.weatherState, field.terrainState, ...Object.values((field.pseudoWeather ?? {}) as Record<string, unknown>)];
  for (const entry of states) if (typeof entry === "object" && entry !== null && (entry as { target?: unknown }).target === "[Field]") delete (entry as { target?: unknown }).target;
  return { ...state, field };
}

/**
 * How often the one-clone path answered and how often it handed the query to the per-roll path (a call that changed the battle).
 * Process-global counters for diagnostics (the bench, tests); no result depends on them.
 */
export const damagePathStats = { one_clone: 0, fell_back: 0 };

/** The reference path: 33 clones, each running the engine's whole move step for one roll. */
function perRoll(engine: Engine, base: DmgBattle, moveId: string, koBlockers: MechanicsConfig["ko_blockers"]): DamageOutcome {
  const facts = factsOf(base);
  const json = JSON.stringify(base.toJSON());
  // The move never reached the damage loop for a reason other than an engine immunity (a charge turn, a failed
  // precondition, a self-targeting move) or that the engine made multi-hit (Parental Bond): its damage is not this query's answer, so nothing is claimed.
  const observed = run(engine, json, moveId, 0, undefined, true);
  if ((!observed.hit && !observed.blocked) || observed.multi) return { kind: "unsupported", reason: "query_kind", position: null };

  const vectors = (crit: boolean): number[] | null => {
    const out: number[] = [];
    for (const roll of ROLLS) {
      const seen = run(engine, json, moveId, roll, crit, false);
      if (!acceptedDraws(seen.draws, false)) return null;
      out.push(seen.damage);
    }
    return out;
  };
  const noCrit = vectors(false);
  const critVector = vectors(true);
  if (noCrit === null || critVector === null || !acceptedDraws(observed.draws, true)) return { kind: "unsupported", reason: "random_effect", position: null };
  return assemble(facts, observed, noCrit, critVector, koBlockers);
}

/** `getDamage`'s answer as a number: the engine's "deals nothing" (`false`, `null`, `undefined`) is 0. */
function damageNumber(damage: unknown): number {
  if (typeof damage === "number") return damage;
  if (damage !== false && damage !== null && damage !== undefined) throw new DamageStateError("getDamage returned a value the probe does not know");
  return 0;
}

/**
 * The one-clone path. The engine's move step runs once, on the declared scenario battle itself (which it
 * spends: a fallback starts from a rebuilt one); where the per-roll path stops at the damage loop, the loop here calls the engine's own
 * `getDamage` 33 times in turn: once with the move's own crit rule
 * (the observation), then the 16 rolls with the crit forced off and the 16 with
 * it forced on, each with the random source answering as the matching per-roll
 * clone's did and the move's hit data reset to what a fresh clone holds. A call
 * is a clone only if it leaves the battle as it found it, so the whole battle
 * (and the active move, but for the hit data) is compared before and after: a
 * difference (an eaten berry, a stacking volatile) returns `null` and the
 * caller takes the per-roll path. A move the engine's immunity steps stop before the damage loop deals
 * nothing; one more clone, answering as a vector clone does, shows its draws. The draws of each call are the prefix before the loop, the
 * call's own and the suffix after it, which is what its per-roll clone drew.
 */
function oneClone(engine: Engine, battle: DmgBattle, rebuild: () => DmgBattle, moveId: string, koBlockers: MechanicsConfig["ko_blockers"]): DamageOutcome | null {
  const facts = factsOf(battle);
  const realPrng = battle.prng;
  const random = recordingRandom({ roll: 0, chance: true });
  battle.prng = random;
  const attacker = battle.sides[0]?.pokemon[0] as DmgPokemon;
  const defender = battle.sides[1]?.pokemon[0] as DmgPokemon;
  const seen = { hit: false, blocked: false, multi: false, crit: false, mutated: false };
  const windows: { start: number; end: number }[] = [];
  const damages: number[] = [];
  battle.actions.hitStepAccuracy = (targets) => targets.map(() => true);
  for (const name of ["hitStepInvulnerabilityEvent", "hitStepTryHitEvent", "hitStepTypeImmunity", "hitStepTryImmunity"] as const) {
    const step = battle.actions[name];
    battle.actions[name] = function (this: unknown, targets, pokemon, move) {
      const results = step.call(this, targets, pokemon, move);
      const index = targets.indexOf(defender);
      if (results !== undefined && index >= 0 && !(results[index] || results[index] === 0)) seen.blocked = true;
      return results;
    };
  }
  // The battle as the engine serialises it (with its own random source back in place, which the recorder would answer wrongly) and the active move without its hit data.
  const snapshot = (move: DmgMove): string => {
    battle.prng = realPrng;
    try {
      return JSON.stringify([withoutLazyTargets(battle.toJSON() as { field: object }), { ...move, moveHitData: undefined }]);
    } finally {
      battle.prng = random;
    }
  };
  battle.actions.hitStepMoveHitLoop = (targets, pokemon, move) => {
    seen.multi = move.multihit !== undefined && move.multihit !== null && move.multihit !== false;
    if (!targets.includes(defender)) return targets.map(() => true);
    seen.hit = true;
    const ownCrit = move.willCrit;
    const hasHitData = move.moveHitData !== undefined;
    const hitData: unknown = hasHitData ? structuredClone(move.moveHitData) : undefined;
    const reset = (): void => {
      if (hasHitData) move.moveHitData = structuredClone(hitData);
      else delete move.moveHitData;
    };
    const before = snapshot(move);
    const call = (roll: number, chance: boolean, crit: boolean | undefined): void => {
      random.answer(roll, chance);
      reset();
      if (crit !== undefined) move.willCrit = crit;
      const start = random.draws.length;
      damages.push(damageNumber(battle.actions.getDamage(pokemon, defender, move, true)));
      windows.push({ start, end: random.draws.length });
    };
    call(0, true, undefined);
    seen.crit = defender.getMoveHitData(move).crit;
    random.answer(0, true);
    if (!seen.multi) {
      for (const crit of [false, true]) for (const roll of ROLLS) call(roll, false, crit);
      if (ownCrit === undefined) delete move.willCrit;
      else move.willCrit = ownCrit;
      reset();
      if (snapshot(move) !== before) seen.mutated = true;
    }
    return targets.map(() => true);
  };
  battle.actions.useMoveInner(moveId, attacker, { target: defender });

  if (seen.mutated) return null;
  if (seen.multi) return { kind: "unsupported", reason: "query_kind", position: null };
  if (!seen.hit && seen.blocked) {
    // An engine immunity or protection stopped the move before the damage loop: every roll deals 0 and the draws are the move step's alone,
    // which one clone with the vectors' chance answer shows (the step's draws do not depend on the damage roll).
    const vector = run(engine, JSON.stringify(rebuild().toJSON()), moveId, 0, false, false);
    if (!acceptedDraws(vector.draws, false) || !acceptedDraws(random.draws, true)) return { kind: "unsupported", reason: "random_effect", position: null };
    return assemble(facts, { hit: false, crit: false, draws: random.draws }, new Array<number>(16).fill(0), new Array<number>(16).fill(0), koBlockers);
  }
  if (!seen.hit) return { kind: "unsupported", reason: "query_kind", position: null };
  // The draws before the loop and after it are the same in every call. That a call's list equals its clone's relies on `acceptedDraws` being this strict:
  // it refuses any draw but one `random(16)` (and the observation's one `randomChance`), so no branch that depends on an answer can change a verdict.
  const first = windows[0] as { start: number; end: number };
  const last = windows[windows.length - 1] as { start: number; end: number };
  const prefix = random.draws.slice(0, first.start);
  const suffix = random.draws.slice(last.end);
  const drawsOf = (index: number): Draw[] => {
    const window = windows[index] as { start: number; end: number };
    return [...prefix, ...random.draws.slice(window.start, window.end), ...suffix];
  };
  const observedDraws = drawsOf(0);
  let accepted = acceptedDraws(observedDraws, true);
  for (let index = 1; index < windows.length; index++) accepted &&= acceptedDraws(drawsOf(index), false);
  if (!accepted) return { kind: "unsupported", reason: "random_effect", position: null };
  return assemble(facts, { hit: true, crit: seen.crit, draws: observedDraws }, damages.slice(1, 17), damages.slice(17, 33), koBlockers);
}

/**
 * Answer a `damage` query on the scenario `scenarioId`. `attackerSet` and
 * `defenderSet` are the engine-checked sets (the probe has validated them).
 * `koBlockers` is the mechanics config's list (ADR-016 section 3). `path` picks
 * the implementation; the results are byte-identical, the one-clone path is
 * the default and the faster.
 */
export function probeDamage(
  engine: Engine,
  scenarioId: string,
  query: DamageQuery,
  attackerSet: PokemonSet,
  defenderSet: PokemonSet,
  koBlockers: MechanicsConfig["ko_blockers"],
  path: DamagePath = "one-clone",
): DamageOutcome {
  const base = declaredBattle(engine, scenarioId, query, attackerSet, defenderSet, path === "per-roll");
  const move = base.dex.moves.get(query.move);
  if (!move.exists || move.id !== query.move) throw new DamageStateError("the move is not an engine move");
  // Outside the v1 scope (ADR-016 section 3): a move that deals no damage of its own, and a multi-hit move.
  // Also fixed and HP-based damage (`damage`, `damageCallback`, `ohko`): a number that is no roll vector.
  if (move.category === "Status" || move.multihit !== undefined || move.damage !== undefined || move.damageCallback !== undefined || move.ohko !== undefined) return { kind: "unsupported", reason: "query_kind", position: null };
  const rebuild = (): DmgBattle => declaredBattle(engine, scenarioId, query, attackerSet, defenderSet, path === "per-roll");
  if (path === "one-clone") {
    const fast = oneClone(engine, base, rebuild, query.move, koBlockers);
    if (fast !== null) {
      damagePathStats.one_clone++;
      return fast;
    }
    damagePathStats.fell_back++;
  }
  // The one-clone path played its move step on the scenario battle itself, so the per-roll path starts from a fresh one.
  return perRoll(engine, path === "one-clone" ? rebuild() : base, query.move, koBlockers);
}

/**
 * The scenario battle with the declared state and nothing played, for a test
 * that plays the same state as a real battle (damage equals the HP lost).
 */
export const declaredDamageScenario = (engine: Engine, scenarioId: string, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): SimBattle =>
  declaredBattle(engine, scenarioId, query, attackerSet, defenderSet) as unknown as SimBattle;
