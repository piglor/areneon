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
import { acceptedDraws, critDraw, type RecordingRandom, recordingRandom } from "./mechanics-random.ts";
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
}
interface DmgPokemon {
  hp: number;
  maxhp: number;
  status: string;
  boosts: Record<string, number>;
  species: { name: string };
  item: string;
  setStatus(status: string): unknown;
  getMoveHitData(move: DmgMove): HitData;
}
interface DmgActions {
  runMegaEvo(pokemon: DmgPokemon): unknown;
  useMoveInner(move: string, pokemon: DmgPokemon, options: { target: DmgPokemon }): unknown;
  getDamage(source: DmgPokemon, target: DmgPokemon, move: DmgMove, suppressMessages: boolean): unknown;
  hitStepAccuracy: (targets: DmgPokemon[]) => boolean[];
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
  prng: RecordingRandom;
  dex: { conditions: { get(id: string): { exists: boolean; effectType: string } }; moves: { get(id: string): { exists: boolean; category: string; multihit?: unknown; id: string } } };
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
    pokemon.setStatus(combatant.status);
    if (pokemon.status !== combatant.status) throw new DamageStateError("the engine does not take that status");
  }
  for (const key of STAT_KEYS) pokemon.boosts[key] = combatant.boosts[key];
  return pokemon;
}

/** The scenario battle with the whole declared state, ready to be cloned. */
function declaredBattle(engine: Engine, scenarioId: string, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): DmgBattle {
  const battle = asDmg(startScenario(engine, scenarioId, { p1: [attackerSet, partnerOf(attackerSet)], p2: [defenderSet, partnerOf(defenderSet)] }));
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
  damage: number;
  crit: boolean;
}

/**
 * One run of the engine's move step on a clone. `crit` sets the active move's
 * `willCrit` just before `getDamage` (the engine's own field, read before any
 * crit draw); undefined leaves the move's own rule. `forceChance` is what the
 * engine's `randomChance` answers.
 */
function run(engine: Engine, base: DmgBattle, moveId: string, roll: number, crit: boolean | undefined, forceChance: boolean): Run {
  const battle = asDmg(engine.sim.Battle.fromJSON(base.toJSON()));
  const random = recordingRandom({ roll, chance: forceChance });
  battle.prng = random;
  const attacker = battle.sides[0]?.pokemon[0] as DmgPokemon;
  const defender = battle.sides[1]?.pokemon[0] as DmgPokemon;
  const seen: Run = { draws: random.draws, hit: false, damage: 0, crit: false };
  // Nothing is dealt and no accuracy is rolled: the step that deals damage reads it instead.
  battle.actions.hitStepAccuracy = (targets) => targets.map(() => true);
  battle.actions.hitStepMoveHitLoop = (targets, pokemon, move) => {
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

/**
 * Answer a `damage` query on the scenario `scenarioId`. `attackerSet` and
 * `defenderSet` are the engine-checked sets (the probe has validated them).
 */
export function probeDamage(engine: Engine, scenarioId: string, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): DamageOutcome {
  const base = declaredBattle(engine, scenarioId, query, attackerSet, defenderSet);
  const move = base.dex.moves.get(query.move);
  if (!move.exists || move.id !== query.move) throw new DamageStateError("the move is not an engine move");
  // Outside the v1 scope (ADR-016 section 3): a move that deals no damage of its own, and a multi-hit move.
  if (move.category === "Status" || move.multihit !== undefined) return { kind: "unsupported", reason: "query_kind", position: null };

  const vectors = (crit: boolean): number[] | null => {
    const out: number[] = [];
    for (const roll of ROLLS) {
      const seen = run(engine, base, query.move, roll, crit, false);
      if (!acceptedDraws(seen.draws, false)) return null;
      out.push(seen.damage);
    }
    return out;
  };
  const noCrit = vectors(false);
  const critVector = vectors(true);
  const observed = run(engine, base, query.move, 0, undefined, true);
  if (noCrit === null || critVector === null || !acceptedDraws(observed.draws, true)) return { kind: "unsupported", reason: "random_effect", position: null };

  // The chance is the engine's own: an exact `randomChance` pair, a crit it always makes, or none.
  const draw = critDraw(observed.draws);
  const crit_chance: DamageAnswer["crit_chance"] =
    observed.hit && observed.crit ? (draw === null ? { kind: "always" } : { kind: "chance", ...draw }) : { kind: "never" };
  const defender = scenarioPokemon(base as unknown as SimBattle, 1, 0) as unknown as DmgPokemon;
  return {
    kind: "computed",
    answer: {
      kind: "damage",
      no_crit: noCrit as DamageAnswer["no_crit"],
      crit: critVector as DamageAnswer["crit"],
      crit_chance,
      hp_current: defender.hp,
      hp_max: defender.maxhp,
      // KO counts are slice 5 (ADR-016 section 7); until then no count is given.
      ko: { kind: "unsupported", reason: "ko_state_change" },
    },
  };
}

/**
 * The scenario battle with the declared state and nothing played, for a test
 * that plays the same state as a real battle (damage equals the HP lost).
 */
export const declaredDamageScenario = (engine: Engine, scenarioId: string, query: DamageQuery, attackerSet: PokemonSet, defenderSet: PokemonSet): SimBattle =>
  declaredBattle(engine, scenarioId, query, attackerSet, defenderSet) as unknown as SimBattle;
