// The `speed_order` probe (ADR-016 section 2, slice 2). Areneon computes no
// Speed or priority: it builds the scenario battle, applies the declared state
// through the engine's own methods, resolves each declared action with the
// engine's `resolveAction` (which reads `getActionSpeed`, the priority events
// and the fractional priority) and orders them with the engine's own
// `comparePriority`. Two actions that compare equal are one tie group, never
// sampled. Mega Evolutions come first, as their own groups in speed order;
// `runMegaEvo` then runs on the scenario battle and every move's priority and
// speed are read afterwards, so a Mega's new Speed applies that turn.
//
// The engine's random source is replaced before anything is declared: any
// draw (Quick Claw, a random target, a sleep duration) ends the probe with an
// `unsupported` result, never a guess. The scenario battle is built for this
// one query and dropped; a live battle is never touched.

import type { Engine, PokemonSet } from "./engine.ts";
import { RandomDraw, firstTarget, need, refusingSource } from "./mechanics-guard.ts";
import { declareHistory, type HBattle, type HistoryDeclaration } from "./mechanics-history.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { Outcome } from "./mechanics-result.ts";
import { startScenario } from "./mechanics-scenario.ts";

type SpeedOrderQuery = Extract<MechanicsQuery["query"], { kind: "speed_order" }>;

/** What the shared declaration and action steps read: the sides, the actions and the field (a `speed_order` and a `move_outcome` query both have them). */
export type DeclaredQuery = Pick<SpeedOrderQuery, "sides" | "actions" | "state">;

/**
 * The gap ledger row a result lists when a Mega Evolution is involved (the
 * game has a known issue with simultaneous Mega order). `checkMechanicsConfig`
 * requires it in the ledger whenever `speed_order` is in scope.
 */
export const MEGA_GAP_ID = "mega-evolution";

/** True when the query evolves a Pokemon this turn or declares one already evolved. */
export const megaInvolved = (query: Pick<DeclaredQuery, "sides" | "actions">): boolean => query.actions.some((action) => action.mega) || query.sides.some((side) => side.some((combatant) => combatant.mega));

// The names the query gives a side's conditions, and the engine's ids for them.
const SIDE_CONDITIONS = [
  ["aurora_veil", "auroraveil"],
  ["light_screen", "lightscreen"],
  ["reflect", "reflect"],
  ["tailwind", "tailwind"],
] as const;
const STAGES = ["atk", "def", "spa", "spd", "spe"] as const;
const TRICK_ROOM = "trickroom";

/** The parts of the engine's objects this probe reads, narrowed here so the shared engine types stay as they are. */
export interface Mon {
  boosts: Record<string, number>;
  status: string;
  hp: number;
  maxhp: number;
  canMegaEvo: string | null | false;
  faint(): unknown;
  setStatus(status: string, source: null, sourceEffect: null, ignoreImmunities: boolean): unknown;
  sethp(hp: number): unknown;
  item: string;
  set: { item: string };
  itemState: unknown;
}
export interface Queued {
  choice: string;
  order: number;
  priority?: number;
  speed: number;
}
interface Condition {
  exists: boolean;
  effectType: string;
}
export interface ProbeBattle {
  prng: object;
  sides: { pokemon: Mon[]; addSideCondition(id: string, source: Mon): unknown; getSideCondition(id: string): unknown; removeSideCondition(id: string): unknown }[];
  field: {
    weather: string;
    terrain: string;
    setWeather(id: string, source: Mon): unknown;
    clearWeather(): unknown;
    setTerrain(id: string, source: Mon): unknown;
    clearTerrain(): unknown;
    addPseudoWeather(id: string, source: Mon): unknown;
    getPseudoWeather(id: string): unknown;
    removePseudoWeather(id: string): unknown;
  };
  queue: { resolveAction(action: object): Queued[] };
  actions: { runMegaEvo(pokemon: Mon): boolean };
  initEffectState(state: { id: string; target: Mon }): unknown;
  dex: { items: { get(name: string): { id: string } }; moves: { get(id: string): { exists: boolean } }; conditions: { get(id: string): Condition } };
  faintMessages(): unknown;
  getRandomTarget(...args: unknown[]): unknown;
  getActionSpeed(action: Queued): void;
  comparePriority(a: Queued, b: Queued): number;
}

/**
 * The two teams of the scenario. A doubles battle needs two Pokemon a side to
 * start, so a side with one declared Pokemon also starts a copy of it; the
 * copy is fainted before anything is declared (`retireCopies`), so it is no
 * active Pokemon: it never acts and never ties a declared Pokemon's Speed.
 */
export function teams(query: DeclaredQuery, sets: PokemonSet[]): { p1: PokemonSet[]; p2: PokemonSet[] } {
  const side = (start: number, length: number): PokemonSet[] => {
    const own = sets.slice(start, start + length);
    return own.length === 1 ? [own[0] as PokemonSet, own[0] as PokemonSet] : own;
  };
  const first = query.sides[0].length;
  return { p1: side(0, first), p2: side(first, query.sides[1].length) };
}

/** Faint the copy of every side with one declared Pokemon, as the engine does for any Pokemon that faints. */
export function retireCopies(battle: ProbeBattle, query: DeclaredQuery): void {
  for (const [index, combatants] of query.sides.entries()) {
    if (combatants.length === 1) (battle.sides[index]?.pokemon[1] as Mon).faint();
  }
  battle.faintMessages();
}

/**
 * The declared field, absolute: every weather, terrain, Trick Room and side condition is cleared first and then set as declared.
 * It runs once before the Evolutions and again after them, because a Mega's ability (Drought, Electric Surge) replaces the field.
 */
function declareField(battle: ProbeBattle, query: DeclaredQuery, lead: Mon): void {
  const { state } = query;
  const kindOf = (id: string): string => (battle.dex.conditions.get(id).exists ? battle.dex.conditions.get(id).effectType : "none");
  battle.field.removePseudoWeather(TRICK_ROOM);
  for (const side of battle.sides) for (const [, id] of SIDE_CONDITIONS) side.removeSideCondition(id);
  if (state.weather === null) battle.field.clearWeather();
  else {
    need(kindOf(state.weather) === "Weather", "the declared weather is not an engine weather");
    battle.field.setWeather(state.weather, lead);
    need(battle.field.weather === state.weather, "the engine refused the declared weather");
  }
  if (state.terrain === null) battle.field.clearTerrain();
  else {
    need(kindOf(state.terrain) === "Terrain", "the declared terrain is not an engine terrain");
    battle.field.setTerrain(state.terrain, lead);
    need(battle.field.terrain === state.terrain, "the engine refused the declared terrain");
  }
  if (state.trick_room) {
    battle.field.addPseudoWeather(TRICK_ROOM, lead);
    need(Boolean(battle.field.getPseudoWeather(TRICK_ROOM)), "the engine refused Trick Room");
  }
  for (const [index, side] of battle.sides.entries()) {
    for (const [name, id] of SIDE_CONDITIONS) {
      if (!state.sides[index]?.[name]) continue;
      side.addSideCondition(id, side.pokemon[0] as Mon);
      need(Boolean(side.getSideCondition(id)), `the engine refused ${id}`);
    }
  }
}

/** Put the declared state on the battle. Every value is absolute: what a switch-in effect set is replaced. */
export function declare(battle: ProbeBattle, query: DeclaredQuery, at: (position: number) => void): void {
  const kindOf = (id: string): string => (battle.dex.conditions.get(id).exists ? battle.dex.conditions.get(id).effectType : "none");
  const lead = battle.sides[0]?.pokemon[0] as Mon;
  const declareStatus = (pokemon: Mon, status: string): void => {
    need(kindOf(status) === "Status", "the declared status is not an engine status");
    pokemon.setStatus(status, null, null, true); // observed fact: type immunities do not apply
    need(pokemon.status === status, "the engine refused the declared status");
  };
  // A status the player saw before the field changed is an observed fact too: Misty Terrain and Electric Terrain
  // refuse it once they are up, so a Pokemon that does not Mega Evolve first tries to take its status here, before
  // the declared field. Anything but success changes nothing: a refusal leaves the Pokemon as it was and the old
  // place below (after the field, after its Evolution) decides, so a switch-in field that blocks the status here
  // cannot refuse a query the old order answered. A random draw is kept and raised at that old place, so the
  // position it reports is unchanged. A Pokemon that Mega Evolves keeps the old order only: its new ability may
  // change the field and the status.
  const taken = new Set<number>();
  let drawn: { place: number; error: unknown } | null = null;
  let numbered = 0;
  for (const [index, combatants] of query.sides.entries()) {
    for (const [slot, combatant] of combatants.entries()) {
      const place = numbered++;
      if (combatant.status === null || combatant.mega || drawn !== null) continue;
      at(place);
      try {
        declareStatus(battle.sides[index]?.pokemon[slot] as Mon, combatant.status);
        taken.add(place);
      } catch (error) {
        if (error instanceof RandomDraw) drawn = { place, error };
      }
    }
  }
  at(0);
  declareField(battle, query, lead);
  let position = 0;
  for (const [index, combatants] of query.sides.entries()) {
    for (const [slot, combatant] of combatants.entries()) {
      const place = position++;
      at(place);
      const pokemon = battle.sides[index]?.pokemon[slot] as Mon;
      if (combatant.mega) need(battle.actions.runMegaEvo(pokemon), "the Pokemon cannot Mega Evolve");
      if (combatant.status !== null) {
        if (drawn?.place === place) throw drawn.error;
        if (!taken.has(place)) declareStatus(pokemon, combatant.status);
      }
      if (combatant.hp !== null) {
        need(combatant.hp <= pokemon.maxhp, "the declared HP exceeds the maximum");
        pokemon.sethp(combatant.hp);
      }
    }
  }
  // A Mega's ability can replace the declared field: it is declared again, absolute (#1243).
  if (query.sides.some((combatants) => combatants.some((combatant) => combatant.mega))) {
    at(0);
    declareField(battle, query, lead);
  }
  restoreStagesAndItems(battle, query);
}

/**
 * The declared stages and items, absolute, written after every Evolution: a Mega's ability can change the other combatants' stages (Intimidate), and the declared stages are absolute (#1243).
 * Declared history can start a volatile with a stage effect (Stockpile), so it runs again after the history is declared.
 */
export function restoreStagesAndItems(battle: ProbeBattle, query: DeclaredQuery): void {
  for (const [index, combatants] of query.sides.entries()) {
    for (const [slot, combatant] of combatants.entries()) {
      const pokemon = battle.sides[index]?.pokemon[slot] as Mon;
      // An Evolution's stage drop can consume the declared item (White Herb): it is put back (#1243).
      // Put back silently, not through `setItem`: its Start event would fire a Seed item again under the declared terrain.
      const itemId = battle.dex.items.get(pokemon.set.item).id;
      if (pokemon.item !== itemId) {
        pokemon.item = itemId;
        pokemon.itemState = battle.initEffectState({ id: itemId, target: pokemon });
      }
      for (const stage of STAGES) pokemon.boosts[stage] = combatant.boosts[stage] as number;
    }
  }
}

/** Sort by the engine's comparison and cut at every pair it calls equal. */
export function groupsOf(battle: ProbeBattle, items: { index: number; action: Queued }[]): number[][] {
  const sorted = [...items].sort((a, b) => battle.comparePriority(a.action, b.action));
  const groups: { index: number; action: Queued }[][] = [];
  for (const item of sorted) {
    const last = groups[groups.length - 1];
    if (last !== undefined && battle.comparePriority((last[0] as { action: Queued }).action, item.action) === 0) last.push(item);
    else groups.push([item]);
  }
  return groups.map((group) => group.map((item) => item.index));
}

/** One declared action as the engine queued it: its Pokemon, its own queue entry and its Mega Evolution entry, if any. */
export interface ResolvedAction {
  index: number;
  pokemon: Mon;
  own: Queued;
  mega: Queued | undefined;
}

/** The engine's own queue entries for the declared actions (`resolveAction`: speed, priority events, fractional priority). */
export function resolveActions(battle: ProbeBattle, query: DeclaredQuery, at: (index: number) => void): ResolvedAction[] {
  return query.actions.map((action, index) => {
    at(index);
    const pokemon = battle.sides[action.side]?.pokemon[action.position] as Mon;
    need(action.move === null || battle.dex.moves.get(action.move).exists, "the action names a move the engine does not have");
    need(!action.mega || (action.move !== null && Boolean(pokemon.canMegaEvo)), "the Pokemon cannot Mega Evolve");
    // A target is given so the engine does not draw one; ordering never reads it beyond the move's own priority events.
    const choice = action.move === null ? { choice: "switch", pokemon } : { choice: "move", pokemon, moveid: action.move, targetLoc: 1, mega: action.mega };
    const queued = battle.queue.resolveAction(choice);
    const own = queued[queued.length - 1] as Queued;
    const megas = queued.filter((item) => item.choice === "megaEvo");
    need(queued.length === megas.length + 1 && own.choice === (action.move === null ? "switch" : "move"), "the action needs more than a move and a Mega Evolution");
    return { index, pokemon, own, mega: megas[0] };
  });
}

/** Run the Mega Evolutions in the engine's order (speed groups of the Evolution entries) and return those groups. */
export function evolve(battle: ProbeBattle, resolved: ResolvedAction[], at: (index: number) => void): number[][] {
  const megaGroups = groupsOf(
    battle,
    resolved.flatMap((item) => (item.mega === undefined ? [] : [{ index: item.index, action: item.mega }])),
  );
  for (const group of megaGroups) {
    for (const index of group) {
      at(index);
      need(battle.actions.runMegaEvo((resolved[index] as { pokemon: Mon }).pokemon), "the Pokemon cannot Mega Evolve");
    }
  }
  return megaGroups;
}

/**
 * Answer a `speed_order` query on a fresh scenario battle. `sets` are the
 * query's declared sets in the contract's position order, each already judged
 * by the target format's validator. Returns `computed`, or `unsupported`
 * (`random_effect`) when the engine draws a random number; throws a
 * `MechanicsProbeError` for a declared state or action the engine refuses.
 */
export function probeSpeedOrder(engine: Engine, scenarioId: string, query: SpeedOrderQuery, sets: PokemonSet[], history: HistoryDeclaration | null = null): Outcome {
  const battle = startScenario(engine, scenarioId, teams(query, sets)) as unknown as ProbeBattle;
  retireCopies(battle, query);
  const refusing = refusingSource(battle.prng);
  battle.prng = refusing;
  const resolveTarget = battle.getRandomTarget.bind(battle);
  battle.getRandomTarget = (...args) => {
    battle.prng = firstTarget;
    try {
      return resolveTarget(...args);
    } finally {
      battle.prng = refusing;
    }
  };
  let at = 0;
  try {
    declare(battle, query, (position) => (at = position));
    if (history !== null) {
      declareHistory(battle as unknown as HBattle, history.resolved, { stall: history.stall, at: (position) => (at = position), on_draw: history.on_draw });
      restoreStagesAndItems(battle, query);
    }
    const resolved = resolveActions(battle, query, (index) => (at = index));
    const megaGroups = evolve(battle, resolved, (index) => (at = index));
    for (const item of resolved) {
      at = item.index;
      battle.getActionSpeed(item.own);
    }
    // Order values decide first (switch 103, Evolution 104, move 200), so switches, then the Evolutions, then the moves.
    const ofChoice = (choice: string) =>
      groupsOf(
        battle,
        resolved.filter((item) => item.own.choice === choice).map((item) => ({ index: item.index, action: item.own })),
      );
    const switchGroups = ofChoice("switch");
    const moveGroups = ofChoice("move");
    const actions = query.actions.map((action, index) => {
      const own = (resolved[index] as { own: Queued }).own;
      const priority = own.priority ?? 0;
      need(Number.isInteger(priority), "the action's priority is not a whole number");
      return { side: action.side, position: action.position, mega: action.mega, order: own.order, priority, speed: own.speed };
    });
    const answer = { kind: "speed_order", order_basis: "turn_start", actions, groups: [...switchGroups, ...megaGroups, ...moveGroups] };
    return { kind: "computed", answer: answer as Extract<Outcome, { kind: "computed" }>["answer"] };
  } catch (error) {
    if (error instanceof RandomDraw) return { kind: "unsupported", reason: "random_effect", position: at };
    throw error;
  }
}
