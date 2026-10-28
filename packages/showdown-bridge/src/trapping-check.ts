// The engine-backed trapping check of the opponent belief (ADR-025 section 3, ticket #1127; asked by ADR-026 N-I).
//
// The player's own move request carries `trapped` and `maybeTrapped` for its active Pokemon, and both are player
// evidence about the opponent's abilities. What the engine shows (read from the pinned source, and decided by the
// scenario below, never by a rule written here): for the last living active, `trapped: true` only for a trap that
// is not hidden and `maybeTrapped` when a foe's species lists a trapping ability (the held slot is skipped, so a
// foe that holds the trapping ability and one that does not differ there); for an earlier active, `trapped: true`
// whenever any trap works on it, a foe's hidden trap included. This module is the port the belief reads
// (`TrappingCheck` of `packages/opponent-model`, declared structurally here because the bridge does not import the
// model): for a candidate foe (the species it is shown as and an ability) it starts a scenario battle of the pinned
// engine in which the player's own side is the real one (the own team as brought, in the request's order, its items,
// abilities, formes and fainted Pokemon as the request shows them) and both foe actives are the candidate, runs the
// engine's own turn rollover and reads the flags of the own request that the engine makes. Per query, memoised per
// candidate.
//
// Soundness: the belief applies the answer as a hard constraint, so the check declines (the candidate then shows
// what was observed, which constrains nothing) whenever the scenario could differ from the real battle in a way that
// matters. It returns null, for no information at all, when the request is not a move request, a field condition is
// in play (Gravity, Magic Room and the like change who is grounded), the own team cannot be matched to the request,
// or no own active carries information. An own active is informative only when its request says no `trapped: true`
// (a Mean Look, a locked move or a recharge raise it of their own, and the reader does not show every such volatile,
// so the flag has causes the scenario does not hold) and the reader shows it with no volatile (Magnet Rise and Ingrain
// change who is grounded). A candidate is declined when the scenario cannot be built or a switch-in effect of the start changed
// an ability or item that the request or the candidate declares.

import type { PublicState } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { type Engine, type PokemonSet, defaultIvs } from "./engine.ts";
import { startScenario } from "./mechanics-scenario.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import type { WBattle, WMon, WSide } from "./world-engine.ts";
import { toId } from "./world-engine.ts";

/** Bumped when the check changes how it builds the scenario or reads a flag. */
export const TRAPPING_CHECK_FORMAT = 1;

/** What an own active shows of the trapping flags: 0 nothing, 1 `maybeTrapped`, 2 `trapped: true`. */
export type TrapLevel = 0 | 1 | 2;

export interface TrapCandidate {
  readonly species: string;
  readonly ability: string;
  readonly item: string | null;
}

/** Structurally the opponent model's `TrappingCheck`. */
export interface TrappingCheckPort {
  readonly observed: readonly TrapLevel[];
  shows(candidate: TrapCandidate): readonly TrapLevel[];
}

export interface TrappingCheckContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  /** The scenario format id (`checkScenario`). */
  readonly scenario_id: string;
}

export interface TrappingCheckInput {
  /** The player's own move request, as the engine sent it. */
  readonly request: EngineRequest;
  /** The player's own team as brought. */
  readonly team: readonly Readonly<PokemonSet>[];
  /** The shared reader's state at the request (own volatiles and field conditions only). */
  readonly state: PublicState;
}

const SEATS = ["p1", "p2"] as const;
const SHOWN_GENDER = /, ([MF])(?:,|$)/;

export function createTrappingCheck(context: TrappingCheckContext, input: TrappingCheckInput): TrappingCheckPort | null {
  const { engine, vocabulary, scenario_id } = context;
  const { request, team, state } = input;
  if (request.active === undefined || request.wait === true || request.teamPreview === true || request.forceSwitch !== undefined) return null;
  if (state.field_conditions.length > 0) return null;
  const seat = SEATS.indexOf(request.side.id as "p1" | "p2");
  if (seat < 0 || request.side.id !== state.player) return null;
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(vocabulary.format_id).mod);
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));

  // The own side from the request, as the world builder reads it.
  const used = new Set<PokemonSet>();
  const ownSets: PokemonSet[] = [];
  for (const entry of request.side.pokemon) {
    const name = entry.ident.slice(entry.ident.indexOf(": ") + 2);
    const matches = team.filter((set) => (set.name === "" ? set.species : set.name) === name);
    if (matches.length !== 1 || used.has(matches[0] as PokemonSet)) return null;
    used.add(matches[0] as PokemonSet);
    const set = structuredClone(matches[0] as PokemonSet) as PokemonSet;
    const gender = SHOWN_GENDER.exec(entry.details)?.[1] ?? "";
    if (gender !== "") set.gender = gender;
    // The request's item and ability are the current ones (an item knocked off or a Trace is in the request).
    set.item = entry.item === "" ? "" : dex.items.get(entry.item).name;
    const ability = entry.ability ?? entry.baseAbility;
    set.ability = dex.abilities.get(ability).name;
    ownSets.push(set);
  }
  if (ownSets.length === 0) return null;

  // The own actives whose flags carry information.
  const indices: number[] = [];
  const observed: TrapLevel[] = [];
  request.active.forEach((slot, index) => {
    const entry = request.side.pokemon[index];
    // A request that says `trapped: true` has an own cause the scenario may not hold (the reader does not show every volatile of a lock or a recharge): no information.
    if (entry === undefined || entry.active !== true || entry.condition === "0 fnt" || slot.trapped === true) return;
    const name = entry.ident.slice(entry.ident.indexOf(": ") + 2);
    const shown = state.own.roster.find((mon) => mon.name === name);
    if (shown === undefined || shown.volatiles.length > 0 || shown.fainted) return;
    indices.push(index);
    observed.push(slot.maybeTrapped === true ? 1 : 0);
  });
  if (indices.length === 0) return null;

  const memo = new Map<string, readonly TrapLevel[]>();

  /** The levels the own actives show with this candidate as both foes, or null when the scenario cannot say. */
  function scenario(candidate: TrapCandidate): readonly TrapLevel[] | null {
    const entry = roster.get(candidate.species);
    const nature = vocabulary.natures[0];
    if (entry === undefined || nature === undefined) return null;
    const foe: PokemonSet = {
      name: "",
      species: dex.species.get(candidate.species).name,
      item: candidate.item === null ? "" : dex.items.get(candidate.item).name,
      ability: dex.abilities.get(candidate.ability).name,
      moves: [dex.moves.get(entry.moves[0] as string).name],
      nature: nature.name,
      gender: "",
      evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
      ivs: defaultIvs(),
      level: vocabulary.rules.adjust_level,
    };
    const teams = { p1: [] as PokemonSet[], p2: [] as PokemonSet[] };
    teams[SEATS[seat] as "p1" | "p2"] = ownSets.map((set) => structuredClone(set) as PokemonSet);
    teams[SEATS[1 - seat] as "p1" | "p2"] = ownSets.map(() => structuredClone(foe) as PokemonSet);
    let battle: WBattle;
    try {
      battle = startScenario(engine, scenario_id, teams) as unknown as WBattle;
    } catch {
      return null;
    }
    const side = battle.sides[seat] as WSide;
    if (side.pokemon.length !== ownSets.length) return null;
    // A switch-in effect of the start (Trace, Neutralizing Gas, an item that swaps) may have changed an ability or item that the request or the candidate declares: the scenario is then not the real state.
    const declared = [...ownSets, ...ownSets.map(() => foe)];
    const present = [...side.pokemon, ...(battle.sides[1 - seat] as WSide).pokemon];
    if (present.some((mon, k) => toId(mon.ability) !== toId((declared[k] as PokemonSet).ability) || toId(mon.item) !== toId((declared[k] as PokemonSet).item))) return null;
    // The own formes (a Mega Evolution that has happened) and the fainted Pokemon, as the request shows them.
    const down: WMon[] = [];
    for (const [index, entry] of request.side.pokemon.entries()) {
      const mon = side.pokemon[index] as WMon;
      if (entry.condition === "0 fnt") {
        down.push(mon);
        continue;
      }
      const shown = toId(entry.details.split(",")[0] as string);
      if (mon.species.id === shown) continue;
      if ((dex.species.get(shown) as { isMega?: boolean }).isMega === true) battle.actions.runMegaEvo(mon);
      else mon.formeChange(shown, null, true);
      if (mon.species.id !== shown) return null;
    }
    for (const mon of down) mon.faint();
    if (down.length > 0) battle.faintMessages();
    if (battle.ended) return null;
    battle.endTurn();
    const made = side.activeRequest as { active?: { trapped?: boolean; maybeTrapped?: boolean }[] } | null | undefined;
    if (made?.active === undefined) return null;
    const levels: TrapLevel[] = [];
    for (const index of indices) {
      const slot = made.active[index];
      if (slot === undefined) return null;
      levels.push(slot.trapped === true ? 2 : slot.maybeTrapped === true ? 1 : 0);
    }
    return levels;
  }

  return {
    observed,
    shows(candidate) {
      const key = `${candidate.species}|${candidate.ability}|${candidate.item ?? ""}`;
      let hit = memo.get(key);
      if (hit === undefined) {
        // A candidate the scenario cannot judge shows what was observed: it constrains nothing.
        hit = scenario(candidate) ?? observed;
        memo.set(key, hit);
      }
      return hit;
    },
  };
}
