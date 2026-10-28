// The part of the damage-aware baseline that reads one request into probe queries (ADR-018 sections 2 to 4),
// shared by the baseline (`damage-aware-actor.ts`) and the candidate feature builder
// (`candidate-features.ts`, ADR-022 section 3): the player's own Pokemon as probe combatants, the opposing
// actives' assumed sets, the declared field, and one probe query with a per-game memo. Nothing here scores
// anything or knows a game rule: the numbers are the probe's, the ids are the engine's.
//
// Excluded by construction, as in the baseline: the real opponent team, the omniscient log, hidden verdicts
// and the seed are not parameters of anything here.

import type { PublicMon, PublicState } from "@areneon/public-state";
import { type AssumedSet, assumedHp, buildAssumedSet } from "./damage-aware-assumed-set.ts";
import type { AnyDamageAwareConfig, DamageAwareConfig } from "./damage-aware-config.ts";
import type { DamageAnswer } from "./damage-aware-score.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome } from "./mechanics-probe.ts";
import { type MechanicsQuery, mechanicsQueryHash, type ParsedMechanicsQuery, readMechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import type { MechanicsResult } from "./mechanics-result.ts";
import type { RulesSnapshot } from "./rules.ts";
import { exportPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The caller gave something that cannot be read: a missing context or team, an unknown request shape, a Pokemon that matches no set. It fails closed. */
export class DamageAwareError extends Error {
  override name = "DamageAwareError";
}

/** What this module reads of the engine's data, narrowed locally. */
export interface ScoreDex {
  moves: { get(id: string): { exists: boolean; id: string; category: string; target: string } };
  items: { get(id: string): { exists: boolean; name: string } };
  species: { get(id: string): { exists: boolean; baseSpecies: string } };
  conditions: { get(id: string): { exists: boolean; effectType: string } };
}
type Json = unknown;
export const object = (value: Json, what: string): Record<string, Json> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new DamageAwareError(`${what} is not an object`);
  return value as Record<string, Json>;
};
export const array = (value: Json, what: string): Json[] => {
  if (!Array.isArray(value)) throw new DamageAwareError(`${what} is not an array`);
  return value;
};
export const text = (value: Json, what: string): string => {
  if (typeof value !== "string") throw new DamageAwareError(`${what} is not a string`);
  return value;
};

/** Target types of a request's move, by what the scorer does with them (the engine's `MoveTarget` names; any other fails closed). */
// The id strings below (target types, the four side conditions, Trick Room, the Terrain effect type, the Status
// category, `fnt`) are the field names of the probe's `DamageState` contract and of the engine's own data, not
// rules of this module. An id the engine's request carries that is in none of the target sets fails closed.
const SINGLE = new Set(["normal", "any", "adjacentFoe"]);
const RANDOM_FOE = new Set(["randomNormal"]);
const EVERY_FOE = new Set(["allAdjacentFoes", "allAdjacent"]);
const NO_FOE = new Set(["self", "adjacentAlly", "adjacentAllyOrSelf", "allySide", "allyTeam", "allies", "foeSide", "all", "scripted"]);
/** The probe's refusal of an action it cannot order (`mechanics-speed.ts`): the one speed-order failure that is a known limit, not a defect. */
const UNQUEUEABLE = "the action needs more than a move and a Mega Evolution";
/** The probe's other speed-order limit: a fractional priority (Quick Claw, Custap Berry and the like), which its integer contract cannot state. */
const FRACTIONAL = "the action's priority is not a whole number";
const CONDITION = /^(\d+)(?:\/(\d+))?(?: ([a-z]+))?$/;

/** One Pokemon as a probe combatant, with the pieces a query needs. */
export interface Fighter {
  paste: string;
  /** Current hit points, or null for full. */
  hp: number | null;
  status: string | null;
  boosts: Record<"atk" | "def" | "spa" | "spd" | "spe", number>;
  mega: boolean;
}
export interface OwnSlot {
  fighter: Fighter;
  name: string;
}
export interface Foe {
  mon: PublicMon;
  assumed: AssumedSet;
  fighter: Fighter;
  /**
   * The foe's location number in a choice (`move 1 1`). Engine convention (`Pokemon.getAtLoc`): location L is the foe in
   * active slot L - 1 (location 1 is slot `a`). Legacy convention (`damage-aware@v1` and `@v2`, bug-for-bug, #1180): the reverse.
   */
  loc: number;
}

/** The reader's state does not show an own Pokemon the request lists: the request plays the `actor-rule` candidate and is counted. */
export class ReaderGap extends Error {
  override name = "ReaderGap";
}

/** The probe raised a `MechanicsProbeError` for a query the caller built: this request is answered without it. */
export class ProbeRefusal extends Error {
  override name = "ProbeRefusal";
}
/** The probe's words for a declared status the engine will not take; an unrevealed ability that blocks the status is then contradicted by it. */
const STATUS_REFUSED = "the engine does not take that status";
/** The probe's refusal of a query for a format its config is not for: a defect of the run, so it propagates. */
const CONFIG_DEFECT = "the mechanics config is for another format";

type DamageState = Extract<MechanicsQuery["query"], { kind: "damage" }>["state"];
type Combatant = Extract<MechanicsQuery["query"], { kind: "damage" }>["attacker"];

/** The probe's two speed-order limits (ADR-022 erratum E-22.3): an action it cannot queue, and one whose priority is not a whole number. */
export type SpeedLimit = "unqueueable" | "fractional_priority";

/** The probe's answer to a query, remembered by its hash: `limit` is a speed order the probe cannot take; `refused` a state the engine does not take. */
export type ProbeMemo = Map<string, MechanicsResult | { limit: SpeedLimit } | { refused: string }>;

/** The counts a view keeps (the baseline's report has more). */
export interface ProbeCounts {
  /** Probe queries asked (a repeated query counts each time it is asked). */
  queries: number;
  /** Probe runs actually made (queries not answered from the memo). */
  probe_runs: number;
  /** Queries the probe answered `unsupported`. */
  unsupported_queries: number;
}

export interface DamageViewOptions {
  engine: Engine;
  vocabulary: TeamVocabulary;
  config: AnyDamageAwareConfig;
  /** The v1 record, whose field rules build assumed sets (`assumedFor`); null for a record that names a belief. */
  v1: DamageAwareConfig | null;
  mechanics: MechanicsConfig;
  snapshot: RulesSnapshot;
  data_class: "synthetic" | "private";
  probe: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
  /**
   * Which foe a choice location names: true is the pinned engine's convention (`getAtLoc`: location L is the foe in
   * active slot L - 1); absent or false keeps the legacy reversed numbering that `damage-aware@v1` and `@v2` played (#1180).
   */
  engine_targets?: boolean;
  /** Test seam (v1 only): the assumed opponent set of a public Pokemon (default `buildAssumedSet`). */
  assume?: (mon: PublicMon) => AssumedSet;
  counts: ProbeCounts;
  memo: ProbeMemo;
}

/**
 * The foes a move part hits, by the request's target type: one by location, every one, or (a random target) every one
 * for the caller to average. `foes` is called only for those target types, so a request that needs no foe builds none (and asks the probe nothing). A locked move (Recharge, the second turn of Dig) has no target type: it scores 0.
 */
export function foesHit(type: string | undefined, target: number | null, foes: () => readonly Foe[]): Foe[] {
  if (type === undefined) return [];
  if (SINGLE.has(type)) {
    if (target === null) throw new DamageAwareError("a move that needs a target has none");
    return foes().filter((foe) => foe.loc === target);
  }
  if (EVERY_FOE.has(type) || RANDOM_FOE.has(type)) return [...foes()];
  if (!NO_FOE.has(type)) throw new DamageAwareError("a move target type the baseline does not know");
  return [];
}

/** What a move's request target type hits among the foes: one by location, a random one, every one, or none (the baseline's sets). */
export function foeTargetKind(type: string | undefined): "single" | "random" | "every" | "none" {
  if (type === undefined) return "none";
  if (SINGLE.has(type)) return "single";
  if (RANDOM_FOE.has(type)) return "random";
  if (EVERY_FOE.has(type)) return "every";
  if (!NO_FOE.has(type)) throw new DamageAwareError("a move target type the baseline does not know");
  return "none";
}

/** True for a move whose target the engine draws (the baseline averages it over the foes). */
export const isRandomFoe = (type: string | undefined): boolean => type !== undefined && RANDOM_FOE.has(type);

export function damageView(options: DamageViewOptions) {
  const { engine, vocabulary, config, v1, mechanics, snapshot, data_class, probe, counts, memo } = options;
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(config.format_id).mod) as unknown as ScoreDex;

  /**
   * One probe answer, or `{ limit }` for a speed order the probe cannot take as a query (an action that
   * queues more than a move and a Mega Evolution, as Focus Punch does, or one with a fractional priority): counted as `unsupported`.
   * Any other `MechanicsProbeError` throws `ProbeRefusal` (the request falls back); every other failure is a defect and propagates.
   */
  function askLimited(value: MechanicsQuery): MechanicsResult | { limit: SpeedLimit } {
    const query = readMechanicsQuery(value);
    const key = mechanicsQueryHash(query);
    counts.queries += 1;
    let result = memo.get(key);
    if (result === undefined) {
      counts.probe_runs += 1;
      try {
        result = probe({ engine, config: mechanics, snapshot }, query).result;
      } catch (error) {
        if (!(error instanceof MechanicsProbeError) || error.message.startsWith(CONFIG_DEFECT)) throw error; // a wrong config is a defect of the run, not of a state
        if (query.query.kind === "speed_order" && error.message.startsWith(UNQUEUEABLE)) result = { limit: "unqueueable" };
        else if (query.query.kind === "speed_order" && error.message.startsWith(FRACTIONAL)) result = { limit: "fractional_priority" };
        else {
          // Any other refusal of a state the caller declared: this request falls back.
          memo.set(key, { refused: error.message });
          throw new ProbeRefusal(error.message);
        }
      }
      memo.set(key, result);
    }
    if ("limit" in result) {
      counts.unsupported_queries += 1;
      return result;
    }
    if ("refused" in result) throw new ProbeRefusal(result.refused); // a memoised refusal (the query was asked again)
    if (result.kind === "refused") throw new DamageAwareError("the probe refused a set the baseline built");
    if (result.kind === "unsupported") counts.unsupported_queries += 1;
    return result;
  }
  /** `askLimited` with a speed-order limit as `null` (no discount is applied). */
  function ask(value: MechanicsQuery): MechanicsResult | null {
    const result = askLimited(value);
    return "limit" in result ? null : result;
  }
  const wrap = (body: MechanicsQuery["query"]): MechanicsQuery => ({ schema_version: 1, data_class, format_id: config.format_id, query: body }) as MechanicsQuery;
  const query = (body: MechanicsQuery["query"]): MechanicsResult | null => ask(wrap(body));
  /** As `query`, with the probe's speed-order limit named instead of `null`. */
  const queryLimited = (body: MechanicsQuery["query"]): MechanicsResult | { limit: SpeedLimit } => askLimited(wrap(body));

  /** The probe's whole answer to one damage query (a `computed` or `unsupported` result); a memoised refusal throws `ProbeRefusal`. */
  const damageResult = (attacker: Fighter, defender: Fighter, move: string, state: PublicState, attackerIsOwn: boolean): MechanicsResult | null =>
    query({ kind: "damage", attacker: combatant(attacker), defender: combatant(defender), move, state: field(state, attackerIsOwn) });

  const damageOf = (attacker: Fighter, defender: Fighter, move: string, state: PublicState, attackerIsOwn: boolean): DamageAnswer | null => {
    const result = damageResult(attacker, defender, move, state, attackerIsOwn);
    return result?.kind === "computed" && result.answer.kind === "damage" ? result.answer : null;
  };
  const maxHp = new Map<string, number>();
  const maxHpOf = (paste: string): number => {
    let hp = maxHp.get(paste);
    if (hp === undefined) {
      const result = query({ kind: "stats", set: paste });
      if (result?.kind !== "computed" || result.answer.kind !== "stats") throw new DamageAwareError("the probe could not give an assumed set's maximum hit points");
      hp = result.answer.stats.hp;
      maxHp.set(paste, hp);
    }
    return hp;
  };

  function combatant(fighter: Fighter): Combatant {
    return { set: fighter.paste, boosts: { ...fighter.boosts }, status: fighter.status, hp: fighter.hp, mega: fighter.mega };
  }

  /** The declared field, the attacker's side first (a query's `sides` are the attacker's, then the defender's). */
  function field(state: PublicState, attackerIsOwn: boolean): DamageState {
    const [first, second] = attackerIsOwn ? [state.own, state.opponent] : [state.opponent, state.own];
    const sideOf = (side: PublicState["own"]): DamageState["sides"][number] => ({
      aurora_veil: "auroraveil" in side.conditions,
      light_screen: "lightscreen" in side.conditions,
      reflect: "reflect" in side.conditions,
      tailwind: "tailwind" in side.conditions,
    });
    const terrain = state.terrain ?? state.field_conditions.find((id) => dex.conditions.get(id).effectType === "Terrain") ?? null;
    return {
      weather: state.weather,
      terrain,
      trick_room: state.field_conditions.includes("trickroom"),
      sides: [sideOf(first), sideOf(second)],
    };
  }

  const boostsOf = (mon: PublicMon): Fighter["boosts"] => ({ atk: mon.boosts.atk ?? 0, def: mon.boosts.def ?? 0, spa: mon.boosts.spa ?? 0, spd: mon.boosts.spd ?? 0, spe: mon.boosts.spe ?? 0 });

  /**
   * The player's own Pokemon at index `slot` of the request's `side.pokemon`, from its own request and own team. A `bench`
   * Pokemon (not in play) is read from the request and the team alone: stat stages 0, and Mega as the reader's roster shows it
   * (not Mega when the reader has never seen it), so a Pokemon never yet in play is no `ReaderGap`.
   */
  function ownSlot(side: Record<string, Json>, team: readonly Readonly<PokemonSet>[], state: PublicState, slot: number, bench = false): OwnSlot {
    const mon = object(array(side.pokemon, "side.pokemon")[slot], "an active Pokemon");
    const ident = text(mon.ident, "ident");
    const name = ident.slice(ident.indexOf(": ") + 2);
    const matches = team.filter((set) => [set.name, set.name === "" ? set.species : "", dex.species.get(set.species).baseSpecies].includes(name));
    if (matches.length !== 1) throw new DamageAwareError("an own Pokemon matches no set, or more than one, of the team as brought");
    const shown = state.own.roster.find((candidate) => candidate.name === name);
    if (shown === undefined && !bench) throw new ReaderGap("the reader does not show an own Pokemon the request lists");
    const condition = CONDITION.exec(text(mon.condition, "condition"));
    if (condition === null) throw new DamageAwareError("an own Pokemon's condition has a shape the baseline does not know");
    const [, hp, max, status] = condition;
    const item = text(mon.item, "item");
    const set: PokemonSet = { ...structuredClone(matches[0] as PokemonSet), item: item === "" ? "" : dex.items.get(item).name };
    return {
      name,
      fighter: {
        paste: exportPaste(engine, [set]),
        hp: max === undefined || Number(hp) === Number(max) ? null : Number(hp),
        status: status === undefined || status === "fnt" ? null : status,
        boosts: shown === undefined || bench ? { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } : boostsOf(shown),
        mega: shown?.mega ?? false,
      },
    };
  }

  /** A benched own Pokemon: the one at index `index` of the request's `side.pokemon`, read from the request and team only. */
  const ownBench = (request: unknown, state: PublicState, team: readonly Readonly<PokemonSet>[], index: number): OwnSlot => ownSlot(object(object(request, "request").side, "side"), team, state, index, true);

  /** The player's own actives of one request, by request slot. */
  const ownSlots = (request: unknown, state: PublicState, team: readonly Readonly<PokemonSet>[]): OwnSlot[] => {
    const body = object(request, "request");
    const side = object(body.side, "side");
    return array(body.active, "active").map((_, slot) => ownSlot(side, team, state, slot));
  };

  /**
   * The assumed set of an opponent's Pokemon, consistent with its public status: the status is a fact the
   * player saw, so an unrevealed ability that blocks it (Limber on a paralysed Pokemon) cannot be the one
   * the Pokemon has. Abilities are tried in the record's order and the engine, through a probe query that
   * declares the status, says which one blocks it. Throws `AssumedSetError` when every ability does. The
   * refusal can also come from an engine hook that is no ability (a clause, a side condition); then the
   * next ability is refused alike and the request ends in the `assumed_set_fallbacks` count, never a crash.
   * Its probe queries add to `queries` and `probe_runs` for every request with a statused foe.
   */
  function assumedFor(mon: PublicMon, state: PublicState): AssumedSet {
    const record = v1 as DamageAwareConfig;
    const assume = options.assume ?? ((shown: PublicMon) => buildAssumedSet({ engine, vocabulary, config: record }, shown));
    let set = assume(mon);
    if (mon.status === null || mon.ability !== null || options.assume !== undefined) return set;
    const ruledOut: string[] = [];
    while (statusBlocked(set, mon, state)) {
      ruledOut.push(set.set.ability);
      set = buildAssumedSet({ engine, vocabulary, config: record }, mon, ruledOut);
    }
    return set;
  }

  /**
   * True when the engine refuses the Pokemon's public status on this assumed set (its ability blocks it): the
   * status is a fact the player saw, so such a set cannot be the Pokemon. Any other refusal is not about the ability:
   * scoring meets it and falls back.
   */
  function statusBlocked(set: AssumedSet, mon: PublicMon, state: PublicState): boolean {
    const fighter: Fighter = { paste: set.paste, hp: null, status: null, boosts: boostsOf(mon), mega: set.mega };
    const move = dex.moves.get(set.set.moves[0] as string).id;
    try {
      damageOf(fighter, { ...fighter, status: mon.status }, move, state, false);
      return false;
    } catch (error) {
      return error instanceof ProbeRefusal && error.message.startsWith(STATUS_REFUSED);
    }
  }

  /**
   * The opposing active Pokemon with their assumed sets, by choice location. `assumedOf` gives the set of the
   * Pokemon at a roster position: v1's field rules, or a sampled world's set. Throws `AssumedSetError` when none can be built.
   */
  function foesOf(state: PublicState, assumedOf: (mon: PublicMon, position: number) => AssumedSet): Foe[] {
    const letters = Object.keys(state.opponent.slots).sort();
    const out: Foe[] = [];
    letters.forEach((letter, position) => {
      const index = state.opponent.slots[letter];
      const mon = index === null || index === undefined ? undefined : state.opponent.roster[index];
      if (mon === undefined || mon.fainted) return;
      const assumed = assumedOf(mon, index as number);
      const percent = assumed.shown_hp_percent;
      if (percent === null || percent === 0) return;
      const max = maxHpOf(assumed.paste);
      const hp = assumedHp(percent, max, config.hp_rounding);
      const fighter: Fighter = { paste: assumed.paste, hp: hp === max ? null : hp, status: mon.status, boosts: boostsOf(mon), mega: assumed.mega };
      out.push({ mon, assumed, fighter, loc: options.engine_targets === true ? position + 1 : letters.length - position });
    });
    return out;
  }

  return { dex, query, queryLimited, damageResult, damageOf, combatant, field, boostsOf, ownSlots, ownBench, assumedFor, statusBlocked, foesOf };
}

export type DamageView = ReturnType<typeof damageView>;
