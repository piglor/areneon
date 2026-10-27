// The damage-aware scripted baseline `damage-aware@<commit>` (ADR-018 sections 1 to 5), an A1 `Actor`.
//
// At each battle request it scores every candidate of the player's own legal choice set by the
// pinned engine's own damage and turn order, asked through the mechanics probe (ADR-016), and plays
// the best non-pending one. It sees only what the player may see: the player's own channel (read by
// the shared public-state reader), the player's own team from `ActorContext.team` and the request.
// The opponent is the public facts of that channel plus the reviewed assumption rules of the
// `DamageAwareConfig` (`buildAssumedSet`), never its real team. No stat, damage or speed number is
// computed here: they are the probe's, scored in integer ppm (`damage-aware-score.ts`).
//
// `damage-aware@v2` (ADR-025 section 6): the same scorer over sampled worlds. Its config record names a belief
// instead of field rules; at each scored request the actor asks the belief port (the opponent model, joined by
// the composition root) for N worlds of the opponent from the reader's opponent half, scores every candidate in
// each world exactly as v1 does and plays the candidate with the best sum (the mean's order). A damage threat is
// still only a revealed move. The worlds' stream is the player's own request, so v2 stays a pure function of the
// player's view. A belief refusal, or no world consistent with the public facts, plays the `actor-rule` candidate
// and is counted.
//
// Excluded by construction: the actor's parameters are the request and an `ActorContext`, whose type
// holds no opponent team, omniscient log, hidden verdict or seed (`runner.ts`); the probe takes only
// a query read from declared sets and state.

import { createInterpreter, type PublicFlag, type PublicInterpreter, type PublicMon, type PublicState, readerSha256 } from "@areneon/public-state";
import { actorRule, NoCandidateError } from "./actor-rule.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";
import { type AssumedSet, AssumedSetError, assumedHp, buildAssumedSet, buildWorldAssumedSet, WorldMismatchError } from "./damage-aware-assumed-set.ts";
import type { AnyDamageAwareConfig, DamageAwareConfig } from "./damage-aware-config.ts";
import { BELIEF_DEFECTS, type BeliefWorldsPort, refusalCode, type WorldLike, worldsStream } from "./damage-aware-worlds.ts";
import { type DamageAwareIdentity, damageAwareIdentity, damageAwareIdentityHash, damageAwarePolicyId } from "./damage-aware-identity.ts";
import { actsBefore, canKnockOut, type DamageAnswer, damageValue, discounted, PPM, type SpeedOrderAnswer } from "./damage-aware-score.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { type MechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
import { type MechanicsQuery, mechanicsQueryHash, type ParsedMechanicsQuery, readMechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsResult } from "./mechanics-result.ts";
import { randomBelow, type RandomLegalState, randomLegal } from "./random-legal.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import type { Actor, ActorContext, Player, RequestRecord } from "./runner.ts";
import { exportPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The actor was given something it cannot score: a missing context or team, an unknown request shape, a Pokemon that matches no set. It fails closed. */
export class DamageAwareError extends Error {
  override name = "DamageAwareError";
}

/** What the baseline counts per game (ADR-018 section 4: a baseline drifting towards `actor-rule` must be visible). All integers. */
export interface DamageAwareCounts {
  /** Requests answered (Team Preview included). */
  decisions: number;
  /** Team Preview requests (the `actor-rule` candidate). */
  preview_decisions: number;
  /** Requests whose candidates were scored. */
  scored_decisions: number;
  /** Probe queries asked (a repeated query counts each time it is asked). */
  queries: number;
  /** Probe runs actually made (queries not answered from this game's memo). */
  probe_runs: number;
  /** Queries the probe answered `unsupported`: scored 0, never approximated. */
  unsupported_queries: number;
  /** Requests answered by the `actor-rule` candidate because the state reader flagged a line. */
  reader_fallbacks: number;
  /** Requests answered by the `actor-rule` candidate because no legal assumed opponent set could be built. */
  assumed_set_fallbacks: number;
  /** Requests where `epsilon_ppm` made the baseline play a random candidate. */
  epsilon_plays: number;
  /** Requests answered by the `actor-rule` candidate because the mechanics probe refused a query (a declared state the engine does not take). A high rate means the baseline plays like `actor-rule`. */
  probe_refusals: number;
}

/** What `damage-aware@v2` counts about its belief per game, beside the v1 counts. All integers. */
export interface BeliefCounts {
  /** The record's number of worlds per scored request. */
  worlds: number;
  /** Scored requests that asked the belief. */
  decisions: number;
  /** Worlds sampled (`worlds` per decision that was not refused). */
  worlds_sampled: number;
  /** Redraws of a world with a set the target format's validator refused, summed. */
  redraws: number;
  /** Worlds dropped because they contradict a public fact of a Pokemon on the field (another species, a revealed move missing, a status its ability blocks) or hold a set the validator refuses once the public facts are applied. */
  contradicted_worlds: number;
  /** Requests where every world was dropped: answered by the `actor-rule` candidate. */
  no_consistent_world: number;
  /** Requests answered by the `actor-rule` candidate because the belief refused, by the model's code (reader_flagged, disguise_assignments_exceeded, legality_redraws_exhausted, ...). */
  refusals: Record<string, number>;
}

export interface DamageAwareReport {
  policy_id: string;
  identity: DamageAwareIdentity;
  /** SHA-256 of the identity's JCS bytes: games with another hash are another baseline. */
  identity_hash: string;
  counts: DamageAwareCounts;
  /** Present for `damage-aware@v2` only; v1's report has no such key. */
  belief?: BeliefCounts;
}

export type DamageAwareActor = Actor & { report(): DamageAwareReport };

export interface DamageAwareOptions {
  engine: Engine;
  vocabulary: TeamVocabulary;
  /** The v1 record (field rules) or the v2 record (a belief, which needs `worlds`). */
  config: AnyDamageAwareConfig;
  mechanics: MechanicsConfig;
  /** The belief port a v2 config needs (and a v1 config refuses): it must be the one the record names by hash, for the record's format. */
  worlds?: BeliefWorldsPort;
  /** The rules snapshot; read from `rules/` once when absent. */
  snapshot?: RulesSnapshot;
  /** The class the probe queries carry (they are never stored). */
  data_class: "synthetic" | "private";
  /** The `<commit>` of the policy id `damage-aware@<commit>`; `v1` for a v1 record and `v2` for a v2 record when absent. */
  commit?: string;
  /** The seed of the Mulberry32 stream `epsilon_ppm` draws from (uint32, the seat's baseline stream). Required when `epsilon_ppm` is above 0; never drawn from otherwise. */
  seed?: RandomLegalState;
  /** Test seams: the probe (default `probeMechanics`) and the random draws (default the `random-legal` stream's). */
  probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
  random?: { below: typeof randomBelow; legal: typeof randomLegal };
  /** Test seam (v1 only): the assumed opponent set of a public Pokemon (default `buildAssumedSet`). */
  assume?: (mon: PublicMon) => AssumedSet;
}

// What this module reads of the engine's data and of a request, narrowed locally.
interface ScoreDex {
  moves: { get(id: string): { exists: boolean; id: string; category: string } };
  items: { get(id: string): { exists: boolean; name: string } };
  species: { get(id: string): { exists: boolean; baseSpecies: string } };
  conditions: { get(id: string): { exists: boolean; effectType: string } };
}
type Json = unknown;
const object = (value: Json, what: string): Record<string, Json> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new DamageAwareError(`${what} is not an object`);
  return value as Record<string, Json>;
};
const array = (value: Json, what: string): Json[] => {
  if (!Array.isArray(value)) throw new DamageAwareError(`${what} is not an array`);
  return value;
};
const text = (value: Json, what: string): string => {
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
interface Fighter {
  paste: string;
  /** Current hit points, or null for full. */
  hp: number | null;
  status: string | null;
  boosts: Record<"atk" | "def" | "spa" | "spd" | "spe", number>;
  mega: boolean;
}
interface OwnSlot {
  fighter: Fighter;
  name: string;
}
interface Foe {
  mon: PublicMon;
  assumed: AssumedSet;
  fighter: Fighter;
  /** The foe's location number in a choice (`move 1 1`: 1 is the foe in the last active slot). */
  loc: number;
}

/** The reader's state does not show an own Pokemon the request lists: the request plays the `actor-rule` candidate and is counted. */
class ReaderGap extends Error {
  override name = "ReaderGap";
}

type DamageState = Extract<MechanicsQuery["query"], { kind: "damage" }>["state"];
type Combatant = Extract<MechanicsQuery["query"], { kind: "damage" }>["attacker"];

/** A v2 request is answered by the `actor-rule` candidate; the cause is already in the belief counts. Never leaves the actor. */
class BeliefFallback extends Error {
  override name = "BeliefFallback";
}

/** The probe raised a `MechanicsProbeError` for a query the baseline built: this request is answered by the `actor-rule` candidate. Never leaves the actor. */
class ProbeRefusal extends Error {
  override name = "ProbeRefusal";
}
/** The probe's words for a declared status the engine will not take; an unrevealed ability that blocks the status is then contradicted by it. */
const STATUS_REFUSED = "the engine does not take that status";
/** The probe's refusal of a query for a format its config is not for: a defect of the run, so it propagates. */
const CONFIG_DEFECT = "the mechanics config is for another format";

const PLAYERS: readonly Player[] = ["p1", "p2"];

/**
 * A damage-aware `Actor` for one seat of one game: create a new one per game (it keeps the reader's
 * state and this game's memo and counts). `ActorContext.team` is required: a request alone does not
 * carry whole sets.
 */
export function damageAwareActor(options: DamageAwareOptions): DamageAwareActor {
  const { engine, vocabulary, config, mechanics, data_class } = options;
  const worlds = options.worlds;
  // A v1 record carries field rules; a v2 record names a belief, and the port must be that belief, for that format.
  const v1 = config.schema_version === 1 ? config : null;
  const v2 = config.schema_version === 2 ? config : null;
  if (v2 !== null) {
    if (worlds === undefined) throw new DamageAwareError("a v2 config names a belief: the actor needs the belief port");
    if (options.assume !== undefined) throw new DamageAwareError("a v2 actor takes its assumed sets from the belief, not from an assume seam");
    if (worlds.prior_format_id !== v2.format_id) throw new DamageAwareError("the usage prior is for another format than the record's: it is never used silently");
    if (worlds.prior_hash !== v2.assumed_set.prior_hash) throw new DamageAwareError("the belief port holds another usage prior than the record names");
    if (worlds.belief_config_hash !== v2.assumed_set.belief_config_hash) throw new DamageAwareError("the belief port holds another belief config than the record names");
  } else if (worlds !== undefined) throw new DamageAwareError("a v1 config names no belief: it takes no belief port");
  const probe = options.probe ?? probeMechanics;
  const random = options.random ?? { below: randomBelow, legal: randomLegal };
  const epsilon = config.epsilon_ppm;
  if (epsilon > 0 && options.seed === undefined) throw new DamageAwareError("epsilon_ppm above 0 needs the seat's baseline seed");
  let stream: RandomLegalState = options.seed ?? 0;
  const snapshot = options.snapshot ?? loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, config.format_id, snapshot).manifest;
  // A policy id that names a record version (`v1`, `v2-x`) must be the version of the record the actor plays: a v1 record never plays as damage-aware@v2.
  const commit = options.commit ?? (v2 === null ? "v1" : "v2");
  const named = /^v(\d+)(?:-|$)/.exec(commit);
  if (named !== null ? named[1] !== String(config.schema_version) : v2 !== null) throw new DamageAwareError("the policy id names another record version than the record the actor plays");
  const policy_id = damageAwarePolicyId(commit);
  const identity = damageAwareIdentity({ config, mechanics, manifest, reader_sha256: readerSha256(), ...(worlds === undefined ? {} : { belief_identity_hash: worlds.identity_hash }) });
  const identity_hash = damageAwareIdentityHash(identity);
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(config.format_id).mod) as unknown as ScoreDex;
  const counts: DamageAwareCounts = {
    decisions: 0, preview_decisions: 0, scored_decisions: 0, queries: 0, probe_runs: 0,
    unsupported_queries: 0, reader_fallbacks: 0, assumed_set_fallbacks: 0, epsilon_plays: 0, probe_refusals: 0,
  };
  const belief: BeliefCounts | null = v2 === null ? null : { worlds: v2.assumed_set.worlds, decisions: 0, worlds_sampled: 0, redraws: 0, contradicted_worlds: 0, no_consistent_world: 0, refusals: {} };
  const assume = options.assume ?? ((mon: PublicMon) => buildAssumedSet({ engine, vocabulary, config: v1 as DamageAwareConfig }, mon));

  // This game's reader, fed each request's new lines only.
  let reader: PublicInterpreter | null = null;
  let seat: Player | null = null;
  let fed = 0;
  // Every line this game's reader flagged, in channel order (the belief refuses while any is on record).
  const flagList: PublicFlag[] = [];
  // Probe results by query hash: a pure function of the query on a pin and config, so a memo changes no answer.
  const memo = new Map<string, MechanicsResult | null | { refused: string }>();

  /**
   * One probe answer, or null for a speed order the probe cannot take as a query (an action that
   * queues more than a move and a Mega Evolution, as Focus Punch does, or one with a fractional priority): counted as `unsupported`,
   * so no discount is applied. Any other `MechanicsProbeError` throws `ProbeRefusal` (the request falls back to `actor-rule`); every other failure is a defect and propagates.
   */
  function ask(value: MechanicsQuery): MechanicsResult | null {
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
        if (query.query.kind === "speed_order" && [UNQUEUEABLE, FRACTIONAL].some((limit) => error.message.startsWith(limit))) {
          // The probe's two known speed-order limits: no discount, counted as unsupported.
          memo.set(key, null);
          counts.unsupported_queries += 1;
          return null;
        }
        // Any other refusal of a state the baseline declared: this request falls back to the actor-rule candidate.
        memo.set(key, { refused: error.message });
        throw new ProbeRefusal(error.message);
      }
      memo.set(key, result);
    }
    if (result === null) {
      counts.unsupported_queries += 1;
      return null;
    }
    if ("refused" in result) throw new ProbeRefusal(result.refused); // a memoised refusal (the query was asked again)
    if (result.kind === "refused") throw new DamageAwareError("the probe refused a set the baseline built");
    if (result.kind === "unsupported") counts.unsupported_queries += 1;
    return result;
  }
  const query = (body: MechanicsQuery["query"]): MechanicsResult | null => ask({ schema_version: 1, data_class, format_id: config.format_id, query: body } as MechanicsQuery);

  const damageOf = (attacker: Fighter, defender: Fighter, move: string, state: PublicState, attackerIsOwn: boolean): DamageAnswer | null => {
    const result = query({ kind: "damage", attacker: combatant(attacker), defender: combatant(defender), move, state: field(state, attackerIsOwn) });
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

  /** The player's own active Pokemon in request slot `slot`, from its own request and own team. */
  function ownSlot(side: Record<string, Json>, team: readonly Readonly<PokemonSet>[], state: PublicState, slot: number): OwnSlot {
    const mon = object(array(side.pokemon, "side.pokemon")[slot], "an active Pokemon");
    const ident = text(mon.ident, "ident");
    const name = ident.slice(ident.indexOf(": ") + 2);
    const matches = team.filter((set) => [set.name, set.name === "" ? set.species : "", dex.species.get(set.species).baseSpecies].includes(name));
    if (matches.length !== 1) throw new DamageAwareError("an own Pokemon matches no set, or more than one, of the team as brought");
    const shown = state.own.roster.find((candidate) => candidate.name === name);
    if (shown === undefined) throw new ReaderGap("the reader does not show an own Pokemon the request lists");
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
        boosts: boostsOf(shown),
        mega: shown.mega,
      },
    };
  }

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
    let set = assume(mon);
    if (mon.status === null || mon.ability !== null || options.assume !== undefined) return set;
    const ruledOut: string[] = [];
    while (statusBlocked(set, mon, state)) {
      ruledOut.push(set.set.ability);
      set = buildAssumedSet({ engine, vocabulary, config: v1 as DamageAwareConfig }, mon, ruledOut);
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
      out.push({ mon, assumed, fighter, loc: letters.length - position });
    });
    return out;
  }

  /**
   * The score of every legal choice of one request, in the legal set's order, against the foes `foesNow`
   * gives (built on first use). Throws `AssumedSetError` for the caller's fallback.
   */
  function scoreChoices(request: RequestRecord, state: PublicState, own: readonly OwnSlot[], foesNow: () => Foe[]): number[] {
    const body = object(request.request, "request");
    const active = array(body.active, "active");
    const scores = new Map<string, number>();
    const threats = new Map<string, string | null>();

    // The first revealed move of foe `loc` that can knock out own slot `slot`, or null.
    const threatOf = (slot: number, foe: Foe): string | null => {
      const key = `${slot}/${foe.loc}`;
      const known = threats.get(key);
      if (known !== undefined) return known;
      const mine = (own[slot] as OwnSlot).fighter;
      let found: string | null = null;
      for (const move of foe.assumed.revealed_moves) {
        if (found !== null || dex.moves.get(move).category === "Status") continue;
        const answer = damageOf(foe.fighter, mine, move, state, false);
        if (answer !== null && canKnockOut(answer)) found = move;
      }
      threats.set(key, found);
      return found;
    };

    const against = (slot: number, move: string, mega: boolean, foe: Foe): number => {
      const mine = own[slot] as OwnSlot;
      const answer = damageOf({ ...mine.fighter, mega: mine.fighter.mega || mega }, foe.fighter, move, state, true);
      if (answer === null) return 0;
      const value = damageValue(answer, config.scoring);
      const threat = value === 0 ? null : threatOf(slot, foe);
      if (threat === null) return value;
      // The foe can knock the actor out with a revealed move: discount the action if the foe acts first.
      const order = query({
        kind: "speed_order",
        sides: [[combatant(mine.fighter)], [combatant(foe.fighter)]],
        actions: [
          { side: 0, position: 0, move, mega },
          { side: 1, position: 0, move: threat, mega: false },
        ],
        state: field(state, true),
      });
      if (order?.kind !== "computed" || order.answer.kind !== "speed_order") return value;
      return actsBefore(order.answer as SpeedOrderAnswer, 1, 0) ? discounted(value, config.scoring.turn_order_discount_ppm) : value;
    };

    const moveScore = (slot: number, part: Extract<SlotPart, { kind: "move" }>): number => {
      const key = `${slot}/${part.index}/${part.target}/${part.mega}`;
      const known = scores.get(key);
      if (known !== undefined) return known;
      const moves = array(object(active[slot], "an active entry").moves, "moves");
      const slotMove = object(moves[part.index - 1], "a move of the request");
      const id = text(slotMove.id, "a move id");
      const data = dex.moves.get(id);
      let value = 0;
      if (data.exists && data.category !== "Status") {
        // A locked move (Recharge, the second turn of Dig) is sent with no target type: it is one fixed part of every candidate, so it scores 0.
        const type = typeof slotMove.target === "string" ? slotMove.target : undefined;
        let targets: Foe[] = [];
        if (type !== undefined && SINGLE.has(type)) {
          if (part.target === null) throw new DamageAwareError("a move that needs a target has none");
          targets = foesNow().filter((foe) => foe.loc === part.target);
        } else if (type !== undefined && EVERY_FOE.has(type)) targets = foesNow();
        else if (type !== undefined && RANDOM_FOE.has(type)) targets = foesNow();
        else if (type !== undefined && !NO_FOE.has(type)) throw new DamageAwareError("a move target type the baseline does not know");
        // A spread move is the sum of its single-target values (v1); a random target is the mean over the foes.
        const sum = targets.reduce((total, foe) => total + against(slot, id, part.mega, foe), 0);
        value = type !== undefined && RANDOM_FOE.has(type) && targets.length > 0 ? Math.floor(sum / targets.length) : sum;
      }
      scores.set(key, value);
      return value;
    };

    return (request.legal as NonNullable<RequestRecord["legal"]>).choices.map((choice) => {
      const parts = parseJointChoice(choice);
      if (parts.length !== active.length) throw new DamageAwareError("a choice has a different number of parts than the request has active Pokemon");
      return parts.reduce((sum, part, slot) => sum + (part.kind === "move" ? moveScore(slot, part) : 0), 0);
    });
  }

  /** The first of the highest totals in the legal set's order wins (as `actor-rule` breaks every tie). */
  function best(request: RequestRecord, totals: readonly number[]): string {
    let pick = 0;
    totals.forEach((total, index) => {
      if (total > (totals[pick] as number)) pick = index;
    });
    return (request.legal as NonNullable<RequestRecord["legal"]>).choices[pick] as string;
  }

  const ownSlots = (request: RequestRecord, state: PublicState, team: readonly Readonly<PokemonSet>[]): OwnSlot[] => {
    const body = object(request.request, "request");
    const side = object(body.side, "side");
    return array(body.active, "active").map((_, slot) => ownSlot(side, team, state, slot));
  };

  /** v1: score one request's candidates on the record's assumed sets. Throws `AssumedSetError` for the caller's fallback. */
  function score(request: RequestRecord, state: PublicState, team: readonly Readonly<PokemonSet>[]): string {
    let foes: Foe[] | null = null;
    const foesNow = (): Foe[] => (foes ??= foesOf(state, (mon) => assumedFor(mon, state)));
    return best(request, scoreChoices(request, state, ownSlots(request, state, team), foesNow));
  }

  /**
   * v2: score every candidate in each of the record's N sampled worlds of the opponent and play the best sum
   * (the mean's order; the divisor is the same for every candidate). A world that contradicts a public fact of
   * a Pokemon on the field is dropped and counted; the belief's own refusals and "no world left" fall back.
   */
  function scoreWorlds(request: RequestRecord, state: PublicState, team: readonly Readonly<PokemonSet>[], flags: readonly PublicFlag[]): string {
    const counted = belief as BeliefCounts;
    const port = worlds as BeliefWorldsPort;
    const own = ownSlots(request, state, team);
    counted.decisions += 1;
    let sampled;
    try {
      sampled = port.sample({ opponent: state.opponent, flags, stream: worldsStream(request), n: counted.worlds });
    } catch (error) {
      const code = refusalCode(error);
      if (code === null || BELIEF_DEFECTS.has(code)) throw error;
      counted.refusals[code] = (counted.refusals[code] ?? 0) + 1;
      throw new BeliefFallback();
    }
    counted.worlds_sampled += sampled.worlds.length;
    counted.redraws += sampled.redraws;
    const sums = (request.legal as NonNullable<RequestRecord["legal"]>).choices.map(() => 0);
    let used = 0;
    for (const { world } of sampled.worlds) {
      let foes: Foe[];
      try {
        foes = foesOf(state, (mon, position) => {
          const set = worldAssumed(mon, position, world);
          if (mon.status !== null && mon.ability === null && statusBlocked(set, mon, state)) throw new WorldMismatchError("the Pokemon's ability in this world blocks the status it is shown with");
          return set;
        });
      } catch (error) {
        // a world whose set the validator refuses once the public facts are applied is dropped like one that contradicts them: one bad world of eight is not a bad request
        if (!(error instanceof WorldMismatchError || error instanceof AssumedSetError)) throw error;
        counted.contradicted_worlds += 1;
        continue;
      }
      scoreChoices(request, state, own, () => foes).forEach((total, index) => {
        sums[index] = (sums[index] as number) + total;
      });
      used += 1;
    }
    if (used === 0) {
      counted.no_consistent_world += 1;
      throw new BeliefFallback();
    }
    return best(request, sums);
  }

  /** The assumed set of the Pokemon at roster `position` in a sampled world; a world with no brought set there is not that Pokemon. */
  function worldAssumed(mon: PublicMon, position: number, world: WorldLike): AssumedSet {
    const slot = world.slots[position];
    if (slot === undefined || !slot.brought || slot.set === null) throw new WorldMismatchError("the world does not bring the Pokemon on the field");
    return buildWorldAssumedSet({ engine, vocabulary, format_id: config.format_id }, mon, slot.set);
  }

  /** Feed the lines the reader has not seen; true when any of them was flagged (a line it could not take). */
  function feed(context: ActorContext): { flagged: boolean; state: PublicState; flags: readonly PublicFlag[] } {
    if (reader === null) {
      if (!PLAYERS.includes(context.player)) throw new DamageAwareError("the context names no player");
      seat = context.player;
      reader = createInterpreter({ player: context.player });
    }
    if (context.player !== seat || context.channel.length < fed) throw new DamageAwareError("an actor plays one seat of one game: create a new one per game");
    let flagged = false;
    for (let index = fed; index < context.channel.length; index++) {
      const line = context.channel[index] as string;
      const reason = reader.feed(line).flagged;
      if (reason !== null) {
        flagged = true;
        flagList.push({ index, line, reason });
      }
    }
    fed = context.channel.length;
    return { flagged, state: reader.snapshot(), flags: flagList };
  }

  function decide(request: RequestRecord, context?: ActorContext): string {
    const legal = request.legal;
    if (legal === undefined) throw new NoCandidateError(`${request.player}: request ${request.request_seq} has no legal choice set`);
    if (context === undefined || context.team === undefined) throw new DamageAwareError("the damage-aware baseline needs the ActorContext with the player's own team");
    counts.decisions += 1;
    const { flagged, state, flags } = feed(context);
    if (request.phase === "preview") {
      counts.preview_decisions += 1;
      return actorRule(legal);
    }
    if (epsilon > 0) {
      const draw = random.below(PPM, stream);
      stream = draw.state;
      if (draw.value < epsilon) {
        const pick = random.legal(legal, stream);
        stream = pick.state;
        counts.epsilon_plays += 1;
        return pick.choice;
      }
    }
    // Only a move request has anything to score (a switch, a pass and a forced switch score 0), and one candidate has one answer.
    if (request.phase !== "move" || legal.choices.length < 2) return actorRule(legal);
    if (flagged) {
      counts.reader_fallbacks += 1;
      return actorRule(legal);
    }
    try {
      const choice = v2 === null ? score(request, state, context.team) : scoreWorlds(request, state, context.team, flags);
      counts.scored_decisions += 1;
      return choice;
    } catch (error) {
      if (error instanceof ReaderGap) counts.reader_fallbacks += 1;
      else if (error instanceof AssumedSetError) counts.assumed_set_fallbacks += 1;
      else if (error instanceof ProbeRefusal) counts.probe_refusals += 1;
      else if (!(error instanceof BeliefFallback)) throw error;
      return actorRule(legal);
    }
  }

  const actor: Actor = (request, context) => decide(request, context);
  return Object.assign(actor, {
    report: (): DamageAwareReport => ({
      policy_id,
      identity,
      identity_hash,
      counts: { ...counts },
      ...(belief === null ? {} : { belief: { ...belief, refusals: { ...belief.refusals } } }),
    }),
  });
}
