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
// Excluded by construction: the actor's parameters are the request and an `ActorContext`, whose type
// holds no opponent team, omniscient log, hidden verdict or seed (`runner.ts`); the probe takes only
// a query read from declared sets and state.

import { createInterpreter, type PublicInterpreter, type PublicMon, type PublicState, readerSha256 } from "@areneon/public-state";
import { actorRule, NoCandidateError } from "./actor-rule.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";
import { type AssumedSet, AssumedSetError, assumedHp, buildAssumedSet } from "./damage-aware-assumed-set.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
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
}

export interface DamageAwareReport {
  policy_id: string;
  identity: DamageAwareIdentity;
  /** SHA-256 of the identity's JCS bytes: games with another hash are another baseline. */
  identity_hash: string;
  counts: DamageAwareCounts;
}

export type DamageAwareActor = Actor & { report(): DamageAwareReport };

export interface DamageAwareOptions {
  engine: Engine;
  vocabulary: TeamVocabulary;
  config: DamageAwareConfig;
  mechanics: MechanicsConfig;
  /** The rules snapshot; read from `rules/` once when absent. */
  snapshot?: RulesSnapshot;
  /** The class the probe queries carry (they are never stored). */
  data_class: "synthetic" | "private";
  /** The `<commit>` of the policy id `damage-aware@<commit>`; `v1` when absent. */
  commit?: string;
  /** The seed of the Mulberry32 stream `epsilon_ppm` draws from (uint32, the seat's baseline stream). Required when `epsilon_ppm` is above 0; never drawn from otherwise. */
  seed?: RandomLegalState;
  /** Test seams: the probe (default `probeMechanics`) and the random draws (default the `random-legal` stream's). */
  probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
  random?: { below: typeof randomBelow; legal: typeof randomLegal };
  /** Test seam: the assumed opponent set of a public Pokemon (default `buildAssumedSet`). */
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

const PLAYERS: readonly Player[] = ["p1", "p2"];

/**
 * A damage-aware `Actor` for one seat of one game: create a new one per game (it keeps the reader's
 * state and this game's memo and counts). `ActorContext.team` is required: a request alone does not
 * carry whole sets.
 */
export function damageAwareActor(options: DamageAwareOptions): DamageAwareActor {
  const { engine, vocabulary, config, mechanics, data_class } = options;
  const probe = options.probe ?? probeMechanics;
  const random = options.random ?? { below: randomBelow, legal: randomLegal };
  const epsilon = config.epsilon_ppm;
  if (epsilon > 0 && options.seed === undefined) throw new DamageAwareError("epsilon_ppm above 0 needs the seat's baseline seed");
  let stream: RandomLegalState = options.seed ?? 0;
  const snapshot = options.snapshot ?? loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, config.format_id, snapshot).manifest;
  const policy_id = damageAwarePolicyId(options.commit ?? "v1");
  const identity = damageAwareIdentity({ config, mechanics, manifest, reader_sha256: readerSha256() });
  const identity_hash = damageAwareIdentityHash(identity);
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(config.format_id).mod) as unknown as ScoreDex;
  const counts: DamageAwareCounts = {
    decisions: 0, preview_decisions: 0, scored_decisions: 0, queries: 0, probe_runs: 0,
    unsupported_queries: 0, reader_fallbacks: 0, assumed_set_fallbacks: 0, epsilon_plays: 0,
  };
  const assume = options.assume ?? ((mon: PublicMon) => buildAssumedSet({ engine, vocabulary, config }, mon));

  // This game's reader, fed each request's new lines only.
  let reader: PublicInterpreter | null = null;
  let seat: Player | null = null;
  let fed = 0;
  // Probe results by query hash: a pure function of the query on a pin and config, so a memo changes no answer.
  const memo = new Map<string, MechanicsResult | null>();

  /**
   * One probe answer, or null for a speed order the probe cannot take as a query (an action that
   * queues more than a move and a Mega Evolution, as Focus Punch does): counted as `unsupported`,
   * so no discount is applied. Every other probe failure is a defect and propagates.
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
        if (!(error instanceof MechanicsProbeError) || query.query.kind !== "speed_order" || !error.message.startsWith(UNQUEUEABLE)) throw error;
        memo.set(key, null);
        counts.unsupported_queries += 1;
        return null;
      }
      memo.set(key, result);
    }
    if (result === null) {
      counts.unsupported_queries += 1;
      return null;
    }
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

  /** The opposing active Pokemon with their assumed sets, by choice location. Throws `AssumedSetError` when none can be built. */
  function foesOf(state: PublicState): Foe[] {
    const letters = Object.keys(state.opponent.slots).sort();
    const out: Foe[] = [];
    letters.forEach((letter, position) => {
      const index = state.opponent.slots[letter];
      const mon = index === null || index === undefined ? undefined : state.opponent.roster[index];
      if (mon === undefined || mon.fainted) return;
      const assumed = assume(mon);
      const percent = assumed.shown_hp_percent;
      if (percent === null || percent === 0) return;
      const max = maxHpOf(assumed.paste);
      const hp = assumedHp(percent, max, config.hp_rounding);
      const fighter: Fighter = { paste: assumed.paste, hp: hp === max ? null : hp, status: mon.status, boosts: boostsOf(mon), mega: assumed.mega };
      out.push({ mon, assumed, fighter, loc: letters.length - position });
    });
    return out;
  }

  /** Score one request's candidates. Throws `AssumedSetError` for the caller's fallback. */
  function score(request: RequestRecord, state: PublicState, team: readonly Readonly<PokemonSet>[]): string {
    const body = object(request.request, "request");
    const side = object(body.side, "side");
    const active = array(body.active, "active");
    let foes: Foe[] | null = null;
    const foesNow = (): Foe[] => (foes ??= foesOf(state));
    const own = active.map((_, slot) => ownSlot(side, team, state, slot));
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

    let best: string | null = null;
    let bestScore = -1;
    for (const choice of (request.legal as NonNullable<RequestRecord["legal"]>).choices) {
      const parts = parseJointChoice(choice);
      if (parts.length !== active.length) throw new DamageAwareError("a choice has a different number of parts than the request has active Pokemon");
      const total = parts.reduce((sum, part, slot) => sum + (part.kind === "move" ? moveScore(slot, part) : 0), 0);
      // The first of equal scores in canonical order wins (as `actor-rule` breaks every tie).
      if (total > bestScore) {
        best = choice;
        bestScore = total;
      }
    }
    return best as string;
  }

  /** Feed the lines the reader has not seen; true when any of them was flagged (a line it could not take). */
  function feed(context: ActorContext): { flagged: boolean; state: PublicState } {
    if (reader === null) {
      if (!PLAYERS.includes(context.player)) throw new DamageAwareError("the context names no player");
      seat = context.player;
      reader = createInterpreter({ player: context.player });
    }
    if (context.player !== seat || context.channel.length < fed) throw new DamageAwareError("an actor plays one seat of one game: create a new one per game");
    let flagged = false;
    for (const line of context.channel.slice(fed)) flagged = reader.feed(line).flagged !== null || flagged;
    fed = context.channel.length;
    return { flagged, state: reader.snapshot() };
  }

  function decide(request: RequestRecord, context?: ActorContext): string {
    const legal = request.legal;
    if (legal === undefined) throw new NoCandidateError(`${request.player}: request ${request.request_seq} has no legal choice set`);
    if (context === undefined || context.team === undefined) throw new DamageAwareError("the damage-aware baseline needs the ActorContext with the player's own team");
    counts.decisions += 1;
    const { flagged, state } = feed(context);
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
      const choice = score(request, state, context.team);
      counts.scored_decisions += 1;
      return choice;
    } catch (error) {
      if (error instanceof ReaderGap) counts.reader_fallbacks += 1;
      else if (error instanceof AssumedSetError) counts.assumed_set_fallbacks += 1;
      else throw error;
      return actorRule(legal);
    }
  }

  const actor: Actor = (request, context) => decide(request, context);
  return Object.assign(actor, {
    report: (): DamageAwareReport => ({ policy_id, identity, identity_hash, counts: { ...counts } }),
  });
}
