// The engine-backed speed-order check of the opponent belief (ADR-025 section 7, ticket #1133). It is the port the belief
// reads (`SpeedCheck` of `packages/opponent-model`, declared structurally here because the bridge does not import the
// model): the isolated pairs of the player's own channel (`speed-pairs.ts`, with the reviewed rules of
// `BeliefConfig.speed_evidence`) as events about opponent slots, and for a candidate set of the slot whether the ADR-016
// `speed_order` probe gives the order the player saw.
//
// Only the probe computes. The opponent side is the candidate set (declared at the shown status and stages, and, if its hit
// points are shown below full, at one hit point: the engine reads hit points for no order key but a full-health priority
// bonus); the own side is the player's own Pokemon from the request in force at the move line and the own team. Both
// actions are the moves that were used. A candidate set whose probe answer puts the observed first Pokemon in an earlier
// group passes, and so does an exact tie (its order is a draw, never sampled); a candidate the probe cannot answer
// (`unsupported`, for instance a random Quick Claw draw, a fractional priority, a set the validator refuses, a Mega forme)
// is `null`: it adds no constraint. Only the Speed-relevant fields of a candidate reach the probe (its nature and Speed
// Stat Points, item and ability; the other Stat Points are zeroed), so candidates that differ in the rest share one answer.

import { type PublicFrame, type PublicMon, type PublicRun, type PublicState, interpretChannel } from "@areneon/public-state";
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { DamageCheckCache, DamageCandidatePort } from "./damage-check.ts";
import { newDamageCheckCache } from "./damage-check.ts";
import { DamageAwareError, type Fighter, ProbeRefusal, damageView } from "./damage-aware-view.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import { type Engine, type PokemonSet, defaultIvs } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "./mechanics-query.ts";
import type { RulesSnapshot } from "./rules.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { type SpeedPair, extractPairs } from "./speed-pairs.ts";
import { exportPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** Bumped when the check declares a pair to the probe or judges its order differently. */
export const SPEED_CHECK_FORMAT = 1;

export type SpeedEvidenceRules = NonNullable<BeliefConfig["speed_evidence"]>;

/** Structurally the opponent model's `SpeedEvent`. */
export interface SpeedEventPort {
  readonly id: string;
  readonly slot: number;
  readonly role: "first" | "second";
}

/** Structurally the opponent model's `SpeedCheck`. */
export interface SpeedCheckPort {
  readonly events: readonly SpeedEventPort[];
  check(event: SpeedEventPort, candidate: DamageCandidatePort): boolean | null;
}

/** What the check found in the channel and did, in integers (the cost note and the tests read it). */
export interface SpeedCheckCounts {
  /** Own-and-opponent pairs of actions in a turn. */
  pairs: number;
  /** Pairs that were isolated and concern an opponent slot (the events). */
  events: number;
  /** Pairs that were not isolated, by the first reason. */
  not_isolated: Record<string, number>;
  /** `check` calls (a repeated call counts each time). */
  calls: number;
  /** Own Pokemon of a pair that the request and the team could not name: the check answered `null` for them. */
  own_unstated: number;
  /** Probe queries asked. */
  queries: number;
  probe_runs: number;
}

export interface SpeedCheckContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  /** The ADR-018 record (it names the format). */
  readonly config: DamageAwareConfig;
  readonly mechanics: MechanicsConfig;
  /** The belief config with its `speed_evidence`: the reviewed rules and the identity-hiding abilities. */
  readonly belief: BeliefConfig;
  readonly snapshot?: RulesSnapshot;
  readonly data_class: "synthetic" | "private";
  /** The game's shared cache (one per game and seat; the damage check's type), else the port keeps its own. */
  readonly cache?: DamageCheckCache;
  /** Test seam: the probe (default `probeMechanics`). */
  readonly probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
}

export interface SpeedCheckInput {
  /** The player's seat (`p1` or `p2`). */
  readonly player: string;
  /** The player's own channel up to the request being answered. */
  readonly lines: readonly string[];
  /** The player's own team as brought. */
  readonly team: readonly Readonly<PokemonSet>[];
  /** The reader's run over `lines` when the caller has it. */
  readonly run?: PublicRun;
}

export interface SpeedCheckHandle extends SpeedCheckPort {
  readonly pairs: readonly SpeedPair[];
  counts(): SpeedCheckCounts;
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const SHOWN_GENDER = /, ([MF])(?:,|$)/;

export function createSpeedCheck(context: SpeedCheckContext, input: SpeedCheckInput): SpeedCheckHandle | null {
  const { engine, vocabulary, config, mechanics, belief, data_class } = context;
  if (belief.speed_evidence === undefined) return null;
  const rules: SpeedEvidenceRules = belief.speed_evidence;
  const cache = context.cache ?? newDamageCheckCache();
  if ((cache.player ??= input.player) !== input.player) throw new DamageAwareError("a speed-check cache belongs to one seat of one game");
  const snapshot = context.snapshot ?? loadRulesSnapshot(engine.commit);
  const probe = context.probe ?? probeMechanics;
  const run = input.run ?? interpretChannel(input.player, input.lines);
  // A channel the reader flagged, or one with no request yet, has nothing to say.
  if (run.flags.length > 0 || run.requests.length === 0) return null;
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const probeCounts = { queries: 0, probe_runs: 0, unsupported_queries: 0 };
  const view = damageView({ engine, vocabulary, config, v1: config, mechanics, snapshot, data_class, probe, counts: probeCounts, memo: cache.memo });
  const dex = view.dex;
  const names = dex as unknown as Record<"species" | "abilities" | "moves" | "items", { get(id: string): { name: string; baseSpecies: string; priority?: number; exists?: boolean } }>;
  const requests = run.requests as EngineRequest[];

  // The abilities a Pokemon in play may hold, as the damage check reads them.
  const ownAbility = new Map<string, string>();
  const ownItem = new Map<string, string>();
  for (const entry of (requests[requests.length - 1] as EngineRequest).side.pokemon) {
    const name = entry.ident.slice(entry.ident.indexOf(": ") + 2);
    ownAbility.set(name, toId(String(entry.ability)));
    ownItem.set(name, toId(String(entry.item)));
  }
  const abilitiesOf = (mon: PublicMon): readonly string[] => {
    if (mon.own) return [ownAbility.get(String(mon.name)) as string];
    if (mon.ability !== null) return [mon.ability];
    return roster.get(String(mon.species ?? mon.base_species))?.abilities ?? rules.non_isolating.abilities;
  };
  const pairs = extractPairs(run.frames, {
    rules: rules.non_isolating,
    priorityOf: (move) => {
      const found = names.moves.get(move);
      return found.exists === true ? (found.priority as number) : null;
    },
    abilitiesOf,
    mayHide: (mon) => [mon.base_species, ...(mon.species === null ? [] : [mon.species])].some((id) => roster.get(id)?.abilities.some((a) => belief.identity_hiding_abilities.includes(a))),
    declarable: (state: PublicState) => state.field_conditions.every((id) => id === "trickroom" || id === state.terrain || dex.conditions.get(id).effectType === "Terrain"),
    // The own Pokemon's item and ability are those of the latest request (they can only differ from the request in force by an item lost since, which the reader's `item_lost` shows).
    unmodelledOwn: (mon) => {
      const ability = ownAbility.get(String(mon.name));
      const item = ownItem.get(String(mon.name));
      return (ability !== undefined && rules.unmodelled.abilities.includes(ability)) || (item !== undefined && item !== "" && rules.unmodelled.items.includes(item));
    },
  });

  const counts: SpeedCheckCounts = { pairs: pairs.length, events: 0, not_isolated: {}, calls: 0, own_unstated: 0, queries: 0, probe_runs: 0 };
  const events: SpeedEventPort[] = [];
  const byEvent = new Map<string, SpeedPair>();
  for (const pair of pairs) {
    if (pair.reason !== null) {
      counts.not_isolated[pair.reason] = (counts.not_isolated[pair.reason] ?? 0) + 1;
      continue;
    }
    const mon = pair.state.opponent.roster[pair.foe.index];
    if (mon === undefined || mon.mega) continue;
    const event: SpeedEventPort = { id: `speed${pair.own.frame}-${pair.foe.frame}`, slot: pair.foe.index, role: pair.first === "foe" ? "first" : "second" };
    events.push(event);
    byEvent.set(event.id, pair);
  }
  counts.events = events.length;

  const natureName = (id: string): string | undefined => vocabulary.natures.find((n) => n.id === id)?.name;
  const answers = cache.answers;

  /**
   * The own Pokemon of a pair as a probe fighter, or null when the probe cannot state it: its set is its team set with the
   * item, ability and gender the request in force at the move line shows, and its state is the reader's. A Pokemon that has
   * Mega Evolved or whose shown forme is not its set's species is not stated (as in the damage check).
   */
  function ownFighter(pair: SpeedPair): Fighter | null {
    const frame = run.frames[pair.own.frame] as PublicFrame;
    const mon = pair.state.own.roster[pair.own.index] as PublicMon;
    const request = requests[frame.request - 1];
    const entry = request?.side.pokemon.find((p) => p.ident.slice(p.ident.indexOf(": ") + 2) === mon.name);
    const name = mon.name as string;
    const matches = input.team.filter((set) => [set.name, set.name === "" ? set.species : "", names.species.get(set.species).baseSpecies].includes(name));
    if (entry === undefined || matches.length !== 1) {
      counts.own_unstated += 1;
      return null;
    }
    const set: PokemonSet = structuredClone(matches[0] as PokemonSet);
    if (mon.mega || toId(entry.details.split(",")[0] as string) !== toId(set.species)) return null;
    const gender = SHOWN_GENDER.exec(entry.details)?.[1];
    if (gender !== undefined) set.gender = gender;
    set.item = entry.item === "" ? "" : names.items.get(entry.item).name;
    set.ability = names.abilities.get(entry.ability as string).name;
    const hp = mon.hp_num === null || mon.hp_den === null || mon.hp_num === mon.hp_den ? null : Math.max(1, mon.hp_num);
    return { paste: exportPaste(engine, [set]), hp, status: mon.status, boosts: view.boostsOf(mon), mega: mon.mega };
  }

  function evaluate(event: SpeedEventPort, candidate: DamageCandidatePort): boolean | null {
    const pair = byEvent.get(event.id) as SpeedPair;
    if (candidate.mega) return null;
    const entry = roster.get(candidate.forme);
    const nature = natureName(candidate.nature);
    if (entry === undefined || nature === undefined || !entry.moves.includes(pair.foe.move)) return null;
    const foeMon = pair.state.opponent.roster[pair.foe.index] as PublicMon;
    // Only what the order reads: the Speed Stat Points (one point elsewhere when there is none, so the validator's "0 Stat Points" check stays quiet).
    const speed = candidate.stat_points.spe;
    const set: PokemonSet = {
      name: "",
      species: names.species.get(candidate.forme).name,
      item: candidate.item === null ? "" : names.items.get(candidate.item).name,
      ability: names.abilities.get(candidate.ability).name,
      moves: [names.moves.get(pair.foe.move).name],
      nature,
      gender: foeMon.gender ?? "",
      evs: { hp: speed > 0 ? 0 : 1, atk: 0, def: 0, spa: 0, spd: 0, spe: speed },
      ivs: defaultIvs(),
      level: foeMon.level ?? vocabulary.rules.adjust_level,
    };
    const paste = exportPaste(engine, [set]);
    try {
      const own = ownFighter(pair);
      if (own === null) return null;
      // A shown 100 is full hit points; below it the Pokemon is declared at one (only a full-health bonus reads hit points).
      const foe: Fighter = { paste, hp: foeMon.hp_num === foeMon.hp_den ? null : 1, status: foeMon.status, boosts: view.boostsOf(foeMon), mega: false };
      const result = view.queryLimited({
        kind: "speed_order",
        sides: [[view.combatant(own)], [view.combatant(foe)]],
        actions: [
          { side: 0, position: 0, move: pair.own.move, mega: false },
          { side: 1, position: 0, move: pair.foe.move, mega: false },
        ],
        state: view.field(pair.state, true),
      } as never);
      if ("limit" in result || result.kind !== "computed" || result.answer.kind !== "speed_order") return null;
      const groups = result.answer.groups;
      const [mine, theirs] = [0, 1].map((a) => groups.findIndex((members) => members.includes(a)));
      // An exact tie is a draw: either order is possible.
      return mine === theirs || (pair.first === "own" ? (mine as number) < (theirs as number) : (theirs as number) < (mine as number));
    } catch (error) {
      if (error instanceof ProbeRefusal || (error instanceof DamageAwareError && error.message.startsWith("the probe refused"))) return null;
      throw error;
    }
  }

  return {
    events,
    pairs,
    check(event, candidate) {
      counts.calls += 1;
      const key = JSON.stringify([event.id, candidate]);
      if (answers.has(key)) return answers.get(key) as boolean | null;
      const verdict = evaluate(event, candidate);
      answers.set(key, verdict);
      return verdict;
    },
    counts: () => ({ ...counts, queries: probeCounts.queries, probe_runs: probeCounts.probe_runs }),
  };
}
