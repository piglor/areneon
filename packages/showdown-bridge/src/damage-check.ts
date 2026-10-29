// The engine-backed damage check of the opponent belief (ADR-025 section 7, ticket #1132). It is the port the belief
// reads (`DamageCheck` of `packages/opponent-model`, declared structurally here because the bridge does not import the
// model): the isolated hits of the player's own channel (`damage-hits.ts`, with the reviewed rules of
// `BeliefConfig.damage_evidence`) as events about opponent slots, and for a candidate set of the slot whether the ADR-016
// damage probe produces some roll whose shown hit points equal what the player saw.
//
// Only the probe computes. The opponent side of a hit is the candidate set (declared at the shown hit points and the
// reader's stages, status and field); the own side is the player's own Pokemon from the request in force at the move
// line and the own team (its exact hit points are in the channel). The rolls are the probe's no-crit vector, or the
// crit vector when the channel showed `-crit`. A shown opponent percentage stands for every hit-point value the
// reviewed display rule maps to it (`hp-display.ts`), and the check passes when some roll takes some such value to some
// value that shows the observed percentage. A candidate the probe cannot answer (`unsupported`, a set the validator
// refuses, a Mega forme) is `null`: it adds no constraint.

import { type PublicFrame, type PublicMon, type PublicRun, type PublicState, interpretChannel } from "@areneon/public-state";
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { DamageAwareError, type Fighter, ProbeRefusal, type ProbeMemo, damageView } from "./damage-aware-view.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import { type Engine, type PokemonSet, defaultIvs } from "./engine.ts";
import { type Hit, extractHits } from "./damage-hits.ts";
import { type HpRange, hpRange, rollsReach } from "./hp-display.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "./mechanics-query.ts";
import type { RulesSnapshot } from "./rules.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { exportPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** Bumped when the check declares a hit to the probe or judges its rolls differently. */
export const DAMAGE_CHECK_FORMAT = 1;

export type DamageEvidenceRules = NonNullable<BeliefConfig["damage_evidence"]>;

/** Structurally the opponent model's `DamageEvent`. */
export interface DamageEventPort {
  readonly id: string;
  readonly slot: number;
  readonly role: "attacker" | "defender";
}

/** Structurally the opponent model's `DamageCandidate`. */
export interface DamageCandidatePort {
  readonly species: string;
  readonly forme: string;
  readonly mega: boolean;
  readonly ability: string;
  readonly item: string | null;
  readonly nature: string;
  readonly stat_points: Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;
}

/** Structurally the opponent model's `DamageCheck`. */
export interface DamageCheckPort {
  readonly events: readonly DamageEventPort[];
  check(event: DamageEventPort, candidate: DamageCandidatePort): boolean | null;
}

/** What the check found in the channel and did, in integers (the cost note and the tests read it). */
export interface DamageCheckCounts {
  /** `|move|` lines that dealt damage. */
  hits: number;
  /** Hits that were isolated and concern an opponent slot (the events). */
  events: number;
  /** Hits that were not isolated, by the first reason. */
  not_isolated: Record<string, number>;
  /** `check` calls (a repeated call counts each time). */
  calls: number;
  /** Probe damage queries asked. */
  queries: number;
  probe_runs: number;
}

export interface DamageCheckContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  /** The ADR-018 record (it names the format; `hp_rounding` is the inverse partner of `rules.shown_hp_rounding`). */
  readonly config: DamageAwareConfig;
  readonly mechanics: MechanicsConfig;
  /** The belief config with its `damage_evidence`: the reviewed rules and the identity-hiding abilities. */
  readonly belief: BeliefConfig;
  readonly snapshot?: RulesSnapshot;
  readonly data_class: "synthetic" | "private";
  /** Test seam: told the probe's rolls and the two hit-point ranges of every judged candidate. */
  readonly observe?: (event: DamageEventPort, rolls: readonly number[], before: HpRange, after: HpRange) => void;
  /** Test seam: the probe (default `probeMechanics`). */
  readonly probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
}

export interface DamageCheckInput {
  /** The player's seat (`p1` or `p2`). */
  readonly player: string;
  /** The player's own channel up to the request being answered. */
  readonly lines: readonly string[];
  /** The player's own team as brought. */
  readonly team: readonly Readonly<PokemonSet>[];
  /** The reader's run over `lines` when the caller has it . */
  readonly run?: PublicRun;
}

export interface DamageCheckHandle extends DamageCheckPort {
  readonly hits: readonly Hit[];
  counts(): DamageCheckCounts;
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const SHOWN_GENDER = /, ([MF])(?:,|$)/;

export function createDamageCheck(context: DamageCheckContext, input: DamageCheckInput): DamageCheckHandle | null {
  const { engine, vocabulary, config, mechanics, belief, data_class } = context;
  if (belief.damage_evidence === undefined) return null;
  const rules: DamageEvidenceRules = belief.damage_evidence;
  const snapshot = context.snapshot ?? loadRulesSnapshot(engine.commit);
  const probe = context.probe ?? probeMechanics;
  const run = input.run ?? interpretChannel(input.player, input.lines);
  if (run.flags.length > 0 || run.frames.length === 0) return null;
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const probeCounts = { queries: 0, probe_runs: 0, unsupported_queries: 0 };
  const memo: ProbeMemo = new Map();
  const view = damageView({ engine, vocabulary, config, v1: config, mechanics, snapshot, data_class, probe, counts: probeCounts, memo });
  const dex = view.dex;
  const names = dex as unknown as Record<"species" | "abilities" | "moves" | "items", { get(id: string): { name: string; baseSpecies: string } }>;
  const requests = run.requests as EngineRequest[];
  // The reviewed target types are checked against the engine's own move data: a mistyped one would silently never match.
  const targets = new Set((dex as unknown as { moves: { all(): { target: string }[] } }).moves.all().map((m) => m.target));
  for (const target of rules.non_isolating.targets) if (!targets.has(target)) throw new DamageAwareError("a reviewed non-isolating target type is not a move target of the engine");

  // The abilities a Pokemon in play may hold: an own Pokemon's from its latest request, an opponent's revealed one, else its species' list.
  const ownAbility = new Map<string, string>();
  const latest = requests[requests.length - 1];
  for (const entry of latest?.side.pokemon ?? []) ownAbility.set(entry.ident.slice(entry.ident.indexOf(": ") + 2), toId(entry.ability ?? entry.baseAbility ?? ""));
  const abilitiesOf = (mon: PublicMon): readonly string[] => {
    if (mon.own) {
      const ability = ownAbility.get(mon.name ?? "");
      return ability === undefined || ability === "" ? rules.non_isolating.abilities : [ability];
    }
    if (mon.ability !== null && !mon.ability_received) return [mon.ability];
    const entry = roster.get(mon.temp_species ?? mon.species ?? mon.base_species) ?? roster.get(mon.base_species);
    return entry === undefined ? rules.non_isolating.abilities : entry.abilities;
  };
  const hits = extractHits(run.frames, {
    rules: rules.non_isolating,
    targetOf: (move) => {
      const found = dex.moves.get(move);
      return found.exists ? found.target : null;
    },
    abilitiesOf,
    mayHide: (mon) => [mon.base_species, ...(mon.species === null ? [] : [mon.species])].some((id) => roster.get(id)?.abilities.some((a) => belief.identity_hiding_abilities.includes(a))),
    declarable: (state: PublicState) => state.field_conditions.every((id) => id === "trickroom" || id === state.terrain || dex.conditions.get(id).effectType === "Terrain"),
  });

  const counts: DamageCheckCounts = { hits: hits.length, events: 0, not_isolated: {}, calls: 0, queries: 0, probe_runs: 0 };
  const events: DamageEventPort[] = [];
  const byEvent = new Map<string, Hit>();
  for (const hit of hits) {
    if (hit.reason !== null) {
      counts.not_isolated[hit.reason] = (counts.not_isolated[hit.reason] ?? 0) + 1;
      continue;
    }
    // One side must be an opponent Pokemon, the other the player's own.
    const foe = hit.attacker.own ? hit.defender : hit.attacker;
    const mon = hit.state.opponent.roster[foe.index];
    if (mon === undefined || mon.mega) continue;
    const event: DamageEventPort = { id: `hit${hit.frame}`, slot: foe.index, role: hit.attacker.own ? "defender" : "attacker" };
    events.push(event);
    byEvent.set(event.id, hit);
  }
  counts.events = events.length;

  const natureName = (id: string): string | undefined => vocabulary.natures.find((n) => n.id === id)?.name;
  const answers = new Map<string, boolean | null>();

  /**
   * The own Pokemon of a hit as a probe fighter, or null when the probe cannot state it: its set is its team set with the
   * item, ability and gender the request in force at the move line shows, and its state is the reader's. A Pokemon that has
   * Mega Evolved (the declared Evolution can start an ability that changes the other combatant, Intimidate) or whose forme
   * is not the one of its set (Palafin's Hero forme) is not stated.
   */
  function ownFighter(hit: Hit, side: { letter: string; index: number }): Fighter | null {
    const frame = run.frames[hit.frame] as PublicFrame;
    const mon = hit.state.own.roster[side.index] as PublicMon;
    const request = requests[frame.request - 1];
    if (request === undefined) throw new DamageAwareError("the channel has no request before a hit");
    const entry = request.side.pokemon.find((p) => p.ident.slice(p.ident.indexOf(": ") + 2) === mon.name);
    if (entry === undefined) throw new DamageAwareError("an own Pokemon is not in the request in force");
    const name = mon.name as string;
    const matches = input.team.filter((set) => [set.name, set.name === "" ? set.species : "", names.species.get(set.species).baseSpecies].includes(name));
    if (matches.length !== 1) throw new DamageAwareError("an own Pokemon matches no set, or more than one, of the team as brought");
    const set: PokemonSet = structuredClone(matches[0] as PokemonSet);
    const shownForme = toId(entry.details.split(",")[0] ?? "");
    // A Mega Evolution declared to the probe can start an ability that changes the other combatant (Intimidate): not stated.
    if (mon.mega || shownForme !== toId(set.species)) return null;
    const gender = SHOWN_GENDER.exec(entry.details)?.[1];
    if (gender !== undefined) set.gender = gender;
    set.item = entry.item === "" ? "" : names.items.get(entry.item).name;
    if (!mon.mega) set.ability = names.abilities.get(entry.ability ?? entry.baseAbility ?? set.ability).name;
    const hp = mon.hp_num === null || mon.hp_den === null || mon.hp_num === mon.hp_den ? null : Math.max(1, mon.hp_num);
    return { paste: exportPaste(engine, [set]), hp, status: mon.status, boosts: view.boostsOf(mon), mega: mon.mega };
  }

  function evaluate(event: DamageEventPort, candidate: DamageCandidatePort): boolean | null {
    const hit = byEvent.get(event.id) as Hit;
    if (candidate.mega) return null;
    const entry = roster.get(candidate.forme);
    const nature = natureName(candidate.nature);
    if (entry === undefined || nature === undefined) return null;
    const foeSide = event.role === "defender" ? hit.defender : hit.attacker;
    const foeMon = hit.state.opponent.roster[foeSide.index] as PublicMon;
    const moves = event.role === "attacker" ? [hit.move] : [entry.moves[0] as string];
    if (event.role === "attacker" && !entry.moves.includes(hit.move)) return null;
    const set: PokemonSet = {
      name: "",
      species: names.species.get(candidate.forme).name,
      item: candidate.item === null ? "" : names.items.get(candidate.item).name,
      ability: names.abilities.get(candidate.ability).name,
      moves: moves.map((m) => names.moves.get(m).name),
      nature,
      gender: foeMon.gender ?? "",
      evs: { ...candidate.stat_points },
      ivs: defaultIvs(),
      level: foeMon.level ?? vocabulary.rules.adjust_level,
    };
    let paste: string;
    try {
      paste = exportPaste(engine, [set]);
    } catch {
      return null;
    }
    try {
      const stats = view.query({ kind: "stats", set: paste });
      if (stats?.kind !== "computed" || stats.answer.kind !== "stats") return null;
      const max = stats.answer.stats.hp;
      // The foe's shown hit points are a range of hit-point values; its fighter is declared at the lowest.
      const range = (percent: number): HpRange | null => hpRange(percent, max, rules.shown_hp_rounding);
      const foeNow = foeMon.hp_num;
      const own = ownFighter(hit, event.role === "defender" ? hit.attacker : hit.defender);
      if (own === null) return null;
      let before: HpRange | null;
      let after: HpRange | null;
      let attacker: Fighter;
      let defender: Fighter;
      if (event.role === "defender") {
        before = range(hit.before.num);
        after = range(hit.after.num);
        if (before === null || after === null) return null;
        attacker = own;
        defender = { paste, hp: hit.before.num === 100 ? null : Math.max(1, before.lo), status: foeMon.status, boosts: view.boostsOf(foeMon), mega: false };
      } else {
        const hpNow = foeNow === null ? null : range(foeNow);
        if (hpNow === null) return null;
        before = { lo: hit.before.num, hi: hit.before.num };
        after = { lo: hit.after.num, hi: hit.after.num };
        attacker = { paste, hp: foeNow === 100 ? null : Math.max(1, hpNow.lo), status: foeMon.status, boosts: view.boostsOf(foeMon), mega: false };
        defender = own;
      }
      const result = view.damageResult(attacker, defender, hit.move, hit.state, event.role === "defender");
      if (result?.kind !== "computed" || result.answer.kind !== "damage") return null;
      const rolls = hit.crit ? result.answer.crit : result.answer.no_crit;
      context.observe?.(event, rolls, before, after);
      return rollsReach(rolls, before, after);
    } catch (error) {
      if (error instanceof ProbeRefusal || (error instanceof DamageAwareError && error.message.startsWith("the probe refused"))) return null;
      throw error;
    }
  }

  return {
    events,
    hits,
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
