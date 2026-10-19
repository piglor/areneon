import type { DecisionContext } from "@areneon/schema/types/v1/decision-context.ts";
import {
  type CollectionInput,
  type DataClass,
  type Engine,
  type Episode,
  higherClass,
  readEpisode,
  rightsFor,
  type StampedTeam,
  type SyntheticTeamHashes,
  teamHash,
  type VersionManifest,
} from "@areneon/showdown-bridge";
import { A2_BUDGET_CONFIG, type BudgetConfig, type TimerRules } from "./deadline.ts";
import type { A2BattleRecord, DecisionRecord } from "./session.ts";

/** An A2 battle record that cannot become an Episode. The message never quotes server text. */
export class A2EpisodeError extends Error {
  override name = "A2EpisodeError";
}

export interface A2EpisodeOptions {
  /** The bot's own team as it brought it; its class is proven by `stampTeam`. */
  team: StampedTeam;
  /** The synthetic teams there are (hashes): a `synthetic` team label outside it is refused. */
  known_synthetic: SyntheticTeamHashes;
  /** The battle's version manifest (`resolveFormat(...).manifest`). */
  manifest: VersionManifest;
  /** The format's VGC Timer settings (`timerRulesOf`): the session does not keep them. */
  timer_rules: TimerRules;
  /** The policy that chose the bot's actions. */
  behaviour_policy_id: string;
  /**
   * How the battle was collected (ADR-009 §4). A `league` or `evaluation`
   * record has `seed: null`: a challenge cannot be seeded. `opponent_team_data_class`
   * is not the caller's to give; it is set from `opponent_team`.
   */
  collection: CollectionInput;
  /**
   * The opponent's team class, when the harness knows it. The bot never sees
   * the opponent's full team (Closed Team Sheet), so it cannot prove the class
   * itself: without this the opponent counts as `private`, and so does the
   * Episode. Never stored. A sparring opponent is a human, whose team is never
   * known: giving one is refused.
   */
  opponent_team?: StampedTeam;
  /** The budget configuration the session ran with (its `safety_margin_ms` is recorded). Default `A2_BUDGET_CONFIG`. */
  budget?: Pick<BudgetConfig, "safety_margin_ms">;
}

/**
 * The format's VGC Timer settings from the pinned engine's rule table
 * (`timerstarting` and its siblings), in whole seconds. `tick` is the server's
 * own timer tick, which is not a rule, so the caller gives it. A format
 * without all five rules is refused: there is no timer to record.
 */
export function timerRulesOf(engine: Engine, formatId: string, tick: number): TimerRules {
  const { Dex } = engine.sim;
  const values = Dex.formats.getRuleTable(Dex.formats.get(formatId)).valueRules;
  const read = (rule: string): number => {
    const value = Number(values.get(rule));
    if (!Number.isSafeInteger(value) || value < 0) throw new A2EpisodeError(`${formatId} has no whole-second ${rule} rule`);
    return value;
  };
  if (!Number.isSafeInteger(tick) || tick < 1) throw new A2EpisodeError("the server timer tick must be a positive whole number of seconds");
  return {
    starting: read("timerstarting"),
    grace: read("timergrace"),
    add_per_turn: read("timeraddperturn"),
    max_per_turn: read("timermaxperturn"),
    max_first_turn: read("timermaxfirstturn"),
    tick,
  };
}

type EpisodeDecision = Episode["decisions"][number];

/** The deadline outcome the session's record supports; anything else is refused, never guessed. */
function outcomeOf(decision: DecisionRecord): NonNullable<EpisodeDecision["deadline"]>["deadline_outcome"] {
  // A decision with a watchdog reason and no timer snapshot never gets here: `decisionOf` refuses it.
  if (decision.watchdog !== undefined) return "fallback_timeout";
  if (decision.budget.policy_budget_ms === 0) return "skipped_low_budget";
  if (decision.elapsed_ms > decision.budget.submit_within_ms) {
    throw new A2EpisodeError(`decision ${decision.request_seq} overran its submit budget with no fallback recorded`);
  }
  return "in_budget";
}

function decisionOf(
  decision: DecisionRecord,
  next: DecisionRecord | undefined,
  { behaviour_policy_id, timer_rules, budget }: Pick<A2EpisodeOptions, "behaviour_policy_id" | "timer_rules" | "budget">,
): EpisodeDecision {
  const { timer } = decision;
  if (timer === null) {
    throw new A2EpisodeError(`decision ${decision.request_seq} has no timer snapshot: its deadline record cannot be written`);
  }
  const elapsed = Math.max(0, Math.ceil(decision.elapsed_ms));
  const context: DecisionContext = {
    timer_source: timer.timer_source,
    timer_rules,
    turn_seconds_left: timer.turn_seconds_left,
    bank_seconds_left: timer.bank_seconds_left,
    grace_seconds_left: timer.grace_seconds_left,
    ...(decision.remaining_decisions_estimate === undefined
      ? {}
      : { remaining_decisions_estimate: decision.remaining_decisions_estimate }),
    safety_margin_ms: (budget ?? A2_BUDGET_CONFIG).safety_margin_ms,
    submit_within_ms: decision.budget.submit_within_ms,
    policy_budget_ms: decision.budget.policy_budget_ms,
  };
  return {
    identity: { request_seq: decision.channel_request_seq, request_hash: decision.request_hash, rqid: decision.rqid },
    choice: decision.choice,
    source_policy_id: behaviour_policy_id,
    decision_context: context,
    deadline: {
      deadline_outcome: outcomeOf(decision),
      elapsed_ms: elapsed,
      // The session decides and sends in one step (Team Preview's send waits for the
      // Open Team Sheets confirmation, which `ots.ts` bounds and forfeits on).
      submitted_after_ms: elapsed,
      // Only a fresh snapshot is "the next one": an update after an `[Unavailable choice]` inherits its timer.
      bank_seconds_after: next === undefined || next.inherited || next.timer === null ? null : next.timer.bank_seconds_left,
      server_auto_choose_detected: false,
    },
  };
}

/**
 * The team as the server played it. The server removes nicknames before a
 * battle (`removeNicknames` in sim/team-validator.ts), so the channel names
 * each Pokémon by species, whatever the bot's paste says. This takes each
 * set's name from the same position of the channel's own Team Preview request
 * (the order is the team's), only where the set names it otherwise. Whether
 * the sets are this player's is `readEpisode`'s check, not this one's.
 */
function playedTeam(team: StampedTeam, lines: readonly string[]): StampedTeam {
  const first = lines.find((line) => line.startsWith("|request|"));
  if (first === undefined) throw new A2EpisodeError("the channel holds no request");
  let idents: unknown;
  try {
    idents = (JSON.parse((first as string).slice("|request|".length)) as { side?: { pokemon?: { ident?: unknown }[] } }).side?.pokemon?.map(
      (mon) => mon.ident,
    );
  } catch {
    idents = undefined;
  }
  if (!Array.isArray(idents) || idents.length !== team.team.length || !idents.every((ident) => typeof ident === "string")) {
    throw new A2EpisodeError("the channel's first request does not list the bot's team");
  }
  return {
    ...team,
    team: team.team.map((set, i) => {
      const ident = idents[i] as string;
      const name = ident.slice(ident.indexOf(": ") + 2);
      return set.name === "" || set.name === name ? set : { ...set, name };
    }) as StampedTeam["team"],
  };
}

/** The turn count the channel ends on: its last `|turn|` line. */
function turnsOf(lines: readonly string[]): number {
  let turns = 0;
  for (const line of lines) {
    const [, kind, count] = line.split("|");
    if (kind === "turn") turns = Number(count);
  }
  return turns;
}

/**
 * The bot's own Episode of one finished A2 battle (ADR-009 slice f, ADR-008
 * §3), from the record the session hands over. Only the bot's own channel,
 * team and decisions go in, each decision with its deadline record and timer
 * context. Nothing of the opponent is read except the class its team was
 * stamped with.
 *
 * The class is the higher of both teams' (ADR-008 §5). The bot never sees the
 * opponent's team, so the opponent is `private` unless the caller gives a
 * stamped `opponent_team`: an A2 Episode is private unless the harness knows
 * both teams are synthetic. A sparring Episode is always private.
 *
 * It fails closed (`A2EpisodeError`) on a battle that did not finish, ended on
 * a fatal error, forfeited its Open Team Sheets decline, was classed a defect,
 * had a server auto-choose, spans several connections (a rejoin's channel
 * restarts, so no one channel holds the battle), or has a decision without a
 * timer snapshot. The result is read back through `readEpisode`, so it is
 * valid by the same checks a stored copy gets (`EpisodeError`).
 *
 * Reading a stored synthetic Episode back: the team is stored under the names
 * the server gave it, so `readEpisode` proves it synthetic only if its
 * `known_synthetic` holds the hash of that played form (when it differs from
 * the paste's). The writer adds that hash itself for its own read-back; a
 * loader must do the same.
 *
 * `refused` is not set: the session's record keeps the refused choice's
 * request, not the engine's message. The update request that follows an
 * `[Unavailable choice]` is its own decision with its own identity.
 */
export async function a2Episode(record: A2BattleRecord, options: A2EpisodeOptions): Promise<Episode> {
  const { team, known_synthetic, manifest, behaviour_policy_id, collection, opponent_team } = options;
  if (record.fatal !== null) throw new A2EpisodeError("the battle ended on a fatal error");
  if (record.result === "unfinished") throw new A2EpisodeError("the battle did not finish");
  if (record.ots !== "confirmed" || record.ots_failure !== null) throw new A2EpisodeError("the Open Team Sheets decline was not confirmed");
  if (record.defect) throw new A2EpisodeError("the battle had a defect: the engine refused a non-pending candidate");
  if (record.server_auto_chooses.length > 0) throw new A2EpisodeError("the server auto-chose for the bot");
  if (record.player === null) throw new A2EpisodeError("the battle has no player seat");
  const { player } = record;
  const [channel, ...rest] = record.channels;
  if (channel === undefined || rest.length > 0) {
    throw new A2EpisodeError(`the battle has ${record.channels.length} channels: an Episode needs exactly one connection's channel`);
  }
  if (channel.player !== player) throw new A2EpisodeError("the channel is not the bot's own");
  if (collection.kind === "sparring" && opponent_team !== undefined) {
    throw new A2EpisodeError("a sparring opponent is a human: their team class is never known");
  }
  const opponent: DataClass = collection.kind === "sparring" ? "private" : (opponent_team?.data_class ?? "private");
  const data_class = collection.kind === "sparring" ? "private" : higherClass(team.data_class, opponent);
  const stored: Episode["provenance"]["collection"] =
    collection.kind === "sparring" ? collection : { ...collection, opponent_team_data_class: opponent };
  const played = playedTeam(team, channel.lines);
  // The class of the paste carries over to the same sets under the names the server gave them.
  const known = team.data_class === "synthetic" && known_synthetic.has(teamHash(team.team))
    ? new Set([...known_synthetic, teamHash(played.team)])
    : known_synthetic;
  const decisions = record.decisions.map((decision, i) => decisionOf(decision, record.decisions[i + 1], options));
  const rival = player === "p1" ? "p2" : "p1";
  return await readEpisode(
    {
      schema_version: 1,
      data_class,
      rights: rightsFor(data_class),
      provenance: { runtime: "A2", behaviour_policy_id, manifest, collection: stored },
      player,
      team: played,
      channel: channel.lines,
      decisions,
      result: { winner: record.result === "win" ? player : record.result === "loss" ? rival : null, turns: turnsOf(channel.lines) },
    },
    { known_synthetic: known },
  );
}
