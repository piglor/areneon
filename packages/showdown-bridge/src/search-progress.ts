// The decision-time search as a seat of the progress evaluation (ADR-026 section 5, slice P): the search policy (a frozen pair, a
// play mode and a `SearchConfig`) as the candidate, and the control it is judged against, the SAME pair played without search in
// that mode (finding N-C, the `self-no-search` opponent kind). Both seats are `PolicySeat.make` factories, so they are built by
// `policyGame` with exactly the options `policyActor` gets (battle name, own team, channel class, seed, game) and guarded as role
// `policy`. Nothing searched leaves the actor: the counts below are integers, wall time is printed by the caller and never stored.

import { enumerateCandidates } from "./choices.ts";
import type { Actor } from "./runner.ts";
import { type PolicyActorOptions, policyActor, type PolicySeat } from "./policy-actor.ts";
import type { PlaySettings } from "./policy-protocol.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import { CONTROL_FALLBACK_KEYS, SEARCH_COUNT_KEYS, SEARCH_TABLE_KEYS, type SearchCounts } from "./progress-records.ts";
import { DROP_REASONS, FALLBACK_REASONS, searchActor, type SearchActor, type SearchCounts as ActorCounts, type SearchSetup } from "./search-actor.ts";

const table = (counts: Record<string, number>): SearchCounts["refusals"] => Object.keys(counts).sort().map((key) => ({ key, count: counts[key] as number }));

/** The actor's per-game counts in the record's shape: the open-keyed tables become lists sorted by key. */
export function searchCountsRecord(counts: ActorCounts): SearchCounts {
  const out: Record<string, unknown> = {};
  for (const key of SEARCH_COUNT_KEYS) out[key] = counts[key];
  out.fallbacks = Object.fromEntries(FALLBACK_REASONS.map((key) => [key, counts.fallbacks[key]]));
  out.worlds_dropped = Object.fromEntries(DROP_REASONS.map((key) => [key, counts.worlds_dropped[key]]));
  for (const key of SEARCH_TABLE_KEYS) out[key] = table(counts[key]);
  return out as unknown as SearchCounts;
}

export interface SearchSeatOptions {
  setup: SearchSetup;
  client: PolicyClient & EvaluationClient;
  /** The search policy id and identity hash (`searchPolicy`). */
  policy_id: string;
  identity_hash: string;
  mode: "sample" | "argmax";
  play?: PlaySettings;
  /** Called with each game's actor as it is built: the caller reads `report()` after the game. */
  onActor: (actor: SearchActor) => void;
}

/** The search candidate's seat: a `searchActor` over the pair, in `mode` for every decision it does not search. */
export function searchSeat(o: SearchSeatOptions): PolicySeat {
  return {
    client: o.client,
    policy_id: o.policy_id,
    sampling: o.mode,
    ...(o.play === undefined ? {} : { play: o.play }),
    make: (options: PolicyActorOptions) => {
      const actor = searchActor({
        setup: o.setup,
        client: o.client,
        battle: options.battle,
        player: options.player,
        team: options.team,
        manifest: options.manifest,
        channel_class: options.channel_class,
        seed: options.seed,
        game: options.game,
        policy_id: o.policy_id,
        identity_hash: o.identity_hash,
        sampling: o.mode,
        ...(o.play === undefined ? {} : { play: o.play }),
      });
      o.onActor(actor);
      return actor;
    },
  };
}

/** The control's counts: decisions, and the fallbacks it shares with the search because they are structural. */
export interface ControlTally {
  counts(): SearchCounts;
}

export interface ControlSeatOptions {
  client: PolicyClient;
  /** The pair's ordinary policy id in `mode` (no `search` member). */
  policy_id: string;
  mode: "sample" | "argmax";
  play?: PlaySettings;
  onTally: (tally: ControlTally) => void;
}

type Structural = (typeof CONTROL_FALLBACK_KEYS)[number];

/** Whether the search would not have searched this request for a structural reason (the enumerator's refusal is the pair's own error). */
function structural(request: Parameters<Actor>[0]): Structural | null {
  if (request.phase === "preview") return "team_preview";
  if (request.phase === "forceSwitch") return "forced_switch";
  return enumerateCandidates(request).candidates.filter((candidate) => !candidate.pending).length < 2 ? "too_few_candidates" : null;
}

/**
 * The no-search control: the very pair, played by `policyActor`, counting the decisions it answers and, by the search's own
 * structural rules (Team Preview, a forced switch, fewer than two non-pending candidates), the ones the search would not have
 * searched either; nothing else (a decision the search could have searched is not a fallback of the control: it was never asked to).
 */
export function controlSeat(o: ControlSeatOptions): PolicySeat {
  return {
    client: o.client,
    policy_id: o.policy_id,
    sampling: o.mode,
    ...(o.play === undefined ? {} : { play: o.play }),
    make: (options: PolicyActorOptions) => {
      const inner: Actor = policyActor(options);
      const counts: ActorCounts = {
        decisions: 0,
        searched_decisions: 0,
        fallbacks: Object.fromEntries(FALLBACK_REASONS.map((key) => [key, 0])) as ActorCounts["fallbacks"],
        worlds_asked: 0,
        worlds_built: 0,
        worlds_refused: 0,
        redraws: 0,
        legality_redraws: 0,
        refusals: {},
        refusal_check_class: {},
        belief_refusals: {},
        worlds_dropped: Object.fromEntries(DROP_REASONS.map((key) => [key, 0])) as ActorCounts["worlds_dropped"],
        resampled: {},
        worlds_scored: 0,
        turns: 0,
        wait_continuations: 0,
        leaf_queries: 0,
        leaf_memo_hits: 0,
        leaf_game_end: 0,
        leaf_hp_ko: 0,
        replies_max_damage: 0,
        replies_all_zero: 0,
        replies_uniform: 0,
      };
      o.onTally({ counts: () => searchCountsRecord(counts) });
      const actor: Actor = async (request, context) => {
        counts.decisions += 1;
        const reason = structural(request);
        if (reason !== null) counts.fallbacks[reason] += 1;
        return inner(request, context);
      };
      return actor;
    },
  };
}
