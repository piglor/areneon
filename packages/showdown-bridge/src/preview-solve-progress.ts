// The Team Preview solver as a seat of the progress evaluation (ADR-029 section 8, ticket #1260): the solver policy (a frozen pair, a
// play mode and a `PreviewSolveConfig`) as the candidate, and the control it is judged against, the SAME pair played without the
// solver in that mode (the `self-no-solve` opponent kind, as ADR-026's `self-no-search`). Both seats are `PolicySeat.make`
// factories, so `policyGame` builds them with exactly the options `policyActor` gets and guards them as role `policy`. Nothing
// solved leaves the actor: the counts below are integers, wall time is printed by the caller and never stored.

import { type ActorFeatures, policyActor, type PolicyActorOptions, type PolicySeat } from "./policy-actor.ts";
import type { PlaySettings } from "./policy-protocol.ts";
import type { PolicyClient } from "./policy-worker.ts";
import { type PreviewSolveActor, type PreviewSolveCounts, previewSolveActor, type PreviewSolveSetup } from "./preview-solve.ts";
import type { PayoffClient } from "./preview-payoff.ts";
import { SOLVE_COUNT_KEYS, SOLVE_FALLBACK_KEYS, SOLVE_PAYOFF_KEYS, type SolveCounts } from "./progress-records.ts";
import type { Actor } from "./runner.ts";

/** The actor's per-game counts in the record's shape: the open-keyed belief refusals become a list sorted by key. */
export function solveCountsRecord(counts: PreviewSolveCounts): SolveCounts {
  const out: Record<string, unknown> = {};
  for (const key of SOLVE_COUNT_KEYS) out[key] = counts[key];
  out.fallbacks = Object.fromEntries(SOLVE_FALLBACK_KEYS.map((key) => [key, counts.fallbacks[key]]));
  out.payoff = Object.fromEntries(SOLVE_PAYOFF_KEYS.map((key) => [key, counts.payoff[key]]));
  out.belief_refusals = Object.keys(counts.belief_refusals)
    .sort()
    .map((key) => ({ key, count: counts.belief_refusals[key] as number }));
  return out as unknown as SolveCounts;
}

export interface SolveSeatOptions {
  setup: PreviewSolveSetup;
  client: PolicyClient & PayoffClient;
  /** The solver policy id and identity hash (`previewSolvePolicy`). */
  policy_id: string;
  identity_hash: string;
  mode: "sample" | "argmax";
  play?: PlaySettings;
  /** The candidate-feature records of the pair (absent: it reads none): the seat's rows and the solver's hypothetical rows come from them. */
  features?: ActorFeatures;
  /** Called with each game's actor as it is built: the caller reads `report()` after the game. */
  onActor: (actor: PreviewSolveActor) => void;
}

/** The solver candidate's seat: a `previewSolveActor` over the pair's own `policyActor` (every request but Team Preview, and the fallback's). */
export function solveSeat(o: SolveSeatOptions): PolicySeat {
  return {
    client: o.client,
    policy_id: o.policy_id,
    sampling: o.mode,
    ...(o.play === undefined ? {} : { play: o.play }),
    ...(o.features === undefined ? {} : { features: o.features }),
    make: (options: PolicyActorOptions) => {
      const actor = previewSolveActor({
        ...(o.features === undefined ? {} : { features: { configs: o.features.configs } }),
        setup: o.setup,
        client: o.client,
        battle: options.battle,
        player: options.player,
        policy_id: o.policy_id,
        identity_hash: o.identity_hash,
        seed: options.seed,
        game: options.game,
        channel_class: options.channel_class,
        base: policyActor(options),
      });
      o.onActor(actor);
      return actor;
    },
  };
}

/** The control's counts: the requests it answered and the Team Previews among them (where the solver would have acted). */
export interface SolveControlTally {
  counts(): SolveCounts;
}

export interface SolveControlSeatOptions {
  client: PolicyClient;
  /** The pair's ordinary policy id in `mode` (no `preview_solve` member). */
  policy_id: string;
  mode: "sample" | "argmax";
  play?: PlaySettings;
  /** The pair's candidate-feature records (absent: it reads none). */
  features?: ActorFeatures;
  onTally: (tally: SolveControlTally) => void;
}

/** The no-solve control: the very pair, played by `policyActor`, counting its decisions and Team Previews; nothing else. */
export function solveControlSeat(o: SolveControlSeatOptions): PolicySeat {
  return {
    client: o.client,
    policy_id: o.policy_id,
    sampling: o.mode,
    ...(o.play === undefined ? {} : { play: o.play }),
    ...(o.features === undefined ? {} : { features: o.features }),
    make: (options: PolicyActorOptions) => {
      const inner: Actor = policyActor(options);
      const counts: PreviewSolveCounts = {
        decisions: 0,
        previews: 0,
        solved: 0,
        fallbacks: Object.fromEntries(SOLVE_FALLBACK_KEYS.map((key) => [key, 0])) as PreviewSolveCounts["fallbacks"],
        belief_refusals: {},
        worlds_asked: 0,
        worlds_refused: 0,
        worlds_unstarted: 0,
        payoff: Object.fromEntries(SOLVE_PAYOFF_KEYS.map((key) => [key, 0])) as unknown as PreviewSolveCounts["payoff"],
        exploitability_ppm: 0,
        support_size: 0,
      };
      o.onTally({ counts: () => solveCountsRecord(counts) });
      const actor: Actor = async (request, context) => {
        counts.decisions += 1;
        if (request.phase === "preview") counts.previews += 1;
        return inner(request, context);
      };
      return actor;
    },
  };
}
