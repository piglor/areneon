// The pure decision of the search and the label built on it (ADR-026 sections 2 and 4; erratum E2 decision 2, SD-b).
//
// `searchDecision` is what `searchActor` decides: worlds of the opponent from the belief, the M replies of each world, one turn
// per (candidate, reply), the leaf, the weighted sum per candidate. It reads one `SearchInput` and the ports in `SearchDecisionDeps`
// and nothing else (no live battle, record, omniscient log, hidden verdict, opponent team or seed), so the actor and the labeller
// share it and a same input gives the same decision. `searchLabel` adds the searched set (`search-set.ts`) and reports, besides
// the choice, each searched candidate's `q_ppm`, the prior of every candidate and the reason a decision is not labelled.

import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs } from "./canonical-json.ts";
import { type Candidate, enumerateCandidates } from "./choices.ts";
import type { Engine } from "./engine.ts";
import type { VersionManifest } from "./formats.ts";
import { observe, playerChannel } from "./player-channel.ts";
import type { TeamSheets } from "./team-sheet-binding.ts";
import type { ValueQueryInput } from "./policy-protocol.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import { decodeRequest } from "./request.ts";
import { legalReplies, ReplyModelError, type ReplyModel } from "./reply-model.ts";
import { sha256Hex } from "./rules.ts";
import type { Player } from "./runner.ts";
import type { SearchCounts, DropReason } from "./search-actor.ts";
import type { SearchConfig } from "./search-config.ts";
import type { CostComponent, CostSink } from "./search-cost.ts";
import { searchedIndices } from "./search-set.ts";
import { hpKoDifferential, outcomeScore } from "./search-leaf.ts";
import { cloneWorld, foeSwitchChoice, playChoices, type TurnResult } from "./search-turn.ts";
import type { DataClass } from "./team-class.ts";
import type { DecisionWorlds, WorldRequest, WorldSampler } from "./world-sampler.ts";

/**
 * What one search decision reads, and nothing else: the world request (own request, own team, the reader's state and flags, the
 * world facts), the real channel up to the request, the candidates to search in prior rank and the 32-byte search stream the
 * harness derived. There is no field for the live battle, a `BattleRecord`, an `OmniscientLog`, hidden verdicts or the seed (a type test).
 */
export interface SearchInput {
  readonly world: WorldRequest;
  readonly channel: readonly string[];
  readonly top: readonly Candidate[];
  readonly stream: Uint8Array;
}

export type SearchDecision =
  | { readonly kind: "fallback"; readonly reason: "belief_refused" | "below_min_valid" }
  | { readonly kind: "choice"; readonly choice: string; readonly scores: readonly { readonly choice: string; readonly sum: number; readonly q_ppm: number }[]; readonly worlds_scored: number };

/** The facts' hash purposes of a world (section 4): the engine seed and the reply draws. */
const ENGINE_PURPOSE = "search-engine/1";
const REPLY_PURPOSE = "search-reply/1";

const hex = (bytes: Uint8Array): string => Buffer.from(bytes).toString("hex");

/** The world's engine seed (`sodium,<32 hex>`): SHA-256 over the JCS of `{purpose: "search-engine/1", search_stream, world}`. */
export const engineSeed = (stream: Uint8Array, world: number): string => `sodium,${sha256Hex(jcs({ purpose: ENGINE_PURPOSE, search_stream: hex(stream), world })).slice(0, 32)}`;

/**
 * Draw `draw` of the world's uniform replies: an index below `n`. SHA-256 over the JCS of `{draw, purpose: "search-reply/1",
 * search_stream, world}`; the first 8 bytes as an unsigned integer scaled by `n` (`floor(u * n / 2^64)`, a bias below `n / 2^64`).
 */
export const replyIndex = (stream: Uint8Array, world: number, draw: number, n: number): number => {
  const u = BigInt(`0x${sha256Hex(jcs({ draw, purpose: REPLY_PURPOSE, search_stream: hex(stream), world })).slice(0, 16)}`);
  return Number((u * BigInt(n)) >> 64n);
};


/** The ports and values a decision reads; `counts` is the caller's tally, which the decision adds to. */
export interface SearchDecisionDeps {
  readonly engine: Engine;
  readonly config: SearchConfig;
  readonly sampler: WorldSampler;
  readonly replyModel: ReplyModel;
  readonly client: PolicyClient & EvaluationClient;
  readonly battle: string;
  readonly player: Player;
  readonly team: PrivateTeamSpec;
  readonly manifest: VersionManifest;
  readonly channel_class: DataClass;
  /** The team-sheet regime the channel is judged against (ADR-027 section 5). */
  readonly regime: TeamSheets;
  readonly policy_id: string;
  /** The worker answers `value` queries (no candidates, no rows). */
  readonly valueOnly: boolean;
  readonly counts: SearchCounts;
  /** Optional wall-clock sink for the cost lines (S0 measure 5); reads nothing back and changes no value. */
  readonly cost?: CostSink;
}

/** `floor(sum / (worlds * 10^6))` clamped to 0..10^6: the weighted mean leaf value in ppm (each world's reply weights sum to 10^6). Integer arithmetic. */
export const qPpm = (sum: number, worlds: number): number => {
  const d = worlds * 1_000_000;
  return Math.min(1_000_000, Math.max(0, (sum - (sum % d)) / d));
};


/** The weights of the M replies of a world, in ppm: the max-damage reply, then the uniform draws sharing the rest. */
const weights = (config: SearchConfig): number[] => {
  const { max_damage_ppm, uniform_draws } = config.replies;
  const share = uniform_draws === 0 ? 0 : (1_000_000 - max_damage_ppm) / uniform_draws;
  return [max_damage_ppm, ...Array.from({ length: uniform_draws }, () => share)];
};

/** The M replies of one world; a world whose opponent cannot be replied in throws `ReplyModelError`. */
function repliesOf(deps: SearchDecisionDeps, world: number, built: Parameters<ReplyModel["reply"]>[0], stream: Uint8Array): { choice: string; weight: number }[] {
  const w = weights(deps.config);
  const { engine, replyModel, counts } = deps;
  const out: { choice: string; weight: number }[] = [];
  if (w[0] !== 0) {
    const reply = replyModel.reply(built);
    counts.replies_max_damage += 1;
    if (reply.kind === "all_zero") counts.replies_all_zero += 1;
    out.push({ choice: reply.choice, weight: w[0] as number });
  }
  if (w.length > 1) {
    // A world's opponent always has a non-pending legal reply (its own request carries no hidden flags); were it ever empty,
    // the undefined choice would make the engine throw, which the turn reports as a refusal and the world is dropped.
    const open = legalReplies(engine, built);
    for (let draw = 0; draw < w.length - 1; draw += 1) {
      out.push({ choice: open[replyIndex(stream, world, draw, open.length)] as string, weight: w[draw + 1] as number });
      counts.replies_uniform += 1;
    }
  }
  return out;
}


export async function searchDecision(input: SearchInput, deps: SearchDecisionDeps): Promise<SearchDecision> {
  const { engine, config, sampler, client, battle, player, team, manifest, channel_class, regime, policy_id, valueOnly, counts, cost } = deps;
  const timeSync = <T>(component: CostComponent, run: () => T): T => {
    if (cost === undefined) return run();
    const began = performance.now();
    try {
      return run();
    } finally {
      cost.add(component, performance.now() - began);
    }
  };
  const timeAsync = async <T>(component: CostComponent, run: () => Promise<T>): Promise<T> => {
    if (cost === undefined) return run();
    const began = performance.now();
    try {
      return await run();
    } finally {
      cost.add(component, performance.now() - began);
    }
  };
  const worlds: DecisionWorlds = timeSync("world_building", () => sampler.sample(input.world, input.stream));
  if (worlds.kind === "refused") return { kind: "fallback", reason: "belief_refused" };
  if (!worlds.searched) return { kind: "fallback", reason: "below_min_valid" };

  const memo = new Map<string, number>();

  /** The value head on the hypothetical channel `input.channel + lines`, which ends at the seat's request (the engine sends it last). */
  async function valueOf(lines: readonly string[], request: string): Promise<number> {
    const key = lines.join("\n");
    const known = memo.get(key);
    if (known !== undefined) {
      counts.leaf_memo_hits += 1;
      return known;
    }
    const observation = await timeAsync("observation_build", () => observe({ channel: playerChannel({ schema_version: 1, data_class: channel_class, player, lines: [...input.channel, ...lines] }, regime), team, manifest }));
    counts.leaf_queries += 1;
    let value_ppm: number;
    if (valueOnly) {
      // Value-only: no candidates and no rows (E2 decision 7); the hypothetical request is not even enumerated.
      value_ppm = (
        await timeAsync("worker_round_trip", () => (client.value as NonNullable<EvaluationClient["value"]>)(battle, {
          schema_version: 1,
          identity: observation.identity,
          observation,
          candidates: [],
          sampling: { kind: "value" },
          policy_id,
        } as ValueQueryInput))
      ).value_ppm;
    } else {
      const candidates = enumerateCandidates(decodeRequest(request)).candidates;
      value_ppm = (
        await timeAsync("worker_round_trip", () =>
          client.evaluate(battle, {
            schema_version: 1,
            identity: observation.identity,
            observation,
            candidates: candidates.map(({ choice, pending }) => ({ choice, pending })),
            sampling: { kind: "evaluate" },
            policy_id,
          } as Parameters<EvaluationClient["evaluate"]>[1]),
        )
      ).value_ppm;
    }
    memo.set(key, value_ppm);
    return value_ppm;
  }

  /** One (candidate, reply) of a world: the leaf value, or the reason the world cannot be scored. */
  async function simulate(serialised: string, seedHex: string, ownChoice: string, foeChoice: string): Promise<{ value: number } | { drop: DropReason }> {
    const clone = timeSync("turn_simulation", () => cloneWorld(engine, serialised, seedHex));
    let turn: TurnResult = timeSync("turn_simulation", () => playChoices(clone, player, { own: ownChoice, foe: foeChoice }));
    counts.turns += 1;
    const lines: string[] = [];
    // A wait request is played on: the opponent must switch, and its choice is the first legal one in canonical order. Each step is a
    // forced-switch turn of the engine (a bench Pokemon is used up), so the chain is as short as the benches.
    for (;;) {
      lines.push(...turn.lines);
      if (turn.refused !== null) return { drop: "choice_refused" };
      if (turn.ended !== null || (JSON.parse(turn.request as string) as { wait?: boolean }).wait !== true) break;
      counts.wait_continuations += 1;
      turn = timeSync("turn_simulation", () => playChoices(clone, player, { own: null, foe: foeSwitchChoice(clone, player) }));
    }
    if (turn.ended !== null) {
      counts.leaf_game_end += 1;
      return { value: outcomeScore(turn.ended) };
    }
    if (config.leaf.kind === "hp-ko-differential") {
      counts.leaf_hp_ko += 1;
      return { value: hpKoDifferential(clone, player) };
    }
    return { value: await valueOf(lines, turn.request as string) };
  }

  const sums = input.top.map(() => 0);
  let scored = 0;
  for (const valid of worlds.valid) {
    let replies;
    try {
      replies = timeSync("max_damage_reply", () => repliesOf(deps, valid.world, valid.built, input.stream));
    } catch (error) {
      if (!(error instanceof ReplyModelError)) throw error;
      counts.worlds_dropped.reply_model += 1;
      continue;
    }
    const serialised = JSON.stringify(valid.built.battle.toJSON());
    const seedHex = engineSeed(input.stream, valid.world);
    const own = input.top.map(() => 0);
    let dropped: DropReason | null = null;
    for (let c = 0; c < input.top.length && dropped === null; c += 1) {
      for (const reply of replies) {
        const result = await simulate(serialised, seedHex, (input.top[c] as Candidate).choice, reply.choice);
        if ("drop" in result) {
          dropped = result.drop;
          break;
        }
        own[c] = (own[c] as number) + reply.weight * result.value;
      }
    }
    if (dropped !== null) {
      counts.worlds_dropped[dropped] += 1;
      continue;
    }
    scored += 1;
    own.forEach((value, c) => {
      sums[c] = (sums[c] as number) + value;
    });
  }
  counts.worlds_scored += scored;
  if (scored < config.min_valid_worlds) return { kind: "fallback", reason: "below_min_valid" };
  const scores = input.top.map((candidate, c) => ({ choice: candidate.choice, sum: sums[c] as number, q_ppm: qPpm(sums[c] as number, scored) }));
  let best = 0;
  for (let c = 1; c < scores.length; c += 1) if ((scores[c] as { sum: number }).sum > (scores[best] as { sum: number }).sum) best = c;
  return { kind: "choice", choice: (scores[best] as { choice: string }).choice, scores, worlds_scored: scored };
}


/** The input of a label: the real request's candidates in canonical order with the pair's prior (one entry per candidate). */
export interface LabelInput {
  readonly world: WorldRequest;
  readonly channel: readonly string[];
  readonly candidates: readonly Candidate[];
  readonly prior_ppm: readonly number[];
  readonly stream: Uint8Array;
}

export type LabelDecision =
  | { readonly kind: "refused"; readonly reason: "too_few_candidates" | "belief_refused" | "below_min_valid"; readonly prior_ppm: readonly number[] }
  | {
      readonly kind: "label";
      readonly choice: string;
      /** The searched candidates (indices into `candidates`, ascending) and their weighted mean leaf value. */
      readonly entries: readonly { readonly candidate: number; readonly q_ppm: number }[];
      readonly prior_ppm: readonly number[];
      /** The weighted sums of the searched candidates in prior rank (what the actor ranks by) and the worlds they were scored over. */
      readonly scores: Extract<SearchDecision, { kind: "choice" }>["scores"];
    };

/** Search the set the config's rule names and report the label: a pure function of the input and the reviewed ports. */
export async function searchLabel(input: LabelInput, deps: SearchDecisionDeps): Promise<LabelDecision> {
  if (input.candidates.filter((candidate) => !candidate.pending).length < 2) return { kind: "refused", reason: "too_few_candidates", prior_ppm: input.prior_ppm };
  const indices = searchedIndices(deps.config, input.candidates, input.prior_ppm);
  const decision = await searchDecision({ world: input.world, channel: input.channel, top: indices.map((i) => input.candidates[i] as Candidate), stream: input.stream }, deps);
  if (decision.kind === "fallback") return { kind: "refused", reason: decision.reason, prior_ppm: input.prior_ppm };
  const asc = [...indices.keys()].sort((a, b) => (indices[a] as number) - (indices[b] as number));
  return {
    kind: "label",
    choice: decision.choice,
    entries: asc.map((k) => ({ candidate: indices[k] as number, q_ppm: (decision.scores[k] as { q_ppm: number }).q_ppm })),
    prior_ppm: input.prior_ppm,
    scores: decision.scores,
  };
}
