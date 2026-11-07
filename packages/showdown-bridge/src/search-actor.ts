// The decision-time search actor, `rerank-1` (ADR-026 sections 1, 2, 4, 6 and 7; slice R part 2).
//
// A search policy is a frozen pair, a play mode and a `SearchConfig`. `searchActor` is an A1 `Actor` that wraps the pair: at a move
// request with at least two non-pending candidates it asks the worker for the policy's prior (`evaluate`), takes the top-k
// non-pending candidates, draws N worlds of the opponent from ADR-025's belief (`world-sampler.ts`), and for every world plays
// every candidate against the same M replies (the probe's max-damage reply at weight max_damage_ppm and `uniform_draws` uniform
// draws over the non-pending legal replies, `reply-model.ts`) for ONE turn on a clone of the world with the same engine seed
// (`search-engine/1`). The player's view of the turn, appended to the real channel, is a hypothetical channel that ends at the
// world's next own request; the worker's value head scores it (batch 1, one query at a time). A turn that ends the game scores
// the game; a `wait` request is played on with the opponent's forced switch (the first legal choice in canonical order: the
// `actor-rule` choice, because the only options of a forced switch are switches, which the max-damage reply never plays). The
// candidate with the highest weighted sum wins, ties by prior rank.
//
// Not searched, so played by the pair itself in its underlying play mode (the control's mode) and counted: Team Preview, forced
// switches, a lone candidate, a set of pending candidates only, a decision with fewer than W_min valid worlds, a belief refusal and
// the A2 clock skip (below `a2_clock_floor_ms` of bank). There is no time cut-off: k, N, M and R are config constants and the work
// is fixed, so the same pin, configs, pair, channel, team, request and stream give the same choice (same machine and software).
// An overrun of the 30 s decision guard is the harness's latency forfeit (`guardActor`), a scored loss for the search side.
//
// The leaves ask for the value alone (ADR-026 erratum E2 decision 7). When the worker states `value_query` (its encoder fills no token
// or field column from candidate rows or the opponent sheet), a leaf is a `value` query: no candidates and no rows, so a pair that
// reads `probe-damage/1` rows can be searched. The root `evaluate` keeps its candidates and rows (built here with the options'
// `features`). A pair that reads candidate features but whose worker does not state `value_query` (`probe-damage/2` and `/3`), the
// belief feature set, and a pair that reads the opponent sheet are refused at construction (`SearchError`, code `features_pair`).
//
// What it reads is the player's own view: `ActorContext` (own channel and team), the request, the shared reader's public state,
// the world facts of the own channel and ADR-025's belief of the opponent half. The live battle, `BattleRecord`, omniscient log,
// hidden verdicts and the seed are not parameters of the decision (`SearchInput`, a type test); the seed reaches only the harness
// derivation of the search stream, as the sampler's `u`. Nothing searched is stored: the actor returns a choice string and the
// episode writers see no more than they see for any policy.

import { interpretChannel, ownChannel, readerSha256, WORLD_FACTS_FORMAT, type WorldFactsData, worldFactsAtRequests } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { type Candidate, enumerateCandidates } from "./choices.ts";
import { buildFeatures, isFeatureSetId } from "./feature-build.ts";
import { createFeatureMemo, FEATURE_SET_BELIEF, FeatureBuildError, type FeatureSetId } from "./candidate-features.ts";
import { type DamageAwareConfig, damageAwareConfigHash, loadDamageAwareConfig } from "./damage-aware-config.ts";
import { DAMAGE_VALUE_FORMAT } from "./damage-aware-score.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { UnsupportedCandidateError } from "./evaluation-harness.ts";
import type { VersionManifest } from "./formats.ts";
import { loadMechanicsConfig, type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { checkScenario } from "./mechanics-scenario.ts";
import { observe, playerChannel } from "./player-channel.ts";
import { type ActorFeatures, policyActor } from "./policy-actor.ts";
import { type PlaySettings, PolicyProtocolError, policyId } from "./policy-protocol.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import { createReplyModel, REPLY_MODEL_FORMAT, type ReplyModel } from "./reply-model.ts";
import type { RulesSnapshot } from "./rules.ts";
import { type Actor, ForfeitSignal, type Player } from "./runner.ts";
import { MECHANICS_ADAPTER_FORMAT } from "./mechanics-result.ts";
import { LEAF_FORMAT, loadSearchConfig, type SearchConfig, searchIdentity, searchIdentityHash } from "./search-config.ts";
import { type LabelDecision, type LabelInput, type SearchDecision, type SearchDecisionDeps, type SearchInput, searchDecision, searchLabel } from "./search-decision.ts";
import type { DataClass } from "./team-class.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import { createWorldBuilder, WORLD_BUILDER_FORMAT, type WorldBuilder } from "./world-builder.ts";
import { createWorldSampler, searchStream, type WorldRequest, type WorldSamplePort, type WorldSampler } from "./world-sampler.ts";
import { loadWorldWriters, type WorldWriters, worldWritersHash } from "./world-writers.ts";
import { sheetRegimeOf } from "./team-sheet-binding.ts";

/** The search was set up, or asked to run, in a way it does not support. A defect of the caller, not a refusal of a decision. */
export class SearchError extends Error {
  override name = "SearchError";
  readonly code: "features_pair" | "missing_context" | "setup_mismatch" | "policy_mismatch";
  constructor(code: SearchError["code"], message: string) {
    super(message);
    this.code = code;
  }
}

/** Why a decision was played by the pair itself and not searched (the keys of `SearchCounts.fallbacks`). */
export const FALLBACK_REASONS = ["team_preview", "forced_switch", "too_few_candidates", "clock_floor", "belief_refused", "below_min_valid"] as const;
export type FallbackReason = (typeof FALLBACK_REASONS)[number];

/** The one reason the A2 clock gives for skipping the search (ADR-026 section 6): the remaining bank is below the config's floor. */
export type SearchSkipReason = "clock_floor";

/** `clock_floor` when the bank left is below the floor, else null. The only clock-dependent behaviour of the search, A2 only. */
export const searchSkipReason = (bank_ms: number, floor_ms: number): SearchSkipReason | null => (bank_ms < floor_ms ? "clock_floor" : null);

/** Why a world was dropped after it was built (not a builder refusal): the reply model could not reply in it, or the engine refused a choice of the turn. */
export const DROP_REASONS = ["reply_model", "choice_refused"] as const;
export type DropReason = (typeof DROP_REASONS)[number];

/** Integer counts of one game (ADR-026 section 5). Wall time is printed by the caller, never stored. */
export interface SearchCounts {
  /** Requests the actor answered. */
  decisions: number;
  /** Decisions the search chose. */
  searched_decisions: number;
  /** Decisions played by the pair, by reason. */
  fallbacks: Record<FallbackReason, number>;
  /** Worlds asked for, built (valid), refused (no attempt valid) and sampler attempts after the first (the search's redraws), summed over the decisions that sampled. */
  worlds_asked: number;
  worlds_built: number;
  worlds_refused: number;
  redraws: number;
  /** ADR-025's redraws inside the draws, counted apart from `redraws`. */
  legality_redraws: number;
  /** Worlds refused, by the last refusal's code and by `check:class` (`-` is none), and belief refusals by the model's code. */
  refusals: Record<string, number>;
  refusal_check_class: Record<string, number>;
  belief_refusals: Record<string, number>;
  /** Built worlds dropped before scoring, by reason, and the X-class state the built worlds resampled. */
  worlds_dropped: Record<DropReason, number>;
  resampled: Record<string, number>;
  /** Worlds that took part in a choice. */
  worlds_scored: number;
  /** Turns simulated, wait requests played on, leaves scored by the value head (queries) and answered from the per-decision memo, game-end and HP leaves. */
  turns: number;
  wait_continuations: number;
  leaf_queries: number;
  leaf_memo_hits: number;
  leaf_game_end: number;
  leaf_hp_ko: number;
  /** Replies drawn: the probe's (and how many of them were the all-zero `actor-rule` choice) and the uniform ones. */
  replies_max_damage: number;
  replies_all_zero: number;
  replies_uniform: number;
}

export interface SearchReport {
  policy_id: string;
  /** The search identity hash (`search:` member of the policy id). */
  identity_hash: string;
  /** Prior label of the belief the worlds came from. */
  prior_label: string;
  counts: SearchCounts;
  /** The reply model's own counts (probe queries and refusals) over the game. */
  reply_model: ReturnType<ReplyModel["counts"]>;
}

const zeros = <K extends string>(keys: readonly K[]): Record<K, number> => Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;

const emptyCounts = (): SearchCounts => ({
  decisions: 0,
  searched_decisions: 0,
  fallbacks: zeros(FALLBACK_REASONS),
  worlds_asked: 0,
  worlds_built: 0,
  worlds_refused: 0,
  redraws: 0,
  legality_redraws: 0,
  refusals: {},
  refusal_check_class: {},
  belief_refusals: {},
  worlds_dropped: zeros(DROP_REASONS),
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
});

/** The reviewed records and ports a search is built from, shared by every game of a run. */
export interface SearchSetup {
  engine: Engine;
  format_id: string;
  vocabulary: TeamVocabulary;
  config: SearchConfig;
  /** The reviewed weights `damageValue` uses for the max-damage reply (`config/baselines/damage-aware.v1.json`). */
  damage_config: DamageAwareConfig;
  mechanics: MechanicsConfig;
  writers: WorldWriters;
  /** The opponent model: ADR-025's belief and `sample` by index, joined by the composition root. */
  port: WorldSamplePort;
  /** The reviewed engine data of the world facts. */
  facts_data: WorldFactsData;
  snapshot?: RulesSnapshot;
  /** The class the probe queries carry (they are never stored). */
  data_class: "synthetic" | "private";
}

export interface SearchSetupInput {
  engine: Engine;
  format_id: string;
  vocabulary: TeamVocabulary;
  port: WorldSamplePort;
  facts_data: WorldFactsData;
  data_class: "synthetic" | "private";
  /** Default: `config/search/rerank-1.v1.json`. */
  config?: SearchConfig;
  damage_config?: DamageAwareConfig;
  mechanics?: MechanicsConfig;
  writers?: WorldWriters;
  snapshot?: RulesSnapshot;
}

/** Join the reviewed records (the committed ones by default) and check they agree: the format, the belief, the prior arm, the HP rounding. */
export function searchSetup(input: SearchSetupInput): SearchSetup {
  const { engine, format_id, vocabulary, port } = input;
  const config = input.config ?? loadSearchConfig(format_id);
  if (config.format_id !== format_id) throw new SearchError("setup_mismatch", "the search config is for another format");
  if (port.identity_hash !== config.belief_identity_hash) throw new SearchError("setup_mismatch", "the belief identity is not the one the search config names");
  if (port.prior_kind !== config.worlds.prior.kind || port.prior_label !== config.worlds.prior.label) throw new SearchError("setup_mismatch", "the prior arm is not the one the search config names");
  if (port.prior_format_id !== format_id) throw new SearchError("setup_mismatch", "the prior is of another format");
  return {
    engine,
    format_id,
    vocabulary,
    config,
    damage_config: input.damage_config ?? loadDamageAwareConfig("v1", vocabulary),
    mechanics: input.mechanics ?? loadMechanicsConfig(format_id),
    writers: input.writers ?? loadWorldWriters(engine, format_id),
    port,
    facts_data: input.facts_data,
    ...(input.snapshot === undefined ? {} : { snapshot: input.snapshot }),
    data_class: input.data_class,
  };
}

/** The search identity (ADR-026 section 5) and the policy id of a frozen pair played with it in `mode`. */
export function searchPolicy(setup: SearchSetup, manifest: VersionManifest, pair: { preview: string; battle: string }, mode: "sample" | "argmax", play: PlaySettings = {}, features?: string): { policy_id: string; identity: ReturnType<typeof searchIdentity>; identity_hash: string } {
  const inputs = {
    belief_identity_hash: setup.port.identity_hash,
    mechanics_config_hash: mechanicsConfigHash(setup.mechanics),
    mechanics_adapter_format: MECHANICS_ADAPTER_FORMAT,
    damage_aware_config_hash: damageAwareConfigHash(setup.damage_config),
    damage_value_format: DAMAGE_VALUE_FORMAT,
    world_writers_hash: worldWritersHash(setup.writers),
    world_builder_format: WORLD_BUILDER_FORMAT,
    world_facts_format: WORLD_FACTS_FORMAT,
    reply_model_format: REPLY_MODEL_FORMAT,
    leaf_format: LEAF_FORMAT,
    reader_sha256: readerSha256(),
    version_manifest: manifest,
  };
  const identity_hash = searchIdentityHash(setup.config, inputs);
  return { policy_id: policyId(pair, mode, play, features, identity_hash), identity: searchIdentity(setup.config, inputs), identity_hash };
}

export { engineSeed, replyIndex, type SearchDecision, type SearchInput } from "./search-decision.ts";

export interface SearchActorOptions {
  setup: SearchSetup;
  /** The frozen pair's worker: the fallback's choice, the prior and the leaf values. */
  client: PolicyClient & EvaluationClient;
  /** Names this game to the client, which allows one outstanding query per battle. */
  battle: string;
  player: Player;
  team: PrivateTeamSpec;
  manifest: VersionManifest;
  /** The class of the player's channel: the battle's, from both teams (`battleClass`). */
  channel_class: DataClass;
  /** The battle seed `runBattle` is given: hashed into the search stream and the sampler's `u`, never passed on. */
  seed: string;
  game: number;
  /** The search policy id and identity hash (`searchPolicy`). */
  policy_id: string;
  identity_hash: string;
  /** The underlying play mode: the fallback's mode, which is the control's. */
  sampling: "argmax" | "sample";
  play?: PlaySettings;
  /**
   * A2 only: the bank left in ms when the search would start (null when unknown). Below `config.a2_clock_floor_ms` the search is
   * skipped and the pair plays (reason `clock_floor`). This is the one clock-dependent behaviour and A1 never sets it.
   */
  bank_ms?: () => number | null;
  /**
   * Called with every searched decision's scores, in prior rank (the weighted sums over the scored worlds). For tests and the
   * timing script; the actor reads nothing back, and the scores are never stored.
   */
  on_decision?: (entry: { request_seq: number; choice: string; scores: Extract<SearchDecision, { kind: "choice" }>["scores"] }) => void;
  /**
   * The reviewed records the candidate features of the ROOT query are built from, for a pair that reads `probe-damage/1`
   * (ADR-022): required exactly then. The leaves carry no rows.
   */
  features?: ActorFeatures;
  /**
   * The frozen pair's checkpoint ids: required for a pair that reads candidate features, where `policy_id` must be the id
   * `searchPolicy(..., features)` gives for the worker's stated feature set (else `SearchError` `policy_mismatch`).
   */
  pair?: { preview: string; battle: string };
  /** Test seam: the reply model (default `createReplyModel` over the setup's reviewed records). */
  reply_model?: ReplyModel;
}

export type SearchActor = Actor & { report(): SearchReport; decide(input: SearchInput): Promise<SearchDecision>; label(input: LabelInput): Promise<LabelDecision> };

export function searchActor(options: SearchActorOptions): SearchActor {
  const { setup, client, battle, player, team, manifest, channel_class, seed, game, policy_id, identity_hash, sampling, play, bank_ms } = options;
  // The channel is judged against the team-sheet regime of the format its manifest names (ADR-027 section 5).
  const regime = sheetRegimeOf(manifest.showdown_commit, manifest.format_id);
  const { engine, config } = setup;
  const stated = client.featureSet ?? null;
  // The leaves are value-only exactly when the worker stated `value_query` (and the client can send one).
  const valueOnly = client.valueQuery === true && client.value !== undefined;
  if ((client.sheetSource ?? null) !== null) throw new SearchError("features_pair", "a pair that reads the opponent sheet is not supported by the search");
  if (stated !== null) {
    if (stated.feature_set === FEATURE_SET_BELIEF) throw new SearchError("features_pair", "a pair that reads the belief feature set is not supported by the search");
    if (!valueOnly) throw new SearchError("features_pair", "a pair that reads candidate features needs a worker that answers value queries");
    if (options.features === undefined) throw new SearchError("features_pair", "a pair that reads candidate features needs the feature configs to build the root query's rows");
    if (!isFeatureSetId(stated.feature_set)) throw new SearchError("features_pair", "the worker stated a candidate feature set this side cannot build");
    if (options.pair === undefined) throw new SearchError("features_pair", "a pair that reads candidate features needs its checkpoint ids to check the policy id");
    if (policy_id !== policyId(options.pair, sampling, play, stated.feature_set_hash, identity_hash)) {
      throw new SearchError("policy_mismatch", "the policy id is not the one for the feature set the worker stated: build it with searchPolicy(..., features)");
    }
  }
  const features: ActorFeatures | undefined = options.features === undefined ? undefined : { ...options.features, memo: options.features.memo ?? createFeatureMemo() };
  const builder: WorldBuilder = createWorldBuilder({
    engine,
    format_id: setup.format_id,
    scenario_id: checkScenario(engine, setup.mechanics),
    hp_rounding: config.worlds.hp_rounding,
    writers: setup.writers,
  });
  const sampler: WorldSampler = createWorldSampler({
    engine,
    vocabulary: setup.vocabulary,
    format_id: setup.format_id,
    port: setup.port,
    builder,
    settings: { worlds: config.worlds.n, attempts: config.worlds.redraws, min_valid: config.min_valid_worlds },
  });
  const replyModel = options.reply_model ?? createReplyModel({
    engine,
    vocabulary: setup.vocabulary,
    config: setup.damage_config,
    mechanics: setup.mechanics,
    data_class: setup.data_class,
    ...(setup.snapshot === undefined ? {} : { snapshot: setup.snapshot }),
  });
  // The pair's own play: Team Preview, forced switches and every decision that is not searched.
  const inner = policyActor({ client, battle, player, team, manifest, channel_class, seed, game, policy_id, sampling, ...(play === undefined ? {} : { play }), ...(features === undefined ? {} : { features }), engine });
  const counts = emptyCounts();

  const fallback = (reason: FallbackReason): void => {
    counts.fallbacks[reason] += 1;
  };

  const deps: SearchDecisionDeps = { engine, config, sampler, replyModel, client, battle, player, team, manifest, channel_class, regime, policy_id, valueOnly, counts };
  const decide = (input: SearchInput): Promise<SearchDecision> => searchDecision(input, deps);

  /** The world request at the channel's last request: the player's own view only. The channel ends in the move request being answered (checked by the observation). */
  function worldRequest(context: { channel: readonly string[]; team: readonly Readonly<PokemonSet>[] }): WorldRequest {
    const run = interpretChannel(player, context.channel, { terrains: [...setup.facts_data.terrains] });
    const frame = run.frames.at(-1) as (typeof run.frames)[number];
    const request = run.requests[frame.request - 1] as EngineRequest;
    const facts = worldFactsAtRequests({ channel: ownChannel(player, context.channel), data: setup.facts_data }).find((r) => r.line === frame.index)?.facts ?? null;
    return { request, team: context.team, state: frame.state, flags: run.flags.filter((flag) => flag.index <= frame.index), facts };
  }

  const actor: Actor = async (request, context) => {
    counts.decisions += 1;
    if (context === undefined || context.player !== player) throw new PolicyProtocolError(`the search actor of ${player} needs its own player's channel`);
    if (context.team === undefined) throw new SearchError("missing_context", `the search actor of ${player} needs its own team in the context`);
    const play_ = async (reason: FallbackReason): Promise<string> => {
      fallback(reason);
      return inner(request, context);
    };
    if (request.phase === "preview") return play_("team_preview");
    if (request.phase === "forceSwitch") return play_("forced_switch");
    let candidates: Candidate[];
    try {
      candidates = enumerateCandidates(request).candidates;
    } catch (error) {
      throw new UnsupportedCandidateError((error as Error).message);
    }
    // A lone candidate or a set of pending ones only has nothing to rank (an empty set is the pair's own error, `NoCandidateError`).
    if (candidates.filter((candidate) => !candidate.pending).length < 2) return play_("too_few_candidates");
    const bank = bank_ms?.() ?? null;
    if (bank !== null && searchSkipReason(bank, config.a2_clock_floor_ms) !== null) return play_("clock_floor");

    let observation;
    let channel;
    try {
      channel = playerChannel({ schema_version: 1, data_class: channel_class, player, lines: context.channel }, regime);
      observation = await observe({ channel, team, manifest });
    } catch (error) {
      throw new PolicyProtocolError(`cannot observe the request: ${(error as Error).message}`);
    }
    const { identity } = observation;
    if (identity.request_seq !== request.request_seq || identity.request_hash !== request.request_hash) {
      throw new PolicyProtocolError("the observation is of another request than the one asked");
    }
    const world = worldRequest({ channel: context.channel, team: context.team });

    // The root query is the pair's normal one: candidates, and the rows of a pair that reads candidate features.
    let candidate_features;
    if (features !== undefined && stated !== null) {
      try {
        candidate_features = buildFeatures(engine, features.configs, channel, team.team, candidates, features.memo as NonNullable<ActorFeatures["memo"]>, stated.feature_set as FeatureSetId);
      } catch (error) {
        if (error instanceof FeatureBuildError) throw new ForfeitSignal("crash", `no candidate features: ${error.code}`);
        throw error;
      }
      const live = client.featureSet ?? null; // as the pair's own actor does, against what the worker states now
      if (live === null || candidate_features.feature_set_hash !== live.feature_set_hash || candidate_features.feature_set !== live.feature_set) {
        throw new PolicyProtocolError("the candidate features this side builds are of another feature set than the worker stated");
      }
      // Not recorded (`features.record`): search games are not training data, so no play-time sidecar is wanted for the root's rows.
    }
    const prior = await client.evaluate(battle, {
      schema_version: 1,
      identity,
      observation,
      candidates: candidates.map(({ choice, pending }) => ({ choice, pending })),
      ...(candidate_features === undefined ? {} : { candidate_features }),
      sampling: { kind: "evaluate" },
      policy_id,
    } as Parameters<EvaluationClient["evaluate"]>[1]);
    const stream = searchStream({ seed, game, player, policy_id, request_seq: identity.request_seq });
    // The searched set by the config's rule (the prior's top-k, equal priors in canonical order, or slot-cover), searched by the shared decision.
    const decision = await searchLabel({ world, channel: context.channel, candidates, prior_ppm: prior.prior_ppm, stream }, deps);
    if (decision.kind === "refused") {
      return play_(decision.reason);
    }
    counts.searched_decisions += 1;
    options.on_decision?.({ request_seq: request.request_seq, choice: decision.choice, scores: decision.scores });
    return decision.choice;
  };

  return Object.assign(actor, {
    decide,
    label: (input: LabelInput): Promise<LabelDecision> => searchLabel(input, deps),
    report(): SearchReport {
      const worldCounts = sampler.counts();
      const built = builder.counts();
      const merged: SearchCounts = {
        ...counts,
        fallbacks: { ...counts.fallbacks },
        worlds_dropped: { ...counts.worlds_dropped },
        worlds_asked: worldCounts.worlds_asked,
        worlds_built: worldCounts.worlds_valid,
        worlds_refused: worldCounts.worlds_refused,
        redraws: worldCounts.redraws,
        legality_redraws: worldCounts.legality_redraws,
        refusals: { ...worldCounts.refusals },
        refusal_check_class: { ...worldCounts.refusal_check_class },
        belief_refusals: { ...worldCounts.belief_refusals },
        resampled: { ...built.resampled },
      };
      return { policy_id, identity_hash, prior_label: setup.port.prior_label, counts: merged, reply_model: replyModel.counts() };
    },
  });
}
