// The Team Preview solver as a policy seat (ADR-029 sections 3, 4, 5 and 7; ticket #1259). Evaluation only: a play-time wrapper
// around a frozen pair (or around ADR-026's search actor), never part of training.
//
// `previewSolveActor` is an A1 `Actor` for ONE seat of ONE game. At the Team Preview request only, it
//   1. reads its own channel and team (the six preview species the opponent shows, its own request), asks ADR-025's belief for
//      the opponent's team and draws `worlds.n` FULL teams from it (`sampleTeam`; never the opponent's real sets),
//   2. starts a hypothetical battle per world (`hypothetical-preview.ts`, branded: no Episode, record or training path takes it)
//      and builds the payoff table U(s, L) of ADR-029 section 3 from the pair's own networks (`preview-payoff.ts`),
//   3. solves the matrix (the integer solver behind the `PreviewSolverPort`) and draws a strategy from our average mixture with
//      `u` of purpose `preview-solve/1`, played as the strategy's more probable order.
// Every other request goes to the wrapped actor. A fallback (a belief refusal, too few worlds, a worker or engine failure, the A2
// clock floor) plays the wrapped actor's own Team Preview choice and is counted by reason; a harness incident stays an incident.
//
// What a solve reads is `PreviewSolveInput` (`hypothetical-preview.ts`): the format, the seat, the own team, the own channel to
// the request and the harness stream. The live battle, the battle record, the omniscient log, hidden verdicts, the battle seed and the
// opponent's team are not parameters of `solvePreview`; the seed reaches only the two harness derivations (the world stream and
// `u`). Nothing solved is stored: the actor returns a choice string. A pair that reads the opponent sheet is refused
// (`features_pair`). A pair that reads candidate features needs the run's `FeatureConfigs` (`features`): every hypothetical
// observation then carries the rows of the pair's own builder, made on that seat's own hypothetical channel and own sets
// (`hypothetical-preview.ts`); a request that cannot be featured drops its world, counted, never a rule fallback.
//
// The bridge does not import the opponent model: the belief's full-team draw and the matrix solver come in as structural ports,
// joined by the composition root in `packages/team-dataset`.

import { createHash } from "node:crypto";
import type { PublicFlag, PublicSide } from "@areneon/public-state";
import { readerSha256 } from "@areneon/public-state";
import { configuredSetHash } from "./feature-build.ts";
import type { FeatureConfigs } from "./candidate-features.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { BELIEF_DEFECTS } from "./damage-aware-worlds.ts";
import type { Engine } from "./engine.ts";
import type { VersionManifest } from "./formats.ts";
import { HYPOTHETICAL_BATTLE_FORMAT, type PreviewSolveInput } from "./hypothetical-preview.ts";
import { type PayoffClient, type PayoffConfig, type PayoffCounts, PAYOFF_LEAF_FORMAT, type PayoffTable, buildPayoffTable, emptyCounts as emptyPayoff } from "./preview-payoff.ts";
import { type PreviewSolveConfig, previewSolveIdentity, previewSolveIdentityHash } from "./preview-solve-config.ts";
import { readPreviewChannel, type ResponderTeamSource, startPreviewWorlds } from "./preview-worlds.ts";
import { type PlaySettings, PolicyProtocolError, policyId } from "./policy-protocol.ts";
import { PolicyReplyError } from "./policy-worker.ts";
import { type Actor, type Player } from "./runner.ts";
import { searchSkipReason } from "./search-actor.ts";
import { sha256Hex } from "./rules.ts";
import type { DataClass } from "./team-class.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The purpose of the draw from the solver's mixture (`PreviewSolveConfig.seed_derivation`). */
export const PREVIEW_SOLVE_PURPOSE = "preview-solve/1";

/** The purpose of the solve's per-decision stream (worlds, engine seeds, back draws): its own, apart from the draw's. */
export const PREVIEW_SOLVE_WORLDS_PURPOSE = "preview-solve-worlds/1";

export type PreviewSolveErrorCode = "setup_mismatch" | "features_pair" | "missing_context" | "belief_defect" | "solver_unbuilt" | "bad_mixture";

export class PreviewSolveError extends Error {
  override name = "PreviewSolveError";
  readonly code: PreviewSolveErrorCode;
  constructor(code: PreviewSolveErrorCode, message: string) {
    super(message);
    this.code = code;
  }
}

export type PreviewSolverKind = "regret-matching" | "regret-matching-plus";

/** The matrix solver (`solve` of the opponent model, joined by the composition root): integer payoffs in ppm, rows ours. */
export interface PreviewSolverPort {
  /** The solver's format string (`SOLVER_FORMAT`): part of the solver identity. */
  readonly format: string;
  solve(matrix: readonly (readonly number[])[], iterations: number, kind: PreviewSolverKind): { readonly row: readonly number[]; readonly col: readonly number[]; readonly exploitability: bigint };
}

/** The structural port of the belief's full-team draw (`OpponentTeamPort` of the composition root). */
export interface PreviewTeamPort {
  /** ADR-025 section 5: the belief identity hash. */
  readonly identity_hash: string;
  readonly prior_hash: string;
  readonly belief_config_hash: string;
  readonly prior_kind: string;
  readonly prior_label: string;
  readonly prior_format_id: string;
  open(input: { readonly opponent: PublicSide; readonly flags: readonly PublicFlag[] }): ResponderTeamSource;
}

/** The reviewed records and ports a solve is built from, shared by every game of a run. */
export interface PreviewSolveSetup {
  engine: Engine;
  format_id: string;
  vocabulary: TeamVocabulary;
  config: PreviewSolveConfig;
  teams: PreviewTeamPort;
  solver: PreviewSolverPort;
  /** The class of the player's channel; the hypothetical battles carry it. */
  data_class: DataClass;
}

/** Join the config and the ports and check they agree: the format, the belief, the prior arm, the solver kind. */
export function previewSolveSetup(input: PreviewSolveSetup): PreviewSolveSetup {
  const { format_id, config, teams } = input;
  if (config.format_id !== format_id) throw new PreviewSolveError("setup_mismatch", "the solver config is for another format");
  if (teams.identity_hash !== config.belief_identity_hash) throw new PreviewSolveError("setup_mismatch", "the belief identity is not the one the solver config names");
  if (teams.prior_kind !== config.worlds.prior.kind || teams.prior_label !== config.worlds.prior.label) throw new PreviewSolveError("setup_mismatch", "the prior arm is not the one the solver config names");
  if (teams.prior_format_id !== format_id) throw new PreviewSolveError("setup_mismatch", "the prior is of another format");
  if (config.solver.kind === "exact-support") throw new PreviewSolveError("solver_unbuilt", "the exact-support solver is declared by the ADR and not built");
  return input;
}

export interface PreviewSolvePolicyResult {
  policy_id: string;
  identity: ReturnType<typeof previewSolveIdentity>;
  identity_hash: string;
}

/** The solver identity (ADR-029 section 4) and the policy id of a pair played with it in `mode`, wrapped around `search` when the wrapped actor searches. */
export function previewSolvePolicy(
  setup: PreviewSolveSetup,
  manifest: VersionManifest,
  pair: { preview: string; battle: string },
  mode: "sample" | "argmax",
  play: PlaySettings = {},
  search?: string,
  features?: string,
): PreviewSolvePolicyResult {
  const inputs = {
    belief_identity_hash: setup.teams.identity_hash,
    reader_sha256: readerSha256(),
    version_manifest: manifest,
    solver_format: setup.solver.format,
    leaf_format: PAYOFF_LEAF_FORMAT,
    hypothetical_battle_format: HYPOTHETICAL_BATTLE_FORMAT,
  };
  const identity_hash = previewSolveIdentityHash(setup.config, inputs);
  return { policy_id: policyId(pair, mode, play, features, search, identity_hash), identity: previewSolveIdentity(setup.config, inputs), identity_hash };
}

const hexBytes = (hex: string): Uint8Array => Uint8Array.from(hex.match(/../g) as string[], (byte) => parseInt(byte, 16));

/** The solve's 32-byte per-decision stream: SHA-256 of the JCS of `{game, player, policy_id, purpose, request_seq, seed}`. The seed reaches only this and `previewSolveU`. */
export function previewSolveStream(parts: { readonly seed: string; readonly game: number; readonly player: string; readonly policy_id: string; readonly request_seq: number }): Uint8Array {
  return hexBytes(sha256Hex(jcs({ game: parts.game, player: parts.player, policy_id: parts.policy_id, purpose: PREVIEW_SOLVE_WORLDS_PURPOSE, request_seq: parts.request_seq, seed: parts.seed })));
}

/** The uniform draw of one solve, an integer in `[0, 2^53 - 1]`: the first 53 bits of SHA-256 over the JCS of `{seed, game, player, policy_id, request_seq, purpose: "preview-solve/1"}`. */
export function previewSolveU(parts: { readonly seed: string; readonly game: number; readonly player: string; readonly policy_id: string; readonly request_seq: number }): number {
  const digest = createHash("sha256").update(jcs({ game: parts.game, player: parts.player, policy_id: parts.policy_id, purpose: PREVIEW_SOLVE_PURPOSE, request_seq: parts.request_seq, seed: parts.seed } as unknown as Json)).digest();
  return Number(digest.readBigUInt64BE(0) >> 11n);
}

/** The first index whose cumulative ppm exceeds `floor(u * 10^6 / 2^53)`: the draw from a mixture that sums to 10^6. */
export function drawFromMixture(mixture: readonly number[], u: number): number {
  const target = Number((BigInt(u) * 1_000_000n) >> 53n);
  let cumulative = 0;
  for (let i = 0; i < mixture.length; i += 1) {
    cumulative += mixture[i] as number;
    if (cumulative > target) return i;
  }
  throw new PreviewSolveError("bad_mixture", "the mixture does not sum to 10^6");
}

export const FALLBACK_REASONS = ["belief_refusal", "too_few_worlds", "worker_failure", "engine_failure", "clock_floor"] as const;
export type FallbackReason = (typeof FALLBACK_REASONS)[number];

export interface SolveDeps {
  engine: Engine;
  vocabulary: TeamVocabulary;
  config: PreviewSolveConfig;
  teams: PreviewTeamPort;
  solver: PreviewSolverPort;
  client: PayoffClient;
  /** The battle id the worker client keys its one-query-at-a-time rule by. */
  battle: string;
  /** The solver policy id: it labels the worker queries. */
  policy_id: string;
  channel_class: DataClass;
  /** The feature configs of a pair that reads candidate features (absent: the pair reads none). */
  features?: { configs: FeatureConfigs };
}

/** What one solve produced, for the actor and for tests and tooling; never stored. */
export interface Solved {
  readonly kind: "solved";
  readonly table: PayoffTable;
  /** Our average mixture over `table.own` and the opponent's over `table.opp`, in ppm. */
  readonly row: readonly number[];
  readonly col: readonly number[];
  /** The matrix exploitability of the output, rounded up to ppm. */
  readonly exploitability_ppm: number;
  readonly support_size: number;
  readonly worlds: WorldCounts;
}

export interface Fallback {
  readonly kind: "fallback";
  readonly reason: FallbackReason;
  /** The belief's code of a refusal. */
  readonly code?: string;
  readonly worlds: WorldCounts;
}

export interface WorldCounts {
  /** Draws asked of the belief. */
  readonly asked: number;
  readonly refused: number;
  readonly unstarted: number;
}

const leafOf = (config: PreviewSolveConfig): PayoffConfig["leaf"] => (config.leaf.kind === "rollout" ? { kind: "rollout", turns: config.leaf.turns } : { kind: "first-request" });

/**
 * Solve one Team Preview decision. It reads only `PreviewSolveInput` (the own view and the harness stream) and returns the table, the
 * mixtures and the counts, or the fallback reason. `sink` collects the payoff builder's counts as they are made, so a solve that
 * throws keeps them. A worker reply that is invalid is thrown (`PolicyReplyError`) and the actor counts it as a fallback; every
 * other failure (a defect of the table, of the hypothetical battle or of the worker's protocol) is thrown and stays an incident.
 */
export async function solvePreview(deps: SolveDeps, input: PreviewSolveInput, sink: PayoffCounts = emptyPayoff()): Promise<Solved | Fallback> {
  const { engine, vocabulary, config, teams, solver, client, battle, policy_id, channel_class } = deps;
  const view = readPreviewChannel(input.seat, input.channel);
  const asked = config.worlds.n;
  const none: WorldCounts = { asked, refused: 0, unstarted: 0 };
  const opened = teams.open({ opponent: view.opponent, flags: view.flags });
  if (opened.kind === "refused") {
    if (BELIEF_DEFECTS.has(opened.code)) throw new PreviewSolveError("belief_defect", `the belief refused the run itself (${opened.code})`);
    return { kind: "fallback", reason: "belief_refusal", code: opened.code, worlds: none };
  }
  const started = startPreviewWorlds({ engine, vocabulary, format_id: input.format_id, source: opened.source, count: asked, stream: input.stream, seat: input.seat, own_team: input.team, channel_class, own_request: view.request, ...(deps.features === undefined ? {} : { features: deps.features }) });
  for (const code of Object.keys(started.refusal_codes)) if (BELIEF_DEFECTS.has(code)) throw new PreviewSolveError("belief_defect", `the belief refused the run itself (${code})`);
  const worlds: WorldCounts = { asked, refused: started.refused, unstarted: started.unstarted };
  if (started.worlds.length < config.worlds.min_valid) {
    // Every draw refused by the belief (a legality that never held, say) is a belief refusal with its code; else the worlds were too few.
    if (started.refused === asked) return { kind: "fallback", reason: "belief_refusal", code: Object.keys(started.refusal_codes).sort()[0] as string, worlds };
    return { kind: "fallback", reason: "too_few_worlds", worlds };
  }
  const payoffConfig: PayoffConfig = { k_own: config.k_own, k_opp: config.k_opp, own_rule: config.selection_rule, leaf: leafOf(config) };
  const table = await buildPayoffTable({ config: payoffConfig, worlds: started.worlds, client, battle, policy_id, own_request: view.request, stream: input.stream, counts: sink });
  const valid = table.counts.worlds_used - table.counts.worlds_dropped;
  // A world the engine refused to play on is dropped and counted by the table; losing worlds that way is the engine's failure, not a lack of draws.
  if (table.matrix === null || valid < config.worlds.min_valid) return { kind: "fallback", reason: table.counts.engine_refusals > 0 ? "engine_failure" : "too_few_worlds", worlds };
  const solution = solver.solve(table.matrix, config.solver.iterations as number, config.solver.kind as PreviewSolverKind);
  const exploitability_ppm = Number((solution.exploitability + 999_999n) / 1_000_000n);
  return { kind: "solved", table, row: solution.row, col: solution.col, exploitability_ppm, support_size: solution.row.filter((ppm) => ppm > 0).length, worlds };
}

export interface PreviewSolveCounts {
  /** Requests the actor answered (all phases). */
  decisions: number;
  /** Team Preview requests. */
  previews: number;
  solved: number;
  fallbacks: Record<FallbackReason, number>;
  /** The belief's codes behind `belief_refusal` fallbacks. */
  belief_refusals: Record<string, number>;
  /** Draws asked, refused (coded) and unstartable, summed over the solves tried. */
  worlds_asked: number;
  worlds_refused: number;
  worlds_unstarted: number;
  /** The payoff builder's counts, summed (entries, value queries, clamps, engine refusals, ...). */
  payoff: PayoffCounts;
  /** Sums over the solves: exploitability in ppm and support size of the mixture; divide by `solved`. */
  exploitability_ppm: number;
  support_size: number;
}

const emptyCounts = (): PreviewSolveCounts => ({
  decisions: 0,
  previews: 0,
  solved: 0,
  fallbacks: { belief_refusal: 0, too_few_worlds: 0, worker_failure: 0, engine_failure: 0, clock_floor: 0 },
  belief_refusals: {},
  worlds_asked: 0,
  worlds_refused: 0,
  worlds_unstarted: 0,
  payoff: emptyPayoff(),
  exploitability_ppm: 0,
  support_size: 0,
});

export interface PreviewSolveReport {
  policy_id: string;
  identity_hash: string;
  prior_label: string;
  counts: PreviewSolveCounts;
}

/** A solved decision as tooling sees it: the table's strategies, the mixture and the choice. Never stored. */
export interface SolvedDecision {
  readonly request_seq: number;
  readonly choice: string;
  readonly solved: Solved;
}

export interface PreviewSolveActorOptions {
  setup: PreviewSolveSetup;
  /** The frozen pair's worker: the prior and the leaf values of the hypothetical observations. */
  client: PayoffClient;
  battle: string;
  player: Player;
  /** The solver policy id and identity hash (`previewSolvePolicy`). */
  policy_id: string;
  identity_hash: string;
  /** The battle seed the runner is given: hashed into the stream and `u`, never passed on. */
  seed: string;
  game: number;
  /** The channel class of the battle (`battleClass`): the hypothetical battles carry it. */
  channel_class: DataClass;
  /** The feature configs of a pair that reads candidate features (`pairFeatureSetup(...).configs`); required exactly when the client states a feature set. */
  features?: { configs: FeatureConfigs };
  /** The wrapped actor: every request that is not Team Preview, and the fallback's Team Preview. It was built with the solver policy id. */
  base: Actor;
  /** A2 only: the bank left in ms (null when unknown). Below `config.a2_clock_floor_ms` the solve is skipped (`clock_floor`). A1 never sets it. */
  bank_ms?: () => number | null;
  /** Called with every solved decision. For tests and tooling; the actor reads nothing back. */
  on_solve?: (decision: SolvedDecision) => void;
}

/**
 * The per-decision budget the config declares (`budget_ms`). Nothing in the solve enforces it: the evaluation wiring must give the seat
 * a `guardActor` Team Preview budget of at least this, or a solve that runs longer is a latency forfeit of the seat.
 */
export const previewSolveBudgetMs = (setup: Pick<PreviewSolveSetup, "config">): number => setup.config.budget_ms;

export type PreviewSolveActor = Actor & { report(): PreviewSolveReport; readonly budget_ms: number };

/**
 * The one failure that makes a solve a counted fallback: the worker's reply to a hypothetical query is invalid. Everything else is a
 * defect or an incident and propagates: a `PayoffError` (another own Team Preview request, a prior that does not match its
 * candidates), a `HypotheticalError`, an `UnsupportedCandidateError` (a candidate shape the worker cannot parse) and any worker or
 * protocol failure. A world the engine refuses to play on is not thrown at all: the table drops and counts it.
 */
const failureReason = (error: unknown): FallbackReason | null => (error instanceof PolicyReplyError ? "worker_failure" : null);

export function previewSolveActor(options: PreviewSolveActorOptions): PreviewSolveActor {
  const { setup, client, battle, player, policy_id, identity_hash, seed, game, channel_class, base, bank_ms } = options;
  const { config } = setup;
  const stated = client.featureSet ?? null;
  if (stated !== null && options.features === undefined) throw new PreviewSolveError("features_pair", "a pair that reads candidate features needs the feature configs to build the rows of its hypothetical requests");
  if (stated === null && options.features !== undefined) throw new PreviewSolveError("features_pair", "feature configs were given for a pair whose worker states no feature set");
  if (stated !== null && options.features !== undefined && configuredSetHash(setup.engine, options.features.configs) !== stated.feature_set_hash) throw new PreviewSolveError("features_pair", "the feature configs build another feature set than the worker stated");
  if (((client as { sheetSource?: unknown }).sheetSource ?? null) !== null) throw new PreviewSolveError("features_pair", "a pair that reads the opponent sheet is not supported by the Team Preview solver");
  const deps: SolveDeps = { engine: setup.engine, vocabulary: setup.vocabulary, config, teams: setup.teams, solver: setup.solver, client, battle, policy_id, channel_class, ...(options.features === undefined ? {} : { features: options.features }) };
  const counts = emptyCounts();

  const actor: Actor = async (request, context) => {
    counts.decisions += 1;
    if (request.phase !== "preview") return base(request, context);
    counts.previews += 1;
    if (context === undefined || context.player !== player) throw new PolicyProtocolError(`the Team Preview solver of ${player} needs its own player's channel`);
    if (context.team === undefined) throw new PreviewSolveError("missing_context", `the Team Preview solver of ${player} needs its own team in the context`);
    const fallback = async (reason: FallbackReason): Promise<string> => {
      counts.fallbacks[reason] += 1;
      return base(request, context);
    };
    const bank = bank_ms?.() ?? null;
    if (bank !== null && searchSkipReason(bank, config.a2_clock_floor_ms) !== null) return fallback("clock_floor");
    const stream = previewSolveStream({ seed, game, player, policy_id, request_seq: request.request_seq });
    let outcome: Solved | Fallback;
    // The work a solve did is counted whether it solved, fell back or failed.
    const sink = emptyPayoff();
    const addWork = (): void => {
      for (const key of Object.keys(counts.payoff) as (keyof PayoffCounts)[]) counts.payoff[key] += sink[key];
    };
    try {
      outcome = await solvePreview(deps, { format_id: setup.format_id, seat: player, team: context.team, channel: context.channel, stream }, sink);
    } catch (error) {
      addWork();
      const reason = failureReason(error);
      if (reason === null) throw error;
      return fallback(reason);
    }
    addWork();
    counts.worlds_asked += outcome.worlds.asked;
    counts.worlds_refused += outcome.worlds.refused;
    counts.worlds_unstarted += outcome.worlds.unstarted;
    if (outcome.kind === "fallback") {
      if (outcome.code !== undefined) counts.belief_refusals[outcome.code] = (counts.belief_refusals[outcome.code] ?? 0) + 1;
      return fallback(outcome.reason);
    }
    counts.solved += 1;
    counts.exploitability_ppm += outcome.exploitability_ppm;
    counts.support_size += outcome.support_size;
    const index = drawFromMixture(outcome.row, previewSolveU({ seed, game, player, policy_id, request_seq: request.request_seq }));
    const choice = (outcome.table.own[index] as PayoffTable["own"][number]).order;
    options.on_solve?.({ request_seq: request.request_seq, choice, solved: outcome });
    return choice;
  };

  return Object.assign(actor, {
    budget_ms: previewSolveBudgetMs(setup),
    report(): PreviewSolveReport {
      return { policy_id, identity_hash, prior_label: setup.teams.prior_label, counts: { ...counts, fallbacks: { ...counts.fallbacks }, belief_refusals: { ...counts.belief_refusals }, payoff: { ...counts.payoff } } };
    },
  });
}
