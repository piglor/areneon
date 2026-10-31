// The best-response responder and the E0-P evaluation (ADR-029 section 2, ticket #1257). Evaluation-only: it measures how much
// a responder that knows a frozen pair's WEIGHTS (white-box) but sees only PUBLIC team information gains by choosing its Team
// Preview against that pair, and how much of that gain needs the knowledge (the blind control).
//
// * The responder (team B) chooses its Preview with NO game outcome. At its Team Preview request it reads what its own channel
//   shows of team A (the six preview species), computes ADR-025's belief over A from the public state, samples `worlds` full
//   teams from it (`sampleTeam`, never A's true sets), starts a hypothetical battle per world (`startHypothetical`) and builds the
//   payoff table of ADR-029 section 3 from the evaluated pair's own networks: rows are the responder's top `k_own` strategies (the
//   evaluated Preview network on the responder's own observation), columns every ordered lead pair of A's strategies. It plays the
//   row with the highest expected entry under the column weights of a model of the evaluated side's play (ties in canonical order):
//     argmax: the number of worlds in which the evaluated Preview network's top order has that lead pair;
//     sample: the evaluated network's distribution summed by lead pair (its mean prior over the worlds);
//     blind:  every lead pair equally (the uniform distribution over A's strategies, which are 30 lead pairs of 6 back sets each).
//   The model is recomputed in the responder's worlds: neither A's true sets nor the realised mixture or draw of the game reach it.
// * Battle play is the evaluated pair's own Battle network in the same mode: only the Preview differs.
// * The room arm is the same construction from team A's side, against the evaluated pair on team B (the mirror opponent).
//
// The module is reachable only from the best-response command (a source scan). It writes no Episode, no training example and
// no stream: the games are plain battle outcomes the command tallies, and the responder's decision is a choice string.

import { interpretChannel } from "@areneon/public-state";
import type { PublicFlag, PublicSide } from "@areneon/public-state";
import type { Decision } from "@areneon/schema/types/v1/best-response-record.ts";
import { type BestResponseOutcome, type BestResponseRecord, BEST_RESPONSE_ARMS, readBestResponseRecord, tallyArm } from "./best-response-record.ts";
import { jcs } from "./canonical-json.ts";
import type { WorldSetLike } from "./damage-aware-assumed-set.ts";
import { mapWorld } from "./world-sampler.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { PlayedGame } from "./evaluation-harness.ts";
import { resolveFormat } from "./formats.ts";
import { HYPOTHETICAL_BATTLE_FORMAT, type HypotheticalPreview, startHypothetical } from "./hypothetical-preview.ts";
import { progressGameStreams, PROGRESS_PAIRING, PROGRESS_SEED_DERIVATION } from "./progress-seed.ts";
import { type PayoffClient, type PayoffConfig, type PayoffTable, PAYOFF_LEAF_FORMAT, type OwnRule, type PayoffLeaf, buildPayoffTable } from "./preview-payoff.ts";
import { type PolicyActorOptions, policyActor, type PolicySeat, policyGame } from "./policy-actor.ts";
import type { PolicyClient } from "./policy-worker.ts";
import { type ProgressPool, progressPlan } from "./progress-evaluation.ts";
import { sha256Hex } from "./rules.ts";
import { validateTeam } from "./teams.ts";
import type { Actor, Player } from "./runner.ts";
import type { DataClass } from "./team-class.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** Bumped when the responder chooses or plays differently for the same input (part of the record's plan). */
export const BEST_RESPONSE_FORMAT = 1;

/** The purpose of the responder's per-decision stream (its own, apart from every other stream). */
export const RESPONDER_PURPOSE = "best-response/1";

/** `k_opp` of the table: take every ordered lead pair (a count no team has), so the blind control can weigh them all equally. */
export const ALL_COLUMNS = 1 << 20;

export class BestResponseError extends Error {
  override name = "BestResponseError";
}

/** The evaluated pair's worker: a policy that asks, evaluates and starts games (`PolicyWorkerClient`; a test passes a stand-in). */
export type BestResponseClient = PolicyClient & PayoffClient & { beginGame(): Promise<void> };

export type PlayMode = "argmax" | "sample";
/** The responder's three uses: against the model of the evaluated side's play (`responder`, team B; `room`, team A) and against the uniform model (`blind`). */
export type ResponderArm = "responder" | "blind" | "room";

/** Declared values, no defaults (the experiment declares them). */
export interface ResponderSettings {
  /** Opponent teams sampled from the belief per decision. */
  worlds: number;
  k_own: number;
  own_rule: OwnRule;
  leaf: PayoffLeaf;
}

/** The structural port of the full-team opponent draw (the composition root joins the opponent model; the bridge does not import it). */
export type ResponderTeamDraw = { readonly kind: "team"; readonly team: { readonly sets: readonly WorldSetLike[] }; readonly redraws: number } | { readonly kind: "refused"; readonly code: string };
export type ResponderTeamSource = { readonly kind: "refused"; readonly code: string } | { readonly kind: "source"; readonly source: { draw(stream: Uint8Array, index: number): ResponderTeamDraw } };
export interface ResponderTeamPort {
  /** ADR-025 section 5: the belief identity hash. */
  readonly identity_hash: string;
  readonly prior_hash: string;
  readonly belief_config_hash: string;
  /** The belief of the reader's opponent half: all the responder knows of the other team. */
  open(input: { readonly opponent: PublicSide; readonly flags: readonly PublicFlag[] }): ResponderTeamSource;
}

/** The responder's 32-byte per-decision stream: SHA-256 of the JCS of `{game, player, policy_id, purpose, request_seq, seed}`. The seed reaches only this function. */
export function responderStream(parts: { readonly seed: string; readonly game: number; readonly player: string; readonly policy_id: string; readonly request_seq: number }): Uint8Array {
  const hex = sha256Hex(jcs({ game: parts.game, player: parts.player, policy_id: parts.policy_id, purpose: RESPONDER_PURPOSE, request_seq: parts.request_seq, seed: parts.seed }));
  return Uint8Array.from(hex.match(/../g) as string[], (byte) => parseInt(byte, 16));
}

/** The integer weight of each column of a table under a model of the evaluated side's play. */
export function columnWeights(opp: PayoffTable["opp"], model: PlayMode | "blind"): number[] {
  return opp.map((column) => {
    if (model === "blind") return 1;
    if (model === "sample") return column.mean_ppm;
    if (column.argmax_worlds === undefined) throw new BestResponseError("the table was built without argmax columns");
    return column.argmax_worlds;
  });
}

export interface Chosen {
  /** Index into the table's own strategies. */
  readonly rank: number;
  /** Floor of the weighted mean of the chosen row. */
  readonly expected_ppm: number;
  /** Columns that carried weight and had an entry in every row. */
  readonly columns_used: number;
}

/**
 * The row with the highest expected entry under integer column weights, ties by the strategy's canonical rank; null when no
 * column carries weight with an entry in every row. A column some row lacks (no world produced it) is left out for every row,
 * so all rows are compared over the same columns.
 */
export function chooseResponse(table: Pick<PayoffTable, "own" | "entries">, weights: readonly number[]): Chosen | null {
  const used: number[] = [];
  weights.forEach((weight, c) => {
    if (weight > 0 && table.entries.every((row) => row[c] !== null && row[c] !== undefined)) used.push(c);
  });
  if (used.length === 0 || table.own.length === 0) return null;
  const totals = table.entries.map((row) => used.reduce((sum, c) => sum + (weights[c] as number) * (row[c] as number), 0));
  let best = 0;
  for (let r = 1; r < totals.length; r += 1) {
    const total = totals[r] as number;
    const top = totals[best] as number;
    if (total > top || (total === top && (table.own[r] as { rank: number }).rank < (table.own[best] as { rank: number }).rank)) best = r;
  }
  const mass = used.reduce((sum, c) => sum + (weights[c] as number), 0);
  return { rank: best, expected_ppm: Math.floor((totals[best] as number) / mass), columns_used: used.length };
}

/** What a Team Preview decision reads of the game: the seat's own view and the harness's labels. */
export interface PreviewContext {
  seat: Player;
  /** The seat's own channel to its Team Preview request. */
  channel: readonly string[];
  /** The seat's own team as brought. */
  team: readonly Readonly<PokemonSet>[];
  /** The battle name the worker client keys its one-query-at-a-time rule by. */
  battle: string;
  seed: string;
  game: number;
  request_seq: number;
  channel_class: DataClass;
}

export interface ResponderDeps {
  engine: Engine;
  vocabulary: TeamVocabulary;
  format_id: string;
  teams: ResponderTeamPort;
  settings: ResponderSettings;
  client: PayoffClient;
  /** The evaluated pair's policy id in the mode. */
  policy_id: string;
  mode: PlayMode;
}

/** A Team Preview decision: what the record keeps, and the `team ...` order each arm plays (none: the evaluated pair's own Preview). */
export interface PreviewResult {
  readonly decision: Decision;
  readonly orders: Readonly<Partial<Record<ResponderArm, string>>>;
}

const ARM_MODEL = (arm: ResponderArm, mode: PlayMode): PlayMode | "blind" => (arm === "blind" ? "blind" : mode);
const REQUEST_PREFIX = "|request|";

/**
 * The responder's decision at one Team Preview request for the given arms. Everything it reads is the seat's own view
 * (`PreviewContext`), the evaluated pair's weights (through the worker's `evaluate`) and the belief's draws.
 */
export async function decidePreview(deps: ResponderDeps, ctx: PreviewContext, arms: readonly ResponderArm[]): Promise<PreviewResult> {
  const { engine, vocabulary, format_id, teams, settings, client, policy_id, mode } = deps;
  const format = resolveFormat(engine, format_id);
  const fallback = (code: string): PreviewResult => ({ decision: { kind: "fallback", code }, orders: {} });
  const run = interpretChannel(ctx.seat, ctx.channel);
  const frame = run.frames.at(-1);
  const request = ctx.channel.findLast((line) => line.startsWith(REQUEST_PREFIX));
  need(frame !== undefined && request !== undefined, "the Team Preview channel holds no line or no request");
  const realRequest = (request as string).slice(REQUEST_PREFIX.length);
  const opened = teams.open({ opponent: (frame as NonNullable<typeof frame>).state.opponent, flags: run.flags });
  if (opened.kind === "refused") return fallback(opened.code);
  const stream = responderStream({ seed: ctx.seed, game: ctx.game, player: ctx.seat, policy_id, request_seq: ctx.request_seq });
  const worlds: { index: number; preview: HypotheticalPreview }[] = [];
  let refused = 0;
  let unstarted = 0;
  for (let index = 0; index < settings.worlds; index += 1) {
    const drawn = opened.source.draw(stream, index);
    if (drawn.kind === "refused") {
      refused += 1;
      continue;
    }
    const world = { slots: drawn.team.sets.map((set, slot) => ({ index: slot, brought: true, set })) };
    try {
      // The sets a battle plays are the validator's normalised copy (a set with no nickname is named by its species, as in the pool's teams
      // a real game plays): the worker's reader checks the engine's request against those names. A team the validator refuses stays refused.
      const sets = mapWorld({ engine, vocabulary }, format_id, world).map((member) => member.set);
      // Called for its in-place normalisation (nicknames), not its verdict: startHypothetical validates the team again and refuses it.
      validateTeam(engine, format, sets);
      worlds.push({ index, preview: startHypothetical({ engine, format_id, seat: ctx.seat, own_team: ctx.team, world_team: sets, stream, world: index, data_class: ctx.channel_class, own_request: realRequest }) });
    } catch (error) {
      // A sampled team the validator refuses as a whole, or a format without Team Preview, is counted; a set the vocabulary cannot map is a defect.
      if (!UNSTARTABLE.has((error as Error).name)) throw error;
      unstarted += 1;
    }
  }
  if (worlds.length === 0) return fallback("no_world");
  const config: PayoffConfig = { k_own: settings.k_own, k_opp: ALL_COLUMNS, own_rule: settings.own_rule, leaf: settings.leaf, argmax_columns: true };
  const table = await buildPayoffTable({ config, worlds, client, battle: ctx.battle, policy_id, own_request: realRequest, stream });
  const choices: { arm: ResponderArm; rank: number; key: string; expected_ppm: number; columns_used: number }[] = [];
  const orders: Partial<Record<ResponderArm, string>> = {};
  for (const arm of arms) {
    const chosen = chooseResponse(table, columnWeights(table.opp, ARM_MODEL(arm, mode)));
    if (chosen === null) return fallback("no_complete_column");
    const strategy = table.own[chosen.rank] as PayoffTable["own"][number];
    choices.push({ arm, rank: chosen.rank, key: `${strategy.leads.join("")}|${strategy.back.join("")}`, expected_ppm: chosen.expected_ppm, columns_used: chosen.columns_used });
    orders[arm] = strategy.order;
  }
  return {
    decision: { kind: "table", counts: table.counts, worlds_unstarted: unstarted, worlds_refused: refused, own: table.own.length, columns: table.opp.length, complete: table.complete, choices },
    orders,
  };
}

/** A guard that counts as one branch: the condition holds or the evaluation stops with the reason. */
function need(ok: boolean, reason: string): void {
  if (!ok) throw new BestResponseError(reason);
}

/** The errors that say one sampled team cannot be started as a hypothetical battle (counted); anything else is a defect of the run. */
const UNSTARTABLE = new Set(["InvalidTeamError", "HypotheticalError"]);

const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");

/** A game's outcome seen from the focal seat; an incident is an error. */
function outcomeOf(game: number, played: PlayedGame, focal: Player): BestResponseOutcome {
  const { outcome } = played;
  if (outcome.kind === "incident") throw new BestResponseError(`game ${game}: ${outcome.incident}: ${outcome.message}`);
  const side = (seat: Player): "focal" | "other" => (seat === focal ? "focal" : "other");
  if (outcome.kind === "forfeit") return { kind: "forfeit", loser: side(outcome.loser), cause: outcome.cause, request_seq: outcome.request_seq, turns: outcome.turns };
  return { kind: "played", winner: outcome.winner === null ? "tie" : side(outcome.winner), turns: outcome.turns };
}

export interface BestResponseOptions {
  engine: Engine;
  config: TrainingRunConfig;
  pool: ProgressPool;
  /** Validation pairs to play: `2 * pairs` games per arm. */
  pairs: number;
  evaluated: {
    generation: number;
    generation_manifest_sha256: string;
    checkpoints: { preview: string; battle: string };
    mode: PlayMode;
    /** The pair's policy id in that mode. */
    policy_id: string;
    client: BestResponseClient;
  };
  responder: {
    settings: ResponderSettings;
    teams: ResponderTeamPort;
    vocabulary: TeamVocabulary;
    /** The Team Preview latency budget of every game: the table is built inside the decision. */
    preview_budget_ms: number;
  };
  training_teams: ReadonlySet<string>;
  league_seeds: ReadonlySet<string>;
  evaluation_key?: string;
  /** Called after each game with its index (progress output only). */
  on_game?: (game: number) => void;
}

/** Play the four arms of every game and return the record. A harness incident, or a failure of the responder, fails the run and nothing is returned. */
export async function playBestResponse(options: BestResponseOptions): Promise<BestResponseRecord> {
  const { engine, config, pool, pairs, evaluated, responder } = options;
  if (!Number.isSafeInteger(pairs) || pairs < 1) throw new BestResponseError("--pairs must be an integer of at least 1");
  if (!Number.isSafeInteger(responder.preview_budget_ms) || responder.preview_budget_ms < 1) throw new BestResponseError("the responder's Preview budget must be a positive whole number of milliseconds");
  const { settings } = responder;
  for (const [name, value] of [["worlds", settings.worlds], ["k_own", settings.k_own]] as const) if (!Number.isSafeInteger(value) || value < 1) throw new BestResponseError(`${name} must be an integer of at least 1`);
  const evaluationKey = options.evaluation_key;
  if (evaluationKey !== undefined && !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(evaluationKey)) throw new BestResponseError("--evaluation-key must be an id token ([A-Za-z0-9][A-Za-z0-9._:-]{0,127})");
  if (evaluationKey === config.seeds.run_key) throw new BestResponseError("--evaluation-key is the run's own key: omit the option to key the plan by it");
  if (evaluated.client.featureSet !== undefined && evaluated.client.featureSet !== null) throw new BestResponseError("features_pair: a pair that reads candidate features is not supported by the responder");
  let plan;
  try {
    plan = progressPlan({ config, pool, games: pairs * 2, training_teams: options.training_teams, league_seeds: options.league_seeds, ...(evaluationKey === undefined ? {} : { evaluation_key: evaluationKey }) });
  } catch (error) {
    throw new BestResponseError((error as Error).message);
  }
  const planKey = evaluationKey ?? config.seeds.run_key;
  const { client, mode, policy_id } = evaluated;
  const deps: ResponderDeps = { engine, vocabulary: responder.vocabulary, format_id: config.format_id, teams: responder.teams, settings, client, policy_id, mode };
  const budget = { preview_ms: responder.preview_budget_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
  const failures: Error[] = [];
  /** One decision per (game, side): the responder and blind arms of team B share one table. */
  const memo = new Map<string, Promise<PreviewResult>>();
  const evaluatedSeat = (): PolicySeat => ({ client, policy_id, sampling: mode });
  const responderSeat = (arm: ResponderArm, side: "a" | "b"): PolicySeat => ({
    ...evaluatedSeat(),
    make: (o: PolicyActorOptions): Actor => {
      const base = policyActor(o);
      const preview_actor: Actor = async (request, context) => {
        need(context?.team !== undefined, "the responder needs its own channel and team");
        const own = context as NonNullable<typeof context>;
        const key = `${gameIndex}|${side}`;
        let found = memo.get(key);
        if (found === undefined) {
          found = decidePreview(deps, { seat: o.player, channel: own.channel, team: own.team as NonNullable<typeof own.team>, battle: o.battle, seed: o.seed, game: o.game, request_seq: request.request_seq, channel_class: o.channel_class }, side === "b" ? ["responder", "blind"] : ["room"]);
          memo.set(key, found);
        }
        let result: PreviewResult;
        try {
          result = await found;
        } catch (error) {
          failures.push(error as Error);
          throw error;
        }
        // No order: the responder had no table to choose from, so the evaluated pair's own Preview is played (the mirror's choice).
        return result.orders[arm] ?? base(request, context);
      };
      return policyActor({ ...o, preview_actor });
    },
  });

  const rows: BestResponseRecord["games"] = [];
  let gameIndex = 0;
  for (let game = 0; game < pairs * 2; game += 1) {
    gameIndex = game;
    const pair = game >> 1;
    const aSeat: Player = game % 2 === 0 ? "p1" : "p2";
    const bSeat = rival(aSeat);
    const [teamA, teamB] = plan.pairs[pair] as (typeof plan.pairs)[number];
    const streams = progressGameStreams(planKey, pair);
    const play = async (seats: Record<Player, PolicySeat>, focal: Player): Promise<BestResponseOutcome> => {
      await client.beginGame();
      const played = await policyGame({
        engine,
        formatId: config.format_id,
        seed: streams.seed,
        game: pair,
        teams: { [aSeat]: teamA.sets, [bSeat]: teamB.sets } as Record<Player, PokemonSet[]>,
        known_synthetic: pool.known_synthetic,
        seats,
        latency_budget: budget,
        maxTurns: config.generation.max_turns,
      });
      if (failures.length > 0) throw new BestResponseError(`game ${game}: the responder failed: ${(failures[0] as Error).message}`);
      return outcomeOf(game, played, focal);
    };
    const seatsOf = (a: PolicySeat, b: PolicySeat): Record<Player, PolicySeat> => ({ [aSeat]: a, [bSeat]: b }) as Record<Player, PolicySeat>;
    const mirror = await play(seatsOf(evaluatedSeat(), evaluatedSeat()), bSeat);
    const answered = await play(seatsOf(evaluatedSeat(), responderSeat("responder", "b")), bSeat);
    const blind = await play(seatsOf(evaluatedSeat(), responderSeat("blind", "b")), bSeat);
    const room = await play(seatsOf(responderSeat("room", "a"), evaluatedSeat()), aSeat);
    const decision = async (side: "a" | "b"): Promise<Decision> => ((await memo.get(`${game}|${side}`)) as PreviewResult).decision;
    rows.push({ game, pair, a_seat: aSeat, mirror, responder: answered, blind, room, b_decision: await decision("b"), a_decision: await decision("a") });
    options.on_game?.(game);
  }
  const ownDataset = config.team_dataset?.dataset_hash;
  const evaluationDataset = ownDataset !== undefined && pool.dataset_hash !== ownDataset ? pool.dataset_hash : undefined;
  const maxTurns = config.generation.max_turns;
  return readBestResponseRecord({
    schema_version: 1,
    data_class: "synthetic",
    run_id: config.seeds.run_key,
    version_manifest: resolveFormat(engine, config.format_id).manifest,
    evaluated: { generation: evaluated.generation, generation_manifest_sha256: evaluated.generation_manifest_sha256, checkpoints: evaluated.checkpoints, mode, policy_id },
    teams: { dataset_hash: ownDataset ?? pool.dataset_hash, ...(evaluationDataset === undefined ? {} : { evaluation_dataset_hash: evaluationDataset }), split: "validation", pairs_used: pairs },
    plan: {
      seed_derivation: PROGRESS_SEED_DERIVATION,
      pairing: PROGRESS_PAIRING,
      games: pairs * 2,
      max_turns: maxTurns,
      latency_budget_ms: { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms },
      responder_preview_budget_ms: responder.preview_budget_ms,
      ...(evaluationKey === undefined ? {} : { evaluation_key: evaluationKey }),
      responder: {
        format: BEST_RESPONSE_FORMAT,
        payoff_leaf_format: PAYOFF_LEAF_FORMAT,
        hypothetical_battle_format: HYPOTHETICAL_BATTLE_FORMAT,
        worlds: settings.worlds,
        k_own: settings.k_own,
        own_rule: settings.own_rule,
        leaf: settings.leaf,
        columns: "all-lead-pairs",
        belief: { identity_hash: responder.teams.identity_hash, prior_hash: responder.teams.prior_hash, belief_config_hash: responder.teams.belief_config_hash },
      },
    },
    result: Object.fromEntries(BEST_RESPONSE_ARMS.map((arm) => [arm, tallyArm(rows.map((row) => row[arm]), maxTurns)])),
    games: rows,
  });
}
