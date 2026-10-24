// The progress evaluation's play (ADR-017 section 2): the frozen pair of one
// generation plays scripted baselines on the validation teams of the run's
// team dataset, in sample and in argmax mode, on A1 with one query per forward
// pass (ADR-010 section 2: evaluation never batches). Every match uses the same
// pairs and seeds (`progress-seed/1`, `progress-pairing/1`), and plays every
// pair TWICE (ADR-017 erratum E2): game g is pair g >> 1, the candidate sits on
// p1 for an even g and on p2 for an odd g, the teams stay on their seats
// (p1 holds the pair's first team) and both games share the pair's battle seed.
//
// A progress evaluation is a progress view, never the Training gate: it
// writes no Episode, and its game records are an evaluation artifact (they
// hold both sides' hidden information) that the training loader refuses.
// A harness incident fails the whole evaluation and commits nothing; a policy
// forfeit is a scored loss for the failing side, kept with its cause.

import { closeSync, fsyncSync, mkdirSync, openSync, writeSync } from "node:fs";
import { join } from "node:path";
import { readerSha256 } from "@areneon/public-state";
import { enumerateCandidates } from "./choices.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { type DamageAwareActor, damageAwareActor } from "./damage-aware-actor.ts";
import { damageAwareIdentity, damageAwareIdentityHash, damageAwarePolicyId } from "./damage-aware-identity.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { guardActor, type LatencyBudget, type PlayedGame } from "./evaluation-harness.ts";
import { resolveFormat } from "./formats.ts";
import { firstLegal } from "./golden.ts";
import type { DamageAwareSetup } from "./league-generation.ts";
import { type PolicyClient } from "./policy-worker.ts";
import { policyGame, type PolicySeat } from "./policy-actor.ts";
import { PROGRESS_PAIRING, PROGRESS_SEED_DERIVATION, progressGameStreams, progressPairs } from "./progress-seed.ts";
import { type BaselineCounts, omniscientLogHash, progressRecordHash, sumBaselineCounts, tallyRows, type ProgressEvaluationRecordV2, type ProgressGameRecord, readProgressEvaluationRecordV2, readProgressGameRecord } from "./progress-records.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, Player } from "./runner.ts";
import type { SyntheticTeamHashes } from "./team-class.ts";
import { teamHash } from "./team-class.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

/** The teams a progress evaluation may play: the validation split, proven synthetic by where it was read from. */
export interface ProgressTeam {
  /** The team's canonical hash (order of the sets does not matter): its identity in the pairing order. */
  canonical_team_hash: string;
  /** The set array the engine's validator normalised the paste to: the team a battle starts with. */
  sets: PokemonSet[];
}

export interface ProgressPool {
  /** The dataset's hash, as the run config names it. */
  dataset_hash: string;
  teams: readonly ProgressTeam[];
  /** The hash (`teamHash`) of every team of the pool: the synthetic proof. */
  known_synthetic: SyntheticTeamHashes;
}

/** The scripted opponents, named by what the ADR names them. */
export const PROGRESS_BASELINES = [
  { actor: "actor-rule", policy_id: "actor-rule@v1" },
  { actor: "random-legal", policy_id: "random-legal@v1" },
] as const;

/** The damage-aware baseline's policy id (ADR-018 section 5; the opt-in third baseline of the progress evaluation). */
export const DAMAGE_AWARE_POLICY_ID = damageAwarePolicyId("v1");

/** What the damage-aware baseline seat reports of one game: its identity hash and counts. */
export interface BaselineReport {
  identity_hash: string;
  counts: BaselineCounts;
}

export const PROGRESS_MODES = ["argmax", "sample"] as const;
export type ProgressMode = (typeof PROGRESS_MODES)[number];

/** A progress evaluation that cannot be made: a refusal, or a harness incident. */
export class ProgressError extends Error {
  override name = "ProgressError";
}

export interface ProgressOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** SHA-256 of `generations/g<N>/manifest.json`. */
  generation_manifest_sha256: string;
  /** The frozen pair's checkpoint ids. */
  checkpoints: { preview: string; battle: string };
  /** The pair's policy id in each mode (the Python side derives them). */
  policy_ids: Record<ProgressMode, string>;
  /** A started worker client with the pair loaded; one query at a time. */
  client: PolicyClient & { beginGame?: () => Promise<void> };
  /**
   * The previous generation's frozen pair (G >= 1; absent at G = 0), behind a worker of its own
   * (a worker session holds one pair). Its matches are played sample against sample and argmax
   * against argmax, on the same plan as the baselines'.
   */
  previous?: {
    checkpoints: { preview: string; battle: string };
    policy_ids: Record<ProgressMode, string>;
    client: PolicyClient & { beginGame?: () => Promise<void> };
  };
  /**
   * Fixed earlier generations' frozen pairs (anchors), each behind a worker of its own and earlier than
   * `generation`. Each is played in sample and argmax against the candidate's same mode, every validation
   * pair from both seats (see `anchor_games`), as every match does.
   */
  anchors?: readonly {
    generation: number;
    checkpoints: { preview: string; battle: string };
    policy_ids: Record<ProgressMode, string>;
    client: PolicyClient & { beginGame?: () => Promise<void> };
  }[];
  /**
   * Games in every anchor match: even, game 2k and 2k+1 are pair k with the candidate on p1 then on p2.
   * Required with `anchors`; refused above twice the number of disjoint pairs.
   */
  anchor_games?: number;
  /**
   * Another run's frozen pair (a cross-run comparison), behind a worker of its own that loads that run's
   * checkpoints. Played sample against sample and argmax against argmax as `run-<mode>`, every validation pair
   * from both seats (see `run_games`). Its run key must differ from this run's.
   */
  versus?: {
    run_id: string;
    run_config_hash: string;
    generation: number;
    checkpoints: { preview: string; battle: string };
    policy_ids: Record<ProgressMode, string>;
    client: PolicyClient & { beginGame?: () => Promise<void> };
  };
  /** Games in every cross-run match: even, laid out as `games`. Required with `versus`. */
  run_games?: number;
  /**
   * The Team Preview ablation (ticket #940): instead of the usual matches, the generation's Battle network plays
   * `random-legal` and `actor-rule` (argmax) with (i) its own learned Preview, (ii) a uniformly random Preview and
   * (iii) an earlier generation's Preview (behind a worker of its own that holds that Preview and this Battle). Same
   * pairs, seeds and both seats as every other match. Excludes `previous` and `anchors`.
   */
  ablation?: {
    generation: { generation: number; checkpoint: string; policy_id: string; client: PolicyClient & { beginGame?: () => Promise<void> } };
    /** The policy id of the learned Battle with a random Preview. */
    random_policy_id: string;
  };
  /**
   * The damage-aware baseline as a third scripted opponent (ADR-018 slice 4; opt-in): the reviewed records it
   * reads and the wall-clock budget of one of its matches. A match whose elapsed time passes the budget fails
   * the whole evaluation (nothing is committed). Its matches are `damage-aware-argmax` and `damage-aware-sample`.
   */
  damage_aware?: {
    setup: DamageAwareSetup;
    /** The most wall-clock milliseconds one damage-aware match may take. */
    match_budget_ms: number;
    /** The clock the budget is measured with (default `performance.now`); a test seam. */
    clock?: () => number;
  };
  pool: ProgressPool;
  /** Games per baseline and previous match: even, two per pair; refused above twice the number of disjoint pairs. */
  games: number;
  /** The first this-many games of every match (at most its games) are stored as ProgressGameRecords. */
  record_games: number;
  /** The run's own training team hashes (`team-set.json`): a pool team in it is refused. */
  training_teams: ReadonlySet<string>;
  /** Every battle seed of the run's committed `plan.json`s: a progress seed in it is refused. */
  league_seeds: ReadonlySet<string>;
}

export interface ProgressResult {
  record: ProgressEvaluationRecordV2;
  /** The stored games, by file name (`<match_id>-<k>.json`). */
  game_records: Map<string, ProgressGameRecord>;
  /** Wall-clock milliseconds per match: a measurement for the operator, never stored in a record. */
  wall_ms: Record<string, number>;
  /** Wall-clock milliseconds per game of each match (mean and slowest): a measurement for the operator, never stored. */
  game_ms: Record<string, { mean: number; max: number }>;
}

const seatOf = (index: number): Player => (index % 2 === 0 ? "p1" : "p2");
const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");
/** Who a seat (or no seat: a tie) is, seen from the candidate. */
const whose = (candidate: Player): Record<string, "candidate" | "opponent" | "tie"> => ({ [candidate]: "candidate", [rival(candidate)]: "opponent", null: "tie" });

type Match = ProgressEvaluationRecordV2["matches"][number];
type Row = ProgressEvaluationRecordV2["games"][number];

/** One game's row and (for a played or forfeited game) the data of its stored form. */
export function rowOf(match_id: string, game: number, pair: number, candidate: Player, played: PlayedGame): Row {
  const { outcome } = played;
  if (outcome.kind === "incident") throw new ProgressError(`game ${game} of ${match_id}: ${outcome.incident}: ${outcome.message}`);
  const common = { match_id, game, pair, candidate_seat: candidate };
  if (outcome.kind === "forfeit") {
    const { cause, request_seq, turns } = outcome;
    return { ...common, outcome: { kind: "forfeit", loser: whose(candidate)[outcome.loser] as "candidate" | "opponent", cause, request_seq, turns } };
  }
  return { ...common, outcome: { kind: "played", winner: whose(candidate)[String(outcome.winner)] as "candidate" | "opponent" | "tie", turns: outcome.turns } };
}

/**
 * The damage-aware baseline as a scripted seat (ADR-018): the identity hash its records give (computed from the
 * records and the pinned engine, not from a game) and a maker of a fresh guarded actor per game, whose `report`
 * checks the policy id and the identity before it hands back the game's counts. The progress evaluation and the
 * private test play it the same way and differ only in the class they stamp on it.
 */
export function damageAwareBaseline(args: { engine: Engine; setup: DamageAwareSetup; manifest: ReturnType<typeof resolveFormat>["manifest"]; data_class: "synthetic" | "private"; budget: LatencyBudget }): {
  identity: string;
  seat: (baselineSeed: number) => { seat: Actor; report: () => BaselineReport };
} {
  const { engine, setup, manifest, data_class, budget } = args;
  const identity = damageAwareIdentityHash(damageAwareIdentity({ config: setup.config, mechanics: setup.mechanics, manifest, reader_sha256: readerSha256() }));
  return {
    identity,
    seat: (seed) => {
      const actor: DamageAwareActor = damageAwareActor({
        ...setup,
        engine,
        data_class,
        commit: DAMAGE_AWARE_POLICY_ID.slice(DAMAGE_AWARE_POLICY_ID.indexOf("@") + 1),
        seed,
      });
      return {
        seat: guardActor(actor, { role: "scripted", latency_budget: budget }),
        report: () => {
          const { policy_id, identity_hash, counts } = actor.report();
          if (policy_id !== DAMAGE_AWARE_POLICY_ID) throw new ProgressError(`the baseline plays as ${policy_id}, not ${DAMAGE_AWARE_POLICY_ID}`);
          if (identity_hash !== identity) throw new ProgressError("a damage-aware game carries another baseline identity hash than its records give");
          // probe_refusals is written only above 0, so a baseline without refusals keeps the record's bytes.
          const { probe_refusals, ...rest } = counts;
          return { identity_hash, counts: probe_refusals > 0 ? { ...rest, probe_refusals } : rest };
        },
      };
    },
  };
}

/** What a stored game is built from. */
export interface GameFacts {
  run_id: string;
  generation: number;
  match_id: string;
  game: number;
  /** The validation pair the game used (both games of a pair share it): `game >> 1`; absent for a game that is not part of a both-seats evaluation. */
  pair?: number;
  /** The set arrays the two seats were brought with. */
  teams: Record<Player, PokemonSet[]>;
  policies: Record<Player, string>;
  played: PlayedGame;
  /** Only a game with a damage-aware seat: that seat, the baseline identity hash and its counts. */
  baseline?: { seat: Player; identity_hash: string; counts: BaselineCounts };
}

/** Build the ProgressGameRecord of a played or forfeited game, and read it back so a bad one cannot leave. */
export function buildProgressGameRecord(facts: GameFacts): ProgressGameRecord {
  const { played } = facts;
  const { record, outcome } = played;
  if (record === null || outcome.kind === "incident") throw new ProgressError("an incident game cannot be stored");
  // The runner records a choice when its actor answers. With two learned seats behind two workers the answers can
  // arrive in either order, so the stored order is by request_seq, then seat: the same bytes on every rerun.
  const ordered = [...record.choices].sort((a, b) => a.request_seq - b.request_seq || (a.player < b.player ? -1 : a.player > b.player ? 1 : 0));
  const decisions: ProgressGameRecord["decisions"] = ordered.map((entry) => {
    const request = record.requests[entry.player][entry.request_seq - 1];
    if (request === undefined || request.request_seq !== entry.request_seq) throw new ProgressError("a choice names a request the record does not hold");
    return {
      player: entry.player,
      request_seq: entry.request_seq,
      candidates: enumerateCandidates(request).candidates.map(({ choice, pending }) => ({ choice, pending })),
      chosen: entry.choice,
      ...(entry.refused === undefined ? {} : { refused: entry.refused }),
    };
  });
  const seen: ProgressGameRecord["outcome"] =
    outcome.kind === "forfeit"
      ? { kind: "forfeit", loser: outcome.loser, cause: outcome.cause, request_seq: outcome.request_seq, turns: outcome.turns }
      : { kind: "played", winner: ({ p1: "p1", p2: "p2", null: "tie" } as const)[String(outcome.winner) as "p1" | "p2" | "null"], turns: outcome.turns };
  const lines = [...record.omniscient_log];
  const team = (seat: Player) => ({ team: facts.teams[seat] as unknown as ProgressGameRecord["teams"]["p1"]["team"], team_hash: record.team_hashes[seat] });
  return readProgressGameRecord({
    schema_version: 1,
    data_class: "synthetic",
    run_id: facts.run_id,
    generation: facts.generation,
    match_id: facts.match_id,
    game: facts.game,
    ...(facts.pair === undefined ? {} : { pair: facts.pair }),
    seed: record.seed,
    version_manifest: record.manifest,
    teams: { p1: team("p1"), p2: team("p2") },
    policies: facts.policies,
    decisions,
    outcome: seen,
    omniscient_log: lines,
    omniscient_log_sha256: omniscientLogHash(lines),
    ...(facts.baseline === undefined ? {} : { baseline: facts.baseline }),
  });
}

/**
 * The teams of each pair, from the pool, refusing a pool team that the run trained on
 * and more games than there are disjoint pairs.
 */
export function progressPlan(options: Pick<ProgressOptions, "config" | "pool" | "games" | "training_teams" | "league_seeds"> & { anchor_games?: number; run_games?: number }): { pairs: [ProgressTeam, ProgressTeam][]; seeds: string[] } {
  const { config, pool, games, training_teams, league_seeds } = options;
  const anchorGames = options.anchor_games ?? 0;
  if (!Number.isSafeInteger(games) || games < 2 || games % 2 !== 0) throw new ProgressError("--games must be an even integer of at least 2: every pair is played from both seats");
  if (!Number.isSafeInteger(anchorGames) || anchorGames < 0 || anchorGames % 2 !== 0) throw new ProgressError("--anchor-games must be an even integer");
  const runGames = options.run_games ?? 0;
  if (!Number.isSafeInteger(runGames) || runGames < 0 || runGames % 2 !== 0) throw new ProgressError("--run-games must be an even integer");
  const needed = Math.max(games, anchorGames, runGames) / 2;
  const byCanonical = new Map(pool.teams.map((team) => [team.canonical_team_hash, team]));
  if (byCanonical.size !== pool.teams.length) throw new ProgressError("a team appears twice in the validation pool");
  for (const team of pool.teams) {
    if (training_teams.has(teamHash(team.sets))) throw new ProgressError("a validation team is in the run's team-set.json: the run trained on it");
    if (!pool.known_synthetic.has(teamHash(team.sets))) throw new ProgressError("a validation team is not proven synthetic");
  }
  const pairs = progressPairs(config.seeds.run_key, [...byCanonical.keys()]);
  if (needed > pairs.length) throw new ProgressError(`${needed * 2} games need ${needed} disjoint team pairs: the validation split gives ${pairs.length}`);
  const seeds: string[] = [];
  for (let pair = 0; pair < needed; pair++) {
    const { seed } = progressGameStreams(config.seeds.run_key, pair);
    if (league_seeds.has(seed)) throw new ProgressError(`the progress seed of pair ${pair} equals a seed of a committed plan`);
    seeds.push(seed);
  }
  return { pairs: pairs.slice(0, needed).map(([a, b]) => [byCanonical.get(a) as ProgressTeam, byCanonical.get(b) as ProgressTeam]), seeds };
}

/**
 * Play every match of the generation's progress evaluation and return its record
 * and the stored games. Refused before any game: a pool team the run trained on,
 * a seed equal to a committed plan's, more games than disjoint pairs. A harness
 * incident in any game throws `ProgressError` and nothing is returned.
 */
export async function playProgress(options: ProgressOptions): Promise<ProgressResult> {
  const { engine, config, generation, client, pool, games, record_games } = options;
  if (!Number.isSafeInteger(record_games) || record_games < 0 || record_games > games) throw new ProgressError("--record-games must be between 0 and --games");
  const anchors = options.anchors ?? [];
  const anchorGames = options.anchor_games ?? 0;
  if (anchors.length > 0 && anchorGames < 2) throw new ProgressError("--anchor-games must be at least 2 when anchors are given");
  if (anchors.length === 0 && anchorGames !== 0) throw new ProgressError("--anchor-games needs at least one anchor");
  if (new Set(anchors.map((a) => a.generation)).size !== anchors.length) throw new ProgressError("an anchor generation may be given once");
  if (anchors.some((a) => !Number.isSafeInteger(a.generation) || a.generation < 0 || a.generation >= generation)) throw new ProgressError("an anchor must be an earlier generation than the one evaluated");
  const { ablation, versus } = options;
  const runGames = options.run_games ?? 0;
  if (versus !== undefined && runGames < 2) throw new ProgressError("--run-games must be at least 2 when another run's pair is given");
  if (versus === undefined && runGames !== 0) throw new ProgressError("--run-games needs another run's pair");
  if (versus !== undefined && versus.run_id === config.seeds.run_key) throw new ProgressError("the other run has the same run key as the evaluated one");
  if (ablation !== undefined) {
    if (anchors.length > 0 || options.previous !== undefined || versus !== undefined) throw new ProgressError("the Preview ablation is played alone: no anchors, no previous-generation matches and no other run");
    if (!Number.isSafeInteger(ablation.generation.generation) || ablation.generation.generation < 0 || ablation.generation.generation >= generation) throw new ProgressError("the ablation's Preview must come from an earlier generation");
  }
  const plan = progressPlan(options);
  const budget: LatencyBudget = { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
  const manifest = resolveFormat(engine, config.format_id).manifest;
  const matches: ProgressEvaluationRecordV2["matches"] = [];
  const rows: Row[] = [];
  const rowsByMatch = new Map<string, Row[]>();
  const stored = new Map<string, ProgressGameRecord>();
  const wall_ms: Record<string, number> = {};
  const game_ms: Record<string, { mean: number; max: number }> = {};
  const clock = options.damage_aware?.clock ?? (() => performance.now());
  type Seat = { seat: PolicySeat | Actor; report?: () => BaselineReport };
  type Opponent = { id: string; anchored?: true; cross?: true; budget_ms?: number; opponent: (mode: ProgressMode) => Match["opponent"]; seat: (mode: ProgressMode, game: number) => Seat; policies: (mode: ProgressMode) => string; begin: () => Promise<void> };
  const opponents: Opponent[] = PROGRESS_BASELINES.map((baseline) => ({
    id: baseline.actor,
    opponent: () => ({ kind: "baseline", policy_id: baseline.policy_id }),
    seat: (_mode, game) => ({ seat: guardActor(baseline.actor === "actor-rule" ? firstLegal : randomLegalActor(progressGameStreams(config.seeds.run_key, game).baseline_seed), { role: "scripted", latency_budget: budget }) }),
    policies: () => baseline.policy_id,
    begin: () => Promise.resolve(),
  }));
  const damage = options.damage_aware;
  if (damage !== undefined) {
    if (ablation !== undefined) throw new ProgressError("the Preview ablation plays random-legal and actor-rule only: no damage-aware matches");
    if (!Number.isSafeInteger(damage.match_budget_ms) || damage.match_budget_ms < 1) throw new ProgressError("the damage-aware match budget must be a positive whole number of milliseconds");
    if (damage.setup.config.epsilon_ppm !== 0) throw new ProgressError("the damage-aware baseline plays an evaluation with epsilon_ppm 0: it draws nothing");
    if (damage.setup.config.format_id !== config.format_id) throw new ProgressError("the damage-aware baseline's records are for another format");
    const baseline = damageAwareBaseline({ engine, setup: damage.setup, manifest, data_class: "synthetic", budget });
    opponents.push({
      id: "damage-aware",
      budget_ms: damage.match_budget_ms,
      opponent: () => ({ kind: "baseline", policy_id: DAMAGE_AWARE_POLICY_ID }),
      // A fresh actor per game: it keeps the reader's state, the game's memo and its counts.
      seat: (_mode, game) => baseline.seat(progressGameStreams(config.seeds.run_key, game).baseline_seed),
      policies: () => DAMAGE_AWARE_POLICY_ID,
      begin: () => Promise.resolve(),
    });
  }
  const { previous } = options;
  // The candidates: the generation's own pair, or the three Previews of the ablation (argmax only, against the baselines).
  type Candidate = { prefix: string; client: PolicyClient & { beginGame?: () => Promise<void> }; policy_ids: Record<ProgressMode, string>; random_preview?: true; candidate_preview?: Match["candidate_preview"] };
  const candidates: Candidate[] = [{ prefix: "", client, policy_ids: options.policy_ids }];
  let modes: readonly ProgressMode[] = PROGRESS_MODES;
  if (ablation !== undefined) {
    modes = ["argmax"];
    candidates.length = 0;
    candidates.push(
      { prefix: "preview-learned-", client, policy_ids: options.policy_ids, candidate_preview: { kind: "learned" } },
      { prefix: "preview-random-", client, policy_ids: { ...options.policy_ids, argmax: ablation.random_policy_id }, random_preview: true, candidate_preview: { kind: "random" } },
      {
        prefix: `preview-g${ablation.generation.generation}-`,
        client: ablation.generation.client,
        policy_ids: { ...options.policy_ids, argmax: ablation.generation.policy_id },
        candidate_preview: { kind: "generation", generation: ablation.generation.generation, checkpoint: ablation.generation.checkpoint },
      },
    );
  }
  if (previous !== undefined) {
    opponents.push({
      id: "previous",
      opponent: (mode) => ({ kind: "previous", policy_id: previous.policy_ids[mode], checkpoints: previous.checkpoints }),
      seat: (mode) => ({ seat: { client: previous.client, policy_id: previous.policy_ids[mode], sampling: mode } }),
      policies: (mode) => previous.policy_ids[mode],
      begin: async () => {
        await previous.client.beginGame?.();
      },
    });
  }
  for (const anchor of anchors) {
    opponents.push({
      id: `anchor-g${anchor.generation}`,
      anchored: true,
      opponent: (mode) => ({ kind: "anchor", generation: anchor.generation, policy_id: anchor.policy_ids[mode], checkpoints: anchor.checkpoints }),
      seat: (mode) => ({ seat: { client: anchor.client, policy_id: anchor.policy_ids[mode], sampling: mode } }),
      policies: (mode) => anchor.policy_ids[mode],
      begin: async () => {
        await anchor.client.beginGame?.();
      },
    });
  }
  if (versus !== undefined) {
    opponents.push({
      id: "run",
      cross: true,
      opponent: (mode) => ({ kind: "run", run_id: versus.run_id, run_config_hash: versus.run_config_hash, generation: versus.generation, policy_id: versus.policy_ids[mode], checkpoints: versus.checkpoints }),
      seat: (mode) => ({ seat: { client: versus.client, policy_id: versus.policy_ids[mode], sampling: mode } }),
      policies: (mode) => versus.policy_ids[mode],
      begin: async () => {
        await versus.client.beginGame?.();
      },
    });
  }
  for (const candidateKind of candidates) {
    for (const baseline of opponents) {
    // Every match plays each pair from both seats (E2); an anchor match may be longer than the others.
    const count = baseline.cross === true ? runGames : baseline.anchored === true ? anchorGames : games;
    const recorded = Math.min(record_games, count);
    for (const mode of modes) {
      const match_id = `${candidateKind.prefix}${baseline.id}-${mode}`;
      const policy_id = candidateKind.policy_ids[mode];
      const opponent_policy = baseline.policies(mode);
      const matchRows: Row[] = [];
      const hashes: string[] = [];
      const reports: BaselineReport[] = [];
      const gameMs: number[] = [];
      const started = clock();
      for (let game = 0; game < count; game++) {
        const gameStarted = clock();
        const candidate = seatOf(game);
        const pair = game >> 1;
        const [first, second] = plan.pairs[pair] as [ProgressTeam, ProgressTeam];
        // Teams stay on their seats; the policies swap. Both games of a pair share the pair's battle seed.
        const teams: Record<Player, PokemonSet[]> = { p1: first.sets, p2: second.sets };
        const streams = progressGameStreams(config.seeds.run_key, pair);
        const seat: PolicySeat = { client: candidateKind.client, policy_id, sampling: mode, ...(candidateKind.random_preview === undefined ? {} : { random_preview: true }) };
        await candidateKind.client.beginGame?.();
        await baseline.begin();
        const opposing = baseline.seat(mode, pair);
        const played = await policyGame({
          engine,
          formatId: config.format_id,
          seed: streams.seed,
          game: pair,
          teams,
          known_synthetic: pool.known_synthetic,
          seats: { [candidate]: seat, [rival(candidate)]: opposing.seat } as Record<Player, PolicySeat | Actor>,
          latency_budget: budget,
          maxTurns: config.generation.max_turns,
        });
        // The baseline's own report of this game (an incident game throws in `rowOf` first, with its message).
        const row = rowOf(match_id, game, pair, candidate, played);
        const report = opposing.report?.();
        if (report !== undefined) reports.push(report);
        matchRows.push(report === undefined ? row : { ...row, baseline_counts: report.counts });
        const now = clock();
        gameMs.push(now - gameStarted);
        // A match that outruns its budget fails the evaluation (fail closed): no partial record leaves.
        if (baseline.budget_ms !== undefined && now - started > baseline.budget_ms) {
          throw new ProgressError(`${match_id} passed its wall-clock budget of ${baseline.budget_ms} ms after ${game + 1} of ${count} games (${Math.round(now - started)} ms): nothing is committed`);
        }
        if (game < recorded) {
          const policies = { [candidate]: policy_id, [rival(candidate)]: opponent_policy } as Record<Player, string>;
          const stored_game = buildProgressGameRecord({
            run_id: config.seeds.run_key,
            generation,
            match_id,
            game,
            pair,
            teams,
            policies,
            played,
            ...(report === undefined ? {} : { baseline: { seat: rival(candidate), identity_hash: report.identity_hash, counts: report.counts } }),
          });
          stored.set(`${match_id}-${game}.json`, stored_game);
          hashes.push(progressRecordHash(stored_game));
        }
      }
      rowsByMatch.set(match_id, matchRows);
      wall_ms[match_id] = Math.round(clock() - started);
      game_ms[match_id] = { mean: Math.round(gameMs.reduce((a, b) => a + b, 0) / gameMs.length), max: Math.round(Math.max(...gameMs)) };
      matches.push(summarise(match_id, policy_id, baseline.opponent(mode), mode, matchRows, hashes, config.generation.max_turns, candidateKind.candidate_preview, reports));
    }
    }
  }
  // Matches and their rows go in match_id order, which the reader requires (`actor-rule`, `anchor-g*`, `previous`, `random-legal`).
  matches.sort((a, b) => (a.match_id < b.match_id ? -1 : 1));
  for (const match of matches) rows.push(...(rowsByMatch.get(match.match_id) as Row[]));
  const record = readProgressEvaluationRecordV2({
    schema_version: 2,
    data_class: "synthetic",
    run_id: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    generation,
    generation_manifest_sha256: options.generation_manifest_sha256,
    checkpoints: options.checkpoints,
    version_manifest: manifest,
    teams: { dataset_hash: pool.dataset_hash, split: "validation", pairs_used: Math.max(games, anchorGames, runGames) / 2 },
    plan: {
      seed_derivation: PROGRESS_SEED_DERIVATION,
      pairing: PROGRESS_PAIRING,
      games_per_match: games,
      recorded_games_per_match: record_games,
      ...(anchors.length === 0 ? {} : { anchor_games_per_match: anchorGames }),
      ...(versus === undefined ? {} : { run_games_per_match: runGames }),
      max_turns: config.generation.max_turns,
      latency_budget_ms: budget,
    },
    matches,
    games: rows,
  });
  return { record, game_records: stored, wall_ms, game_ms };
}

function summarise(match_id: string, policy_id: string, opponent: Match["opponent"], mode: ProgressMode, rows: readonly Row[], hashes: string[], maxTurns: number, candidate_preview: Match["candidate_preview"] | undefined, reports: readonly BaselineReport[]): Match {
  return {
    match_id,
    policy_id,
    opponent,
    mode,
    ...(candidate_preview === undefined ? {} : { candidate_preview }),
    games: rows.length,
    ...tallyRows(rows, maxTurns),
    game_record_hashes: hashes.sort(),
    ...(reports.length === 0 ? {} : { baseline: { identity_hash: (reports[0] as BaselineReport).identity_hash, counts: sumBaselineCounts(reports.map((report) => report.counts)) } }),
  };
}

/** Write a result's files under `dir` (created if needed): `progress.json` and `games/<name>.json`, each durable. */
export function writeProgress(dir: string, result: ProgressResult): void {
  mkdirSync(join(dir, "games"), { recursive: true });
  const put = (path: string, value: unknown): void => {
    const descriptor = openSync(path, "w");
    try {
      writeSync(descriptor, `${jcs(value as Json)}\n`);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
  };
  for (const [name, record] of result.game_records) put(join(dir, "games", name), record);
  put(join(dir, "progress.json"), result.record);
}
