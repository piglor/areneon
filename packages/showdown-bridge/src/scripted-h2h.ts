// The scripted head-to-head (ADR-025 section 13, ticket #1130): two scripted baselines play each other on the
// ADR-017 E2 plan, built on the progress evaluation's machinery and with no learned policy and no worker.
//
// * Plan: the validation split's disjoint ordered pairs (`progress-pairing/1`, `progressPlan`), every pair played
//   TWICE, arm `a` on p1 for an even game and on p2 for an odd one, the teams staying on their seats, both games of
//   a pair sharing the pair's battle seed (`progress-seed/1`). The pair is the unit of the analysis.
// * Arms: a baseline is built from its reviewed config record (`damage-aware@v1` and `@v2` read theirs, `v2` also the
//   belief port of the composition root), named in the record by its baseline identity hash, and plays under
//   `guardActor` with role `scripted`: any harness incident (a baseline that throws, an overrun) fails the run and
//   nothing is written. A forfeit-free scripted game is the normal case; a forfeit is a scored loss for that arm.
// * A game is a function of its index alone (the pair's seed, the arms' own streams), so any split of the games over
//   game processes gives the same rows; the rows are merged in game order and the record has no wall-clock value.
// * `wall_time_cap_ms` is a declared cap: a run that passes it, checked after every game, fails as a gate failure and
//   commits nothing.

import { readerSha256 } from "@areneon/public-state";
import { type DamageAwareActor, damageAwareActor } from "./damage-aware-actor.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { damageAwareIdentity, damageAwareIdentityHash } from "./damage-aware-identity.ts";
import type { Engine } from "./engine.ts";
import { guardActor, type LatencyBudget, type PlayedGame } from "./evaluation-harness.ts";
import { resolveFormat } from "./formats.ts";
import { firstLegal } from "./golden.ts";
import { damageAwareSetup, damageAwareSetupV2, type DamageAwareSetup } from "./league-generation.ts";
import { policyGame } from "./policy-actor.ts";
import { PROGRESS_PAIRING, PROGRESS_SEED_DERIVATION, progressGameStreams } from "./progress-seed.ts";
import { type ProgressPool, progressPlan } from "./progress-evaluation.ts";
import { type BaselineCounts, isDamageAware, sumBaselineCounts } from "./progress-records.ts";
import { randomLegalActor } from "./random-legal.ts";
import { type BeliefCounts, readScriptedH2HRecord, type ScriptedGameRow, type ScriptedHeadToHeadRecord, sumBeliefCounts, tallyScripted } from "./scripted-h2h-record.ts";
import type { Actor, Player } from "./runner.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** The scripted baselines a head-to-head can play, by policy id. */
export const SCRIPTED_POLICY_IDS = ["actor-rule@v1", "random-legal@v1", "damage-aware@v1", "damage-aware@v2"] as const;

/** A head-to-head that cannot be made: a refusal, a harness incident, or the wall-time gate. */
export class ScriptedH2HError extends Error {
  override name = "ScriptedH2HError";
}

/** What a damage-aware arm reports of one game. */
export interface ArmGameReport {
  counts: BaselineCounts;
  /** Present for an arm that plays on belief worlds. */
  belief?: BeliefCounts;
}

export interface ScriptedArm {
  policy_id: string;
  /** The baseline identity hash (ADR-018 section 5); present for a damage-aware arm. */
  identity_hash?: string;
  /** The belief the arm plays on (damage-aware@v2): its identity, prior and config hashes and the worlds per request. */
  belief?: { identity_hash: string; prior_hash: string; belief_config_hash: string; worlds: number };
  /** A fresh guarded seat for one game; `baselineSeed` is the pair's baseline stream seed. */
  seat: (baselineSeed: number) => { actor: Actor; report?: () => ArmGameReport };
}

export interface ScriptedArmContext {
  engine: Engine;
  formatId: string;
  budget: LatencyBudget;
  /** The composition root's belief port factory; `damage-aware@v2` needs it. */
  beliefWorlds?: BeliefWorldsFactory | undefined;
}

/** Build one arm from its policy id and the reviewed records it reads. */
export function scriptedArm(policyId: string, context: ScriptedArmContext): ScriptedArm {
  const { engine, formatId, budget } = context;
  const guarded = (actor: Actor): Actor => guardActor(actor, { role: "scripted", latency_budget: budget });
  if (policyId === "actor-rule@v1") return { policy_id: policyId, seat: () => ({ actor: guarded(firstLegal) }) };
  if (policyId === "random-legal@v1") return { policy_id: policyId, seat: (seed) => ({ actor: guarded(randomLegalActor(seed)) }) };
  if (policyId !== "damage-aware@v1" && policyId !== "damage-aware@v2") throw new ScriptedH2HError(`${policyId} is not a scripted baseline of the head-to-head (${SCRIPTED_POLICY_IDS.join(", ")})`);
  const v2 = policyId === "damage-aware@v2";
  if (v2 && context.beliefWorlds === undefined) throw new ScriptedH2HError("damage-aware@v2 needs the belief port of the composition root (packages/team-dataset)");
  let setup: DamageAwareSetup;
  try {
    setup = v2 ? damageAwareSetupV2(engine, formatId, context.beliefWorlds as BeliefWorldsFactory) : damageAwareSetup(engine, formatId);
  } catch (error) {
    throw new ScriptedH2HError(`${policyId}: ${(error as Error).message}`);
  }
  const manifest = resolveFormat(engine, formatId, setup.snapshot).manifest;
  const { worlds } = setup;
  const identity_hash = damageAwareIdentityHash(
    damageAwareIdentity({ config: setup.config, mechanics: setup.mechanics, manifest, reader_sha256: readerSha256(), ...(worlds === undefined ? {} : { belief_identity_hash: worlds.identity_hash }) }),
  );
  const named = setup.config.schema_version === 2 ? setup.config.assumed_set.worlds : 0;
  return {
    policy_id: policyId,
    identity_hash,
    ...(worlds === undefined ? {} : { belief: { identity_hash: worlds.identity_hash, prior_hash: worlds.prior_hash, belief_config_hash: worlds.belief_config_hash, worlds: named } }),
    seat: (seed) => {
      const actor: DamageAwareActor = damageAwareActor({ ...setup, engine, data_class: "synthetic", commit: policyId.slice(policyId.indexOf("@") + 1), seed });
      return {
        actor: guarded(actor),
        report: () => {
          const report = actor.report();
          if (report.policy_id !== policyId) throw new ScriptedH2HError(`the arm ${policyId} plays as ${report.policy_id}`);
          if (report.identity_hash !== identity_hash) throw new ScriptedH2HError(`${policyId}: a game carries another baseline identity hash than the arm's records give`);
          // probe_refusals is written only above 0, so a baseline without refusals keeps the counts' bytes.
          const { probe_refusals, ...rest } = report.counts;
          const counts = probe_refusals > 0 ? { ...rest, probe_refusals } : rest;
          if (report.belief === undefined) return { counts };
          const { worlds: _worlds, refusals, ...others } = report.belief;
          return { counts, belief: { ...others, refusals: Object.keys(refusals).sort().map((code) => ({ code, count: refusals[code] as number })) } };
        },
      };
    },
  };
}

export interface ScriptedPlayOptions {
  engine: Engine;
  config: TrainingRunConfig;
  pool: ProgressPool;
  /** Validation pairs to play: `2 * pairs` games. */
  pairs: number;
  arms: { a: ScriptedArm; b: ScriptedArm };
  /** Play only the games `g mod of = index`; absent plays all. */
  shard?: { index: number; of: number };
  /** The declared wall-clock cap in milliseconds; passing it fails the run (nothing is committed). */
  wall_time_cap_ms?: number;
  /** The clock the cap is measured with (default `performance.now`); a test seam. */
  clock?: () => number;
}

const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");

/** One game's row; an incident game (a harness failure) is an error. */
function rowOf(game: number, aSeat: Player, played: PlayedGame, seats: { a: { report?: () => ArmGameReport }; b: { report?: () => ArmGameReport } }): ScriptedGameRow {
  const { outcome } = played;
  if (outcome.kind === "incident") throw new ScriptedH2HError(`game ${game}: ${outcome.incident}: ${outcome.message}`);
  const a = seats.a.report?.();
  const b = seats.b.report?.();
  const who = (seat: Player | null): "a" | "b" | "tie" => (seat === null ? "tie" : seat === aSeat ? "a" : "b");
  return {
    game,
    pair: game >> 1,
    a_seat: aSeat as "p1" | "p2",
    outcome:
      outcome.kind === "forfeit"
        ? { kind: "forfeit", loser: who(outcome.loser) as "a" | "b", cause: outcome.cause, request_seq: outcome.request_seq, turns: outcome.turns }
        : { kind: "played", winner: who(outcome.winner), turns: outcome.turns },
    ...(a === undefined ? {} : { a_counts: a.counts }),
    ...(b === undefined ? {} : { b_counts: b.counts }),
    ...(a?.belief === undefined ? {} : { a_belief: a.belief }),
    ...(b?.belief === undefined ? {} : { b_belief: b.belief }),
  };
}

/** Play this process's share of the games and return their rows in game order. */
export async function playScriptedGames(options: ScriptedPlayOptions): Promise<ScriptedGameRow[]> {
  const { engine, config, pool, pairs, arms, shard } = options;
  if (!Number.isSafeInteger(pairs) || pairs < 1) throw new ScriptedH2HError("--pairs must be an integer of at least 1");
  if (arms.a.policy_id === arms.b.policy_id) throw new ScriptedH2HError("the two arms must be different baselines");
  if (shard !== undefined && (!Number.isSafeInteger(shard.of) || shard.of < 1 || !Number.isSafeInteger(shard.index) || shard.index < 0 || shard.index >= shard.of)) throw new ScriptedH2HError("a shard is INDEX/COUNT with INDEX below COUNT");
  const cap = options.wall_time_cap_ms;
  if (cap !== undefined && (!Number.isSafeInteger(cap) || cap < 1)) throw new ScriptedH2HError("the wall-time cap must be a positive whole number of milliseconds");
  let plan;
  try {
    // No learned policy trains here, so there is no team set or league plan to keep the pairs clear of.
    plan = progressPlan({ config, pool, games: pairs * 2, training_teams: new Set(), league_seeds: new Set() });
  } catch (error) {
    throw new ScriptedH2HError((error as Error).message);
  }
  const clock = options.clock ?? (() => performance.now());
  const budget: LatencyBudget = { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
  const rows: ScriptedGameRow[] = [];
  const started = clock();
  for (let game = 0; game < pairs * 2; game++) {
    if (shard !== undefined && game % shard.of !== shard.index) continue;
    const pair = game >> 1;
    const aSeat: Player = game % 2 === 0 ? "p1" : "p2";
    const [first, second] = plan.pairs[pair] as (typeof plan.pairs)[number];
    const streams = progressGameStreams(config.seeds.run_key, pair);
    const seatA = arms.a.seat(streams.baseline_seed);
    const seatB = arms.b.seat(streams.baseline_seed);
    const played = await policyGame({
      engine,
      formatId: config.format_id,
      seed: streams.seed,
      game: pair,
      teams: { p1: first.sets, p2: second.sets },
      known_synthetic: pool.known_synthetic,
      seats: { [aSeat]: seatA.actor, [rival(aSeat)]: seatB.actor } as Record<Player, Actor>,
      latency_budget: budget,
      maxTurns: config.generation.max_turns,
    });
    rows.push(rowOf(game, aSeat, played, { a: seatA, b: seatB }));
    const elapsed = clock() - started;
    if (cap !== undefined && elapsed > cap) {
      throw new ScriptedH2HError(`the wall-time cap of ${cap} ms passed after ${rows.length} games of this process (${Math.round(elapsed)} ms): a gate failure, nothing is committed`);
    }
  }
  return rows;
}

/** The record of a head-to-head from every game's row; the games must be exactly 0..2*pairs-1, each once. */
export function buildScriptedRecord(args: { engine: Engine; config: TrainingRunConfig; pool: ProgressPool; pairs: number; arms: { a: ScriptedArm; b: ScriptedArm }; rows: readonly ScriptedGameRow[]; wall_time_cap_ms?: number }): ScriptedHeadToHeadRecord {
  const { engine, config, pool, pairs, arms } = args;
  const rows = [...args.rows].sort((x, y) => x.game - y.game);
  if (rows.length !== pairs * 2 || rows.some((row, index) => row.game !== index)) throw new ScriptedH2HError("a game is missing or was played twice");
  const maxTurns = config.generation.max_turns;
  const armOf = (arm: ScriptedArm, side: "a" | "b"): ScriptedHeadToHeadRecord["arms"]["a"] => {
    const counts = rows.map((row) => row[`${side}_counts`]);
    const beliefs = rows.map((row) => row[`${side}_belief`]);
    return {
      policy_id: arm.policy_id,
      ...(arm.identity_hash === undefined ? {} : { identity_hash: arm.identity_hash }),
      ...(isDamageAware(arm.policy_id) ? { counts: sumBaselineCounts(counts as BaselineCounts[]) } : {}),
      ...(arm.belief === undefined ? {} : { belief: { ...arm.belief, totals: sumBeliefCounts(beliefs as BeliefCounts[]) } }),
    };
  };
  return readScriptedH2HRecord({
    schema_version: 1,
    data_class: "synthetic",
    run_id: config.seeds.run_key,
    version_manifest: resolveFormat(engine, config.format_id).manifest,
    teams: { dataset_hash: pool.dataset_hash, split: "validation", pairs_used: pairs },
    plan: {
      seed_derivation: PROGRESS_SEED_DERIVATION,
      pairing: PROGRESS_PAIRING,
      games: pairs * 2,
      max_turns: maxTurns,
      latency_budget_ms: { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms },
      ...(args.wall_time_cap_ms === undefined ? {} : { wall_time_cap_ms: args.wall_time_cap_ms }),
    },
    arms: { a: armOf(arms.a, "a"), b: armOf(arms.b, "b") },
    result: tallyScripted(rows, maxTurns),
    games: rows,
  });
}
