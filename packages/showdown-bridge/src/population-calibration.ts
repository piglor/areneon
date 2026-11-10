// The anchor calibration of a population run (ADR-021 section 4, slice 7): generation 0 plays every pair of anchors
// (`random-legal@v1`, `actor-rule@v1` and, when the config lists it, `damage-aware@v1`) for
// `population.rating.calibration_pairs` team pairs, each pair played twice with the anchors' seats swapped (the teams stay on
// their seats, as every pair of the ladder does). The Python driver fits the anchors' strengths from the games once and
// freezes them for the run. No policy worker is involved: both seats are scripted baselines behind the latency guard.
//
// Seeds come from `league-seed/1` at generation 0 (`calibrationGameStreams`), which no training plan uses. Anchor `a`'s
// stream is the pair's `baseline_seed` and anchor `b`'s is that plus one (modulo 2^32), by role and not by seat, so two
// anchors of one kind never repeat each other's draws.

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { baselineActor, baselineRecords, type DamageAwareSetup, GenerationError, type TrainingPool, writeDurable } from "./league-generation.ts";
import { type CalibrationGamePlan, planCalibration } from "./population-plan.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { policyGame } from "./policy-actor.ts";
import type { ForfeitCause, Player } from "./runner.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

/** Names the layout of `calibration.json`; a change to what it holds is a new number. */
export const CALIBRATION_FORMAT = "population-calibration/1";

export interface CalibrationOptions {
  engine: Engine;
  config: TrainingRunConfig;
  pool: TrainingPool;
  damage_aware?: DamageAwareSetup;
  damage_aware_v3?: DamageAwareSetup;
  damage_aware_v4?: DamageAwareSetup;
  damage_aware_v2?: DamageAwareSetup;
  belief_worlds?: BeliefWorldsFactory | undefined;
}

export interface CalibrationGameRow {
  game: number;
  pair: number;
  half: 0 | 1;
  seed: string;
  /** The two anchors' policy ids, `a` before `b`. */
  a: string;
  b: string;
  seats: Record<Player, string>;
  outcome: "played" | "forfeit";
  winner: Player | null;
  turns: number;
  forfeit?: { loser: Player; cause: ForfeitCause };
}

export interface CalibrationResult {
  run_key: string;
  run_config_hash: string;
  calibration_pairs: number;
  /** The anchors' policy ids, sorted. */
  anchors: string[];
  games: CalibrationGameRow[];
  manifest: VersionManifest;
  /** Hashes of the teams the calibration drew, sorted and unique. */
  team_hashes: string[];
}

/** Play the calibration plan, one game after the other. A harness incident in any game fails the calibration (nothing is retried). */
export async function playCalibration(options: CalibrationOptions): Promise<CalibrationResult> {
  const { engine, config, pool } = options;
  const plan = planCalibration(config, pool.teams.length);
  const withRecords = { ...options, ...baselineRecords(options, config) };
  const budget = { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
  const rows: CalibrationGameRow[] = [];
  for (const entry of plan) {
    const seatFor = (anchor: CalibrationGamePlan["a"]): ReturnType<typeof baselineActor>["actor"] => {
      const seed = anchor.policy_id === entry.a.policy_id ? entry.baseline_seed : (entry.baseline_seed + 1) % 2 ** 32;
      return baselineActor(withRecords, anchor, { baseline_seed: seed }, budget).actor;
    };
    const { outcome, record } = await policyGame({
      engine,
      formatId: config.format_id,
      seed: entry.seed,
      game: entry.game,
      teams: { p1: pool.teams[entry.teams.p1] as PokemonSet[], p2: pool.teams[entry.teams.p2] as PokemonSet[] },
      known_synthetic: pool.known_synthetic,
      seats: { p1: seatFor(entry.seats.p1), p2: seatFor(entry.seats.p2) },
      latency_budget: budget,
      maxTurns: config.generation.max_turns,
    });
    if (outcome.kind === "incident") throw new GenerationError(`calibration game ${entry.game} (${entry.seed}): ${outcome.incident}: ${outcome.message}`);
    const base = { game: entry.game, pair: entry.pair, half: entry.half, seed: entry.seed, a: entry.a.policy_id, b: entry.b.policy_id, seats: { p1: entry.seats.p1.policy_id, p2: entry.seats.p2.policy_id } };
    if (outcome.kind === "forfeit") {
      rows.push({ ...base, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser: outcome.loser, cause: outcome.cause } });
      continue;
    }
    if (record === null) throw new GenerationError(`calibration game ${entry.game} (${entry.seed}): a played game has no record`);
    rows.push({ ...base, outcome: "played", winner: record.winner, turns: record.turns });
  }
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) drawn.add(pool.hashes[entry.teams[player]] as string);
  return {
    run_key: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    calibration_pairs: config.population?.rating.calibration_pairs as number,
    anchors: [...new Set(plan.flatMap((entry) => [entry.a.policy_id, entry.b.policy_id]))].sort(),
    games: rows,
    manifest: resolveFormat(engine, config.format_id).manifest,
    team_hashes: [...drawn].sort(),
  };
}

/** Write `calibration.json` below `dir`: the schema version, the format name and the result. */
export function writeCalibration(dir: string, result: CalibrationResult): void {
  mkdirSync(dir, { recursive: true });
  writeDurable(join(dir, "calibration.json"), `${jcs({ schema_version: 1, format: CALIBRATION_FORMAT, ...result } as unknown as Json)}\n`);
}
