// The measurement behind ADR-022 section 5's step-2 trigger (ticket #1030): the cost of the candidate feature
// builder on recorded battle requests, and the "probe-dominated" rate of the recorded choices. Evaluation code
// only: it reads the recorded decisions of finished games (each seat's own channel, own team and candidates,
// the choice taken) and nothing else; no module of the actors, the runner or the loop imports it. Output is
// counts, integers and milliseconds, never a species, move, team or channel line.
//
// Sources: an Episode of a training generation (its own channel and team; the candidates come from
// `enumerateCandidates` on the request, as `policyActor` makes them) and a stored progress game (the omniscient
// record is never read for a channel: the battle is replayed through the runner from its seed, teams and
// choices, and each seat's own channel is the replay's).
import { type CandidateFeatures, type FeatureRow } from "./candidate-features.ts";
import { type Candidate, enumerateCandidates } from "./choices.ts";
import { parseJointChoice } from "./damage-aware-choice.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { FeaturePool } from "./feature-pool.ts";
import type { FeatureResult, FeatureServiceSpec, FeatureTask } from "./feature-service.ts";
import { decodeRequest } from "./request.ts";
import { resimulate } from "./resimulate.ts";
import { sha256Hex } from "./rules.ts";
import type { Player } from "./runner.ts";

/** One recorded decision of one seat: the task the builder is asked, and the choice that was taken. */
export interface BenchDecision {
  task: FeatureTask;
  chosen: string;
  /** Which kind of seat took it (a label the caller gives: `learned`, `baseline`, ...). */
  seat: string;
}

export interface Skipped {
  /** Requests that are not a move request (Team Preview, forced switch): the builder has no rows for them. */
  not_move: number;
  /** Games that could not be replayed. */
  unreplayable_games: number;
}

const REQUEST = "|request|";

/** The lines up to and including the player's `seq`-th request, or null. */
function cutAt(lines: readonly string[], seq: number): string[] | null {
  let seen = 0;
  const end = lines.findIndex((line) => line.startsWith(REQUEST) && ++seen === seq);
  return end < 0 ? null : lines.slice(0, end + 1);
}

/** Decisions of one seat from its channel, team and recorded choices: move requests only. */
function seatDecisions(game: string, seat: string, player: Player, channel: readonly string[], team: PokemonSet[], choices: { request_seq: number; choice: string }[], skipped: Skipped): BenchDecision[] {
  const out: BenchDecision[] = [];
  for (const { request_seq, choice } of choices) {
    const lines = cutAt(channel, request_seq);
    if (lines === null) continue;
    const decoded = decodeRequest((lines[lines.length - 1] as string).slice(REQUEST.length));
    if (decoded.phase !== "move") {
      skipped.not_move++;
      continue;
    }
    const candidates: Pick<Candidate, "choice" | "pending">[] = enumerateCandidates(decoded).candidates.map(({ choice: c, pending }) => ({ choice: c, pending }));
    out.push({ task: { game, player, lines, team, candidates }, chosen: choice, seat });
  }
  const last = out[out.length - 1];
  if (last !== undefined) last.task.final = true;
  return out;
}

/** The decisions of a stored Episode (parsed JSON of a file the loop wrote). `key` names the game and seat. */
export function episodeDecisions(episode: unknown, key: string, seat: string, skipped: Skipped): BenchDecision[] {
  if ((episode as { data_class?: unknown }).data_class !== "synthetic") throw new Error("the bench reads synthetic Episodes only");
  const e = episode as { player: Player; channel: string[]; team: { team: PokemonSet[] }; decisions: { choice: string; identity: { request_seq: number } }[] };
  return seatDecisions(key, seat, e.player, e.channel, e.team.team, e.decisions.map((d) => ({ request_seq: d.identity.request_seq, choice: d.choice })), skipped);
}

/** A stored progress game record (parsed JSON): the fields this reads. Its omniscient log is not read. */
export interface ProgressGame {
  data_class: string;
  seed: string;
  policies: Record<Player, string>;
  teams: Record<Player, { team: PokemonSet[] }>;
  decisions: { player: Player; request_seq: number; chosen: string }[];
}

/** Replay the stored game through the runner (seed, teams, choices) and take each seat's own channel from the replay. */
export async function progressDecisions(engine: Engine, formatId: string, maxTurns: number, record: ProgressGame, key: string, seatOf: (policy: string) => string, skipped: Skipped): Promise<BenchDecision[]> {
  if (record.data_class !== "synthetic") throw new Error("the bench reads synthetic progress games only");
  let channels: Record<Player, string[]>;
  try {
    channels = (
      await resimulate(engine, {
        format_id: formatId,
        seed: record.seed,
        max_turns: maxTurns,
        teams: { p1: record.teams.p1.team, p2: record.teams.p2.team },
        choices: record.decisions.map((d) => ({ player: d.player, request_seq: d.request_seq, choice: d.chosen })),
      })
    ).channels;
  } catch {
    skipped.unreplayable_games++;
    return [];
  }
  return (["p1", "p2"] as const).flatMap((player) =>
    seatDecisions(`${key}:${player}`, seatOf(record.policies[player]), player, channels[player], record.teams[player].team, record.decisions.filter((d) => d.player === player).map((d) => ({ request_seq: d.request_seq, choice: d.chosen })), skipped),
  );
}

// ---------------------------------------------------------------- the probe-dominated rate

export interface DominatedCount {
  /** Slot parts of recorded choices that play a move with a computed row and have a best row above 0. */
  evaluable: number;
  /** Of those: the chosen move's mean_ppm is below half the best row of the same Pokemon. */
  dominated: number;
  /** Slot parts that are not evaluable, by reason. */
  not_evaluable: { switch_or_pass: number; no_damage_row: number; unsupported_row: number; best_zero: number };
}

export const emptyDominated = (): DominatedCount => ({ evaluable: 0, dominated: 0, not_evaluable: { switch_or_pass: 0, no_damage_row: 0, unsupported_row: 0, best_zero: 0 } });

/**
 * The probe-dominated rate's counts for one recorded choice (EXP-DF2's secondary measure, "computed on assumed
 * sets"): per slot, the chosen move's value is the largest `mean_ppm` among the rows of that move (the chosen
 * target's row for a single target, any foe's for a spread move; the chosen Mega flag) and the best row is the
 * largest `mean_ppm` among all computed rows of the slot; it is dominated when twice the chosen value is below it.
 */
export function addDominated(into: DominatedCount, features: Pick<CandidateFeatures, "rows">, chosen: string): void {
  parseJointChoice(chosen).forEach((part, slot) => {
    if (part.kind !== "move") {
      into.not_evaluable.switch_or_pass++;
      return;
    }
    const rows: FeatureRow[] = features.rows.filter((r) => r.slot === slot);
    const own = rows.filter((r) => r.move === part.index && r.mega === part.mega && (part.target === null || r.target === part.target || rows.every((o) => o.move !== part.index || o.target !== part.target)));
    if (own.length === 0) return void into.not_evaluable.no_damage_row++;
    const computed = (list: FeatureRow[]): number[] => list.flatMap((r) => (r.kind === "computed" ? [r.mean_ppm] : []));
    const value = computed(own);
    if (value.length === 0) return void into.not_evaluable.unsupported_row++;
    const best = Math.max(...computed(rows));
    if (best === 0) return void into.not_evaluable.best_zero++;
    into.evaluable++;
    if (2 * Math.max(...value) < best) into.dominated++;
  });
}

// ---------------------------------------------------------------- statistics

const percentile = (sorted: number[], share: number): number => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * share) - 1)] as number;

export interface LatencySummary {
  n: number;
  median_ms: number;
  mean_ms: number;
  p95_ms: number;
  p99_ms: number;
  max_ms: number;
}

export function summarise(values: readonly number[]): LatencySummary | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const r = (x: number): number => Math.round(x * 10) / 10;
  return { n: sorted.length, median_ms: r(percentile(sorted, 0.5)), mean_ms: r(sorted.reduce((a, b) => a + b, 0) / sorted.length), p95_ms: r(percentile(sorted, 0.95)), p99_ms: r(percentile(sorted, 0.99)), max_ms: r(sorted[sorted.length - 1] as number) };
}

export interface RunStats {
  decisions: number;
  featured: number;
  errors: Record<string, number>;
  latency: LatencySummary | null;
  rows: { total: number; per_decision_mean_x100: number; unsupported: Record<string, number>; computed: number };
  probe: { queries: number; probe_runs: number; memo_hit_ppm: number; /** Refused queries turned into `unsupported` rows (#1246). */ probe_refusals: number };
  /** `probe-damage/2` only: the four new arrays, rows by kind. */
  arrays?: Record<string, { rows: number; computed: number; unsupported: Record<string, number> }>;
  /** `probe-damage/2` only: where the builder spent its probe queries and time, summed over the decisions (ms rounded). */
  parts?: Record<string, { queries: number; probe_runs: number; ms: number }>;
  total_builder_ms: number;
  /** SHA-256 of the JSON of every outcome in decision order: equal for equal rows. */
  rows_digest: string;
}

/** The arrays `probe-damage/2` adds to `rows`. */
const ARRAYS_2 = ["threats", "speeds", "own_priorities", "foe_priorities"] as const;

export function statsOf(results: readonly FeatureResult[]): RunStats {
  const errors: Record<string, number> = {};
  const unsupported: Record<string, number> = {};
  let rows = 0;
  let computed = 0;
  let featured = 0;
  let queries = 0;
  let runs = 0;
  let refusals = 0;
  const arrays: NonNullable<RunStats["arrays"]> = {};
  const parts: NonNullable<RunStats["parts"]> = {};
  for (const r of results) {
    for (const [name, cost] of Object.entries(r.parts ?? {})) {
      const sum = (parts[name] ??= { queries: 0, probe_runs: 0, ms: 0 });
      sum.queries += cost.queries;
      sum.probe_runs += cost.probe_runs;
      sum.ms += cost.ms;
    }
    queries += r.counts.queries;
    runs += r.counts.probe_runs;
    refusals += r.counts.probe_refusals;
    if (r.outcome.kind === "error") {
      errors[r.outcome.code] = (errors[r.outcome.code] ?? 0) + 1;
      continue;
    }
    featured++;
    for (const name of ARRAYS_2) {
      const list = (r.outcome.features as unknown as Record<string, { kind: string; reason?: string }[] | undefined>)[name];
      if (list === undefined) continue;
      const entry = (arrays[name] ??= { rows: 0, computed: 0, unsupported: {} });
      for (const row of list) {
        entry.rows++;
        if (row.kind === "computed") entry.computed++;
        else entry.unsupported[row.reason as string] = (entry.unsupported[row.reason as string] ?? 0) + 1;
      }
    }
    for (const row of r.outcome.features.rows) {
      rows++;
      if (row.kind === "computed") computed++;
      else unsupported[row.reason] = (unsupported[row.reason] ?? 0) + 1;
    }
  }
  return {
    decisions: results.length,
    featured,
    errors,
    latency: summarise(results.map((r) => r.ms)),
    rows: { total: rows, per_decision_mean_x100: featured === 0 ? 0 : Math.round((rows * 100) / featured), unsupported, computed },
    probe: { queries, probe_runs: runs, memo_hit_ppm: queries === 0 ? 0 : Math.floor(((queries - runs) * 1_000_000) / queries), probe_refusals: refusals },
    ...(Object.keys(arrays).length === 0 ? {} : { arrays }),
    ...(Object.keys(parts).length === 0 ? {} : { parts: Object.fromEntries(Object.entries(parts).map(([name, sum]) => [name, { ...sum, ms: Math.round(sum.ms) }])) }),
    total_builder_ms: Math.round(results.reduce((a, r) => a + r.ms, 0)),
    rows_digest: sha256Hex(JSON.stringify(results.map((r) => r.outcome))),
  };
}

// ---------------------------------------------------------------- the run

export interface PoolRun {
  size: number;
  wall_ms: number;
  decisions_per_second_x100: number;
  /** The single-thread builder time over this pool's wall time (x100): the measured speed-up, never above the thread count. */
  speedup_x100: number;
  /** The rows equal the single-thread run's, byte for byte. */
  rows_equal_single: boolean;
}

export interface BenchReport {
  single_thread: RunStats;
  pools: PoolRun[];
  dominated: Record<string, DominatedCount>;
}

/** Single-thread run (pool size 0, in decision order), then each pool size on the same decisions. */
export async function runBench(spec: FeatureServiceSpec, decisions: readonly BenchDecision[], poolSizes: readonly number[]): Promise<BenchReport> {
  const tasks = decisions.map((d) => d.task);
  const single = new FeaturePool(0, spec);
  const results = await single.run(tasks);
  const stats = statsOf(results);
  const dominated: Record<string, DominatedCount> = {};
  decisions.forEach((d, k) => {
    const outcome = (results[k] as FeatureResult).outcome;
    if (outcome.kind !== "features") return;
    addDominated((dominated[d.seat] ??= emptyDominated()), outcome.features, d.chosen);
  });
  const pools: PoolRun[] = [];
  for (const size of poolSizes) {
    const pool = new FeaturePool(size, spec);
    try {
      // Warm-up: one task of a distinct game per thread (the pool gives each to the least loaded thread), so every thread has loaded the engine before the clock starts.
      await pool.run(Array.from({ length: size }, (_, k) => ({ ...(tasks[0] as FeatureTask), game: `warm-${k}`, final: true })));
      const started = performance.now();
      const rerun = await pool.run(tasks.map((t) => ({ ...t, game: `${t.game}#${size}` })));
      const wall = performance.now() - started;
      pools.push({ size, wall_ms: Math.round(wall), decisions_per_second_x100: Math.round((tasks.length * 100_000) / wall), speedup_x100: Math.round((stats.total_builder_ms * 100) / wall), rows_equal_single: statsOf(rerun).rows_digest === stats.rows_digest });
    } finally {
      await pool.close();
    }
  }
  return { single_thread: stats, pools, dominated };
}

// ---------------------------------------------------------------- the generation wall-time estimate

export interface ControlPhases {
  play_ms: number;
  load_ms: number;
  /** Seat decisions of the control generation (its run-metrics `counts.decisions`). */
  decisions: number;
}

export interface WallTimeEstimate {
  control_play_plus_load_ms: number;
  limit_ms: number;
  builder_ms_per_decision_mean: number;
  /** Builder CPU time at play and again at stream writing, one thread. */
  builder_cpu_ms: number;
  by_threads: { threads: number; added_ms: number; total_ms: number; ratio_x100: number; met: boolean }[];
}

/**
 * Control play plus load, plus the builder run twice (at play and at stream writing) over every seat decision of
 * the control generation at the sample's mean cost per decision (a decision that is not a move request costs
 * nothing, so the mean is over all sample decisions the caller passes), divided over `threads`; the limit is
 * 1.5 times the control's play plus load. Estimate: builder time on otherwise idle threads, no overlap with play.
 */
export function estimateWallTime(control: ControlPhases, meanMsPerDecision: number, threads: readonly number[], limitPermille = 1500): WallTimeEstimate {
  const base = control.play_ms + control.load_ms;
  const cpu = 2 * meanMsPerDecision * control.decisions;
  return {
    control_play_plus_load_ms: base,
    limit_ms: Math.floor((base * limitPermille) / 1000),
    builder_ms_per_decision_mean: Math.round(meanMsPerDecision * 10) / 10,
    builder_cpu_ms: Math.round(cpu),
    by_threads: threads.map((n) => {
      const added = Math.round(cpu / n);
      const total = base + added;
      return { threads: n, added_ms: added, total_ms: total, ratio_x100: Math.round((total * 100) / base), met: total * 1000 <= base * limitPermille };
    }),
  };
}
