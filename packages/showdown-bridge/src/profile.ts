// The profile timers of a game process (ADR-028 section 1, #1191): where a generation's play time goes.
// Off unless `enableProfile()` was called, so a process that never asks pays one boolean test per call and
// every value it computes is untouched: a timer only reads the clock. The totals are diagnostics for the
// run-metrics record (`profile`); nothing in the play path, the loader or the learner reads them, and they
// reach no file the generation manifest hashes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

/** The phases a game process times (milliseconds, wall clock). */
export const PROFILE_PHASES = ["e_sim", "e_judge", "e_probe", "o_obs", "ipc_wait", "pool_load", "merge"] as const;
export type ProfilePhase = (typeof PROFILE_PHASES)[number];

let on = false;
const totals: Record<ProfilePhase, number> = { e_sim: 0, e_judge: 0, e_probe: 0, o_obs: 0, ipc_wait: 0, pool_load: 0, merge: 0 };
let decisions = 0;

/** Start timing in this process. */
export function enableProfile(): void {
  on = true;
}

/** Whether this process times its phases. */
export function profiling(): boolean {
  return on;
}

/** Forget the totals and stop timing (tests). */
export function resetProfile(): void {
  on = false;
  decisions = 0;
  for (const phase of PROFILE_PHASES) totals[phase] = 0;
}

/** A mark for `addSince`: the clock when profiling, else 0 (no clock read). */
export function startTimer(): number {
  return on ? performance.now() : 0;
}

/** Add the time since `mark` to `phase` (nothing when profiling is off). */
export function addSince(phase: ProfilePhase, mark: number): void {
  if (on) totals[phase] += performance.now() - mark;
}

/** Run `fn`, adding its wall time to `phase`; exactly `fn()` when profiling is off. */
export function timed<T>(phase: ProfilePhase, fn: () => T): T {
  if (!on) return fn();
  const mark = performance.now();
  try {
    return fn();
  } finally {
    totals[phase] += performance.now() - mark;
  }
}

/** As `timed`, for a function that returns a promise: the time until it settles. */
export function timedAsync<T>(phase: ProfilePhase, fn: () => Promise<T>): Promise<T> {
  if (!on) return fn();
  const mark = performance.now();
  return fn().finally(() => {
    totals[phase] += performance.now() - mark;
  });
}

/** Count Episode decision entries (both seats, baseline seats included). */
export function countDecisions(count: number): void {
  if (on) decisions += count;
}

/** One game process's figures (whole milliseconds; the peak resident set is omitted where it cannot be read). */
export interface GameProcessFigures {
  e_sim_ms: number;
  e_judge_ms: number;
  e_probe_ms: number;
  o_obs_ms: number;
  ipc_wait_ms: number;
  pool_load_ms: number;
  merge_ms: number;
  cpu_ms: number;
  peak_rss_bytes?: number;
  decisions: number;
}

/** The process's totals so far; `peak_rss_bytes` is the caller's reading (it knows where `/proc` is). */
export function processFigures(peak_rss_bytes: number | null): GameProcessFigures {
  const cpu = process.cpuUsage();
  const whole = (value: number): number => Math.floor(value);
  return {
    e_sim_ms: whole(totals.e_sim),
    e_judge_ms: whole(totals.e_judge),
    e_probe_ms: whole(totals.e_probe),
    o_obs_ms: whole(totals.o_obs),
    ipc_wait_ms: whole(totals.ipc_wait),
    pool_load_ms: whole(totals.pool_load),
    merge_ms: whole(totals.merge),
    cpu_ms: whole((cpu.user + cpu.system) / 1000),
    ...(peak_rss_bytes === null ? {} : { peak_rss_bytes }),
    decisions,
  };
}

/** One policy worker's figures, as the worker left them (`worker-<pid>.json`). */
export interface WorkerFigures {
  cpu_ms: number;
  peak_rss_bytes?: number;
  queries: number;
  w_enc_ms: number;
  w_fwd_ms: number;
  w_enc_cpu_ms: number;
  w_fwd_cpu_ms: number;
}

const GAME_PREFIX = "game-";
const WORKER_PREFIX = "worker-";

/** Leave this process's figures in `dir` as `game-<index>.json`. */
export function writeProcessFigures(dir: string, index: number, figures: GameProcessFigures): void {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, `${GAME_PREFIX}${index}.json`), `${JSON.stringify(figures)}\n`);
}

const sum = (values: readonly number[]): number => values.reduce((a, b) => a + b, 0);

/** The `profile` line of the generation: every game process's and worker's file in `dir`, summed (the `RunMetrics` `profile` without its `load`). Files are read in name order so the lists are stable. */
export function readProfile(dir: string): Record<string, unknown> {
  const names = readdirSync(dir).sort();
  const read = <T>(prefix: string): T[] => names.filter((name) => name.startsWith(prefix) && name.endsWith(".json")).map((name) => JSON.parse(readFileSync(join(dir, name), "utf8")) as T);
  const games = read<GameProcessFigures>(GAME_PREFIX);
  const workers = read<WorkerFigures>(WORKER_PREFIX);
  const withRss = <T extends { cpu_ms: number; peak_rss_bytes?: number }>(figures: T[]): { cpu_ms: number; peak_rss_bytes?: number }[] => figures.map(({ cpu_ms, peak_rss_bytes }) => ({ cpu_ms, ...(peak_rss_bytes === undefined ? {} : { peak_rss_bytes }) }));
  return {
    decisions: sum(games.map((g) => g.decisions)),
    game_processes: {
      processes: games.length,
      cpu_ms: sum(games.map((g) => g.cpu_ms)),
      e_sim_ms: sum(games.map((g) => g.e_sim_ms)),
      e_judge_ms: sum(games.map((g) => g.e_judge_ms)),
      e_probe_ms: sum(games.map((g) => g.e_probe_ms)),
      o_obs_ms: sum(games.map((g) => g.o_obs_ms)),
      ipc_wait_ms: sum(games.map((g) => g.ipc_wait_ms)),
      pool_load_ms: sum(games.map((g) => g.pool_load_ms)),
      merge_ms: sum(games.map((g) => g.merge_ms)),
      per_process: games.map((g) => ({ ...withRss([g])[0], decisions: g.decisions })),
    },
    policy_workers: {
      workers: workers.length,
      queries: sum(workers.map((w) => w.queries)),
      cpu_ms: sum(workers.map((w) => w.cpu_ms)),
      w_enc_ms: sum(workers.map((w) => w.w_enc_ms)),
      w_fwd_ms: sum(workers.map((w) => w.w_fwd_ms)),
      w_enc_cpu_ms: sum(workers.map((w) => w.w_enc_cpu_ms)),
      w_fwd_cpu_ms: sum(workers.map((w) => w.w_fwd_cpu_ms)),
      per_process: withRss(workers),
    },
  };
}
