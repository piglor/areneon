// The frozen past pool of a population run (ADR-021 sections 3 and 4, slice 10): the snapshots a member may
// play, the PFSP `f_hard` weights over them in integers, and the draw of which snapshots a member meets in a
// generation. Everything here is a pure function of one committed file, `past-pool.json` of the generation
// before (written by the Python driver, `areneon_ml.loop.past_pool`): the pool's membership (which snapshots
// exist, after eviction) and each member's pair-score tallies against them over the rating window. A killed
// generation therefore replays the same plan. Eviction and the tallies are the driver's; this module reads the
// file, checks it and turns it into weights and draws.

import { GenerationError } from "./league-generation.ts";
import { populationDraw } from "./population-seed.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** Names the file layout and the draw rules below; a change to either is a new number. */
export const PAST_POOL_FORMAT = "past-pool/1";
/** Draw indices of purpose `opponent-draw` for the past pool, clear of the ladder's `0..ladder pairs - 1`. */
export const PAST_SELECT_BASE = 1_000_000;
export const PAST_ASSIGN_BASE = 2_000_000;
const FULL_PPM = 1_000_000;

export class PastPoolError extends GenerationError {
  override name = "PastPoolError";
}

export interface PastSnapshot {
  /** `m<k>@g<G>`: member k's frozen pair at the end of generation G. */
  id: string;
  member: string;
  generation: number;
  policy_id: string;
  preview: string;
  battle: string;
}

/** A member's pair-score counts (0, 250000, 500000, 750000, 1000000 ppm) against one snapshot, over the rating window. */
export interface PastCell {
  member: string;
  snapshot: string;
  scores: [number, number, number, number, number];
}

export interface PastPoolFile {
  schema_version: 1;
  format: typeof PAST_POOL_FORMAT;
  run_id: string;
  run_config_hash: string;
  /** The committed generation this file follows: its snapshots are the pool the NEXT generation plays. */
  generation: number;
  snapshots: PastSnapshot[];
  cells: PastCell[];
}

type PastSettings = NonNullable<TrainingRunConfig["population"]>["past_pool"];

const SNAPSHOT_ID = /^m(0|[1-9][0-9]?)@g(0|[1-9][0-9]{0,8})$/;
const HEX64 = /^[0-9a-f]{64}$/;
const POLICY = /^policy:[0-9a-f]{64}$/;

const fail = (message: string): never => {
  throw new PastPoolError(message);
};

const memberIndex = (member: string): number => Number(member.slice(1));
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const hasKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean => Object.keys(value).sort().join() === [...keys].sort().join();

/**
 * `[generation, member index]` of the pool after `generation` is committed: snapshots at every multiple of
 * `snapshot_every` (generation 0 included), the newest `max_size - (generation-0 pairs)` of the others, and the
 * generation-0 pairs always (oldest evicted first, generation-0 pairs never). The Python twin is `pool_keys`.
 */
export function poolKeys(members: number, snapshotEvery: number, maxSize: number, generation: number): [number, number][] {
  const taken: [number, number][] = [];
  for (let g = 0; g <= generation; g += snapshotEvery) for (let k = 0; k < members; k++) taken.push([g, k]);
  const zero = taken.filter(([g]) => g === 0);
  const rest = taken.filter(([g]) => g !== 0);
  const room = Math.max(0, maxSize - zero.length);
  return [...zero, ...(room === 0 ? [] : rest.slice(-room))];
}

/** An empty pool, for a check that only validates a config. */
export const emptyPastPool = (config: TrainingRunConfig, generation = 0): PastPoolFile => ({ schema_version: 1, format: PAST_POOL_FORMAT, run_id: config.seeds.run_key, run_config_hash: "", generation, snapshots: [], cells: [] });

/**
 * Read `past-pool.json`: the shape, the run, the sort order (snapshots by generation then member index), snapshot
 * ids that equal their member and generation, generations on the `snapshot_every` grid and below the file's, a size
 * within `max_size` (generation-0 pairs are never evicted, so the pool may exceed it by exactly those), and cells
 * sorted by member then snapshot, each naming a member of the run and a snapshot of the pool, with non-negative counts.
 */
export function readPastPool(value: unknown, config: TrainingRunConfig, runConfigHash: string): PastPoolFile {
  const population = config.population;
  if (population === undefined) return fail("past pool: the config has no population section");
  if (!isObject(value) || !hasKeys(value, ["cells", "format", "generation", "run_config_hash", "run_id", "schema_version", "snapshots"])) return fail("past pool: not a past-pool file");
  if (value.schema_version !== 1 || value.format !== PAST_POOL_FORMAT) return fail("past pool: not a version 1 past-pool file");
  if (value.run_id !== config.seeds.run_key || value.run_config_hash !== runConfigHash) return fail("past pool: the file belongs to another run or config");
  const generation = value.generation;
  if (!Number.isSafeInteger(generation) || (generation as number) < 0) return fail("past pool: generation must be a non-negative integer");
  if (!Array.isArray(value.snapshots) || !Array.isArray(value.cells)) return fail("past pool: snapshots and cells must be lists");
  const settings = population.past_pool;
  const snapshots = value.snapshots.map((entry): PastSnapshot => {
    if (!isObject(entry) || !hasKeys(entry, ["battle", "generation", "id", "member", "policy_id", "preview"])) return fail("past pool: a snapshot has the wrong keys");
    const { id, member, policy_id, preview, battle } = entry as Record<string, unknown>;
    const at = entry.generation;
    const match = typeof id === "string" ? SNAPSHOT_ID.exec(id) : null;
    if (match === null || id !== `${String(member)}@g${String(at)}`) return fail("past pool: a snapshot's id must be its member and generation");
    if (!Number.isSafeInteger(at) || (at as number) % settings.snapshot_every !== 0 || (at as number) > (generation as number)) return fail("past pool: a snapshot must be frozen on the snapshot_every grid, no later than the file's generation");
    if (memberIndex(member as string) >= population.members) return fail("past pool: a snapshot names a member the run does not have");
    if (typeof policy_id !== "string" || !POLICY.test(policy_id) || typeof preview !== "string" || !HEX64.test(preview) || typeof battle !== "string" || !HEX64.test(battle)) return fail("past pool: a snapshot's pair ids are malformed");
    return { id: id as string, member: member as string, generation: at as number, policy_id, preview, battle };
  });
  const expected = poolKeys(population.members, settings.snapshot_every, settings.max_size, generation as number);
  if (snapshots.length !== expected.length || snapshots.some((entry, index) => entry.generation !== (expected[index] as [number, number])[0] || memberIndex(entry.member) !== (expected[index] as [number, number])[1])) {
    return fail("past pool: the snapshots are not the pool the config gives at that generation");
  }
  const ids = new Set(snapshots.map((entry) => entry.id));
  const cells = value.cells.map((entry): PastCell => {
    if (!isObject(entry) || !hasKeys(entry, ["member", "scores", "snapshot"])) return fail("past pool: a cell has the wrong keys");
    const { member, snapshot, scores } = entry as { member: unknown; snapshot: unknown; scores: unknown };
    if (typeof member !== "string" || !/^m(0|[1-9][0-9]?)$/.test(member) || memberIndex(member) >= population.members) return fail("past pool: a cell names a member the run does not have");
    if (typeof snapshot !== "string" || !ids.has(snapshot)) return fail("past pool: a cell names a snapshot that is not in the pool");
    if (!Array.isArray(scores) || scores.length !== 5 || scores.some((n) => !Number.isSafeInteger(n) || (n as number) < 0) || (scores as number[]).reduce((a, b) => a + b, 0) < 1) return fail("past pool: a cell needs five non-negative counts and at least one pair");
    return { member, snapshot, scores: scores as PastCell["scores"] };
  });
  const keyOf = (cell: PastCell): string => `${String(memberIndex(cell.member)).padStart(4, "0")}/${cell.snapshot}`;
  if (!cells.every((cell, index) => index === 0 || keyOf(cells[index - 1] as PastCell) < keyOf(cell))) return fail("past pool: cells must be sorted by member then snapshot, each once");
  return { schema_version: 1, format: PAST_POOL_FORMAT, run_id: value.run_id as string, run_config_hash: runConfigHash, generation: generation as number, snapshots, cells };
}

/**
 * The PFSP `f_hard` weight (ADR-021 section 3) in integers: `max(floor_ppm, (10^6 - s)^p / 10^(6 (p - 1)))`, `s` the
 * member's mean pair score against the snapshot in ppm (`score_sum_ppm // pairs`), the quotient floored. A snapshot with
 * no games gets the largest weight the rule gives, 10^6 (a score of 0), which is also what OpenAI Five does for a new
 * opponent. The weight is at most 10^6 and at least `floor_ppm`.
 */
export function pfspWeight(settings: PastSettings, scores: readonly number[] | undefined): number {
  if (scores === undefined) return FULL_PPM;
  const pairs = scores.reduce((a, b) => a + b, 0);
  if (pairs < 1) return FULL_PPM;
  const sum = 250_000 * scores.reduce((total, count, index) => total + index * count, 0);
  const s = Math.floor(sum / pairs);
  const p = BigInt(settings.pfsp_power);
  const raw = BigInt(FULL_PPM - s) ** p / 10n ** (6n * (p - 1n));
  return Math.max(settings.floor_ppm, Number(raw));
}

/** The weights of the snapshots in `candidates` for `member`, in the candidates' order. */
export function pfspWeights(settings: PastSettings, pool: PastPoolFile, member: string, candidates: readonly PastSnapshot[]): number[] {
  const cells = new Map(pool.cells.filter((cell) => cell.member === member).map((cell) => [cell.snapshot, cell.scores]));
  return candidates.map((snapshot) => pfspWeight(settings, cells.get(snapshot.id)));
}

/** A weighted pick: `draw % total` falls in the cumulative weights; all-zero weights count as equal. */
function pickWeighted(weights: readonly number[], draw: bigint): number {
  const total = weights.reduce((a, b) => a + b, 0);
  const equal = total === 0;
  const sum = equal ? weights.length : total;
  let at = Number(draw % BigInt(sum));
  for (let i = 0; i < weights.length; i++) {
    at -= equal ? 1 : (weights[i] as number);
    if (at < 0) return i;
  }
  return weights.length - 1;
}

/**
 * The snapshots one member meets this generation, one per past pair, in pair order. The candidates are the pool minus the
 * snapshots that are the member's own current pair (`current_policy_id`): playing it would put one policy on both seats. No
 * candidate means no past pair (the caller gives those seats to self-play). At most
 * `min(max_past_per_member, candidates, pairs)` distinct snapshots are drawn, one by one without replacement weighted by
 * `pfspWeight` (`opponent-draw` index `PAST_SELECT_BASE + j`); each pair then takes one of those, weighted again
 * (`PAST_ASSIGN_BASE + i`). The 64-bit draw is reduced modulo the weight total (at most 64 * 10^6), a bias below 2^-40.
 */
export function drawPast(config: TrainingRunConfig, generation: number, member: string, pairs: number, current_policy_id: string, pool: PastPoolFile): PastSnapshot[] {
  const population = config.population;
  if (population === undefined) return fail("past pool: the config has no population section");
  const settings = population.past_pool;
  const candidates = pool.snapshots.filter((snapshot) => snapshot.policy_id !== current_policy_id);
  if (candidates.length === 0 || pairs === 0) return [];
  const weights = pfspWeights(settings, pool, member, candidates);
  const run_key = config.seeds.run_key;
  const wanted = Math.min(settings.max_past_per_member, candidates.length, pairs);
  const remaining = candidates.map((_, index) => index);
  const chosen: number[] = [];
  for (let j = 0; j < wanted; j++) {
    const at = pickWeighted(
      remaining.map((index) => weights[index] as number),
      populationDraw(run_key, generation, member, "opponent-draw", PAST_SELECT_BASE + j),
    );
    chosen.push(remaining.splice(at, 1)[0] as number);
  }
  chosen.sort((a, b) => a - b);
  const chosenWeights = chosen.map((index) => weights[index] as number);
  const out: PastSnapshot[] = [];
  for (let i = 0; i < pairs; i++) {
    out.push(candidates[chosen[pickWeighted(chosenWeights, populationDraw(run_key, generation, member, "opponent-draw", PAST_ASSIGN_BASE + i))] as number] as PastSnapshot);
  }
  return out;
}
