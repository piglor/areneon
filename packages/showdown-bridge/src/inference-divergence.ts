// How far batched self-play inference strayed from batch 1 (ADR-028 section 2 F2, #1195). A diagnostic of the training play phase:
// nothing here is read by the loop, the loader or the learner, written to a hashed file, or part of any identity.
//
// The pair's own seats report each decision they play (`on_answer`). A deterministic part of them is kept (a hash of the decision's
// own sampling draw and request, so the same decisions are kept whatever the order the games finished in), and when the games are
// done the kept queries are scored again at batch 1 with each decision's own `u`: the share whose choice differs from the played
// one, and the largest difference between the batched and the batch-1 prior (the policy's probabilities at temperature 1, in ppm).
//
// A run whose `generation.inference` says `device: "cuda"` (ADR-030 step 4, #1458) plays on a GPU worker started from the unshipped
// `ml/gpu-env` Python; the batch-1 control is always the CPU worker, so the record (which names the device) is then the GPU worker's
// share of choice differences. GPU and CPU results are not guaranteed equal.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type Json, jcs } from "./canonical-json.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import type { EvaluateQueryInput, InferenceSettings, PolicyQueryInput } from "./policy-protocol.ts";

/** The share of the decisions the sample must reach (ppm), and the least number of decisions when that many were played. */
export const DIVERGENCE_SHARE_PPM = 10_000;
export const DIVERGENCE_MIN_SAMPLE = 256;
/**
 * While the games run the queries of the decisions with the smallest keys are kept, at least `DIVERGENCE_KEEP_FLOOR` of them and
 * `DIVERGENCE_KEEP_PPM` of the decisions so far (twice the target share), so the final sample (the smallest keys of all) is a
 * subset of what was kept: a decision dropped for a large key has, among random keys, almost no chance to be among the smallest 1%
 * of a generation that has grown many times. The keys of every decision are cheap; only the queries are bounded.
 */
export const DIVERGENCE_KEEP_PPM = 20_000;
export const DIVERGENCE_KEEP_FLOOR = 512;

/** The record of run metrics (`RunMetrics.inference_divergence`): integers, counts of decisions. */
export interface InferenceDivergence {
  max_batch: number;
  /** `cuda` when the worker that played answered on the GPU (ADR-030 step 4); absent for the CPU batching. */
  device?: "cuda";
  decisions: number;
  sampled: number;
  differing: number;
  max_prior_diff_ppm: number;
  target_met: boolean;
}

interface Kept {
  key: bigint;
  query: PolicyQueryInput;
  choice: string;
}


/** A worker the measure starts for itself: the same checkpoints, never batching. */
export type SingleWorker = PolicyClient & EvaluationClient & { start(): Promise<void>; close(): Promise<void> };

/** The sampling key of a decision: the first 8 bytes of SHA-256 over the JCS of its request and draw, big endian. */
export function divergenceKey(query: PolicyQueryInput): bigint {
  const u = query.sampling.kind === "sample" ? query.sampling.u : 0;
  const digest = createHash("sha256")
    .update(jcs({ purpose: "inference-divergence/1", policy_id: query.policy_id, request_hash: query.identity.request_hash, request_seq: query.identity.request_seq, u } as unknown as Json))
    .digest();
  return digest.readBigUInt64BE(0);
}

/** How many of `decisions` the sample holds: 1% of them, at least 256 (all of them when fewer were played). */
export function sampleTarget(decisions: number): number {
  return Math.min(decisions, Math.max(DIVERGENCE_MIN_SAMPLE, Math.ceil((decisions * DIVERGENCE_SHARE_PPM) / 1_000_000)));
}

/** Collects the pair's decisions of one game process and scores a deterministic sample of them again at batch 1. */
export class DivergenceSampler {
  readonly max_batch: number;
  readonly device: "cuda" | undefined;
  #decisions = 0;
  readonly #kept: Kept[] = [];

  readonly #keep_ppm: number;
  readonly #keep_floor: number;

  /** `keep_ppm` and `keep_floor` bound how many queries are kept while the games run (a test passes others). */
  constructor(max_batch: number, keep_ppm: number = DIVERGENCE_KEEP_PPM, keep_floor: number = DIVERGENCE_KEEP_FLOOR, device?: "cuda") {
    this.max_batch = max_batch;
    this.device = device;
    this.#keep_ppm = keep_ppm;
    this.#keep_floor = keep_floor;
  }

  /** Drop the queries with the largest keys beyond the bound; the kept ones are sorted by key. */
  #prune(): void {
    this.#kept.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    this.#kept.length = Math.min(this.#kept.length, this.#bound());
  }

  #bound(): number {
    return Math.max(this.#keep_floor, Math.ceil((this.#decisions * this.#keep_ppm) / 1_000_000));
  }

  /** Observe one decision of the pair's own seat: the query it sent and the choice the batched worker answered. Never changes a byte. */
  readonly record = (query: PolicyQueryInput, choice: string): void => {
    this.#decisions += 1;
    this.#kept.push({ key: divergenceKey(query), query, choice });
    if (this.#kept.length > Math.ceil(1.25 * this.#bound())) this.#prune();
  };

  get decisions(): number {
    return this.#decisions;
  }

  /**
   * Score the sample again. `batched` is the worker that played (its `evaluate` calls are sent together, so they batch as the play did);
   * `makeSingle` makes a worker over the same checkpoints started without batching, which is started only when there is something to
   * score and closed afterwards. Null when no decision was played.
   */
  async measure(batched: EvaluationClient, makeSingle: () => SingleWorker): Promise<InferenceDivergence | null> {
    if (this.#decisions === 0) return null;
    const target = sampleTarget(this.#decisions);
    this.#prune();
    const chosen = this.#kept.slice(0, target);
    const result = (differing: number, largest: number): InferenceDivergence => ({ max_batch: this.max_batch, ...(this.device === undefined ? {} : { device: this.device }), decisions: this.#decisions, sampled: chosen.length, differing, max_prior_diff_ppm: largest, target_met: chosen.length === target });
    if (chosen.length === 0) return result(0, 0);
    const asEvaluate = (query: PolicyQueryInput): EvaluateQueryInput => ({ ...query, sampling: { kind: "evaluate" } }) as EvaluateQueryInput;
    const priors = await Promise.all(chosen.map((kept, i) => batched.evaluate(`divergence-${i}`, asEvaluate(kept.query))));
    const single = makeSingle();
    await single.start();
    try {
      let differing = 0;
      let largest = 0;
      for (const [i, kept] of chosen.entries()) {
        const again = await single.ask("divergence", kept.query);
        if (again.choice !== kept.choice) differing += 1;
        const alone = await single.evaluate("divergence", asEvaluate(kept.query));
        const together = (priors[i] as { prior_ppm: number[] }).prior_ppm;
        if (together.length !== alone.prior_ppm.length) throw new RangeError("the batched and the batch-1 prior list different candidates");
        for (const [j, ppm] of together.entries()) largest = Math.max(largest, Math.abs(ppm - (alone.prior_ppm[j] as number)));
      }
      return result(differing, largest);
    } finally {
      await single.close();
    }
  }
}

/** Read a game process's divergence record; null when it played nothing (diagnostics only: a missing or odd file is no failure). */
export function readDivergenceFile(path: string): InferenceDivergence | null {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as InferenceDivergence | null;
  } catch {
    return null;
  }
}

/** The sum of several processes' records (one `max_batch` and one device): counts add, the largest difference is the largest, the target is met by all. */
export function sumDivergence(parts: readonly InferenceDivergence[]): InferenceDivergence | null {
  const first = parts[0];
  if (first === undefined) return null;
  if (parts.some((part) => part.max_batch !== first.max_batch || part.device !== first.device)) throw new RangeError("the divergence records are of different batchings or devices");
  return {
    max_batch: first.max_batch,
    ...(first.device === undefined ? {} : { device: first.device }),
    decisions: parts.reduce((n, p) => n + p.decisions, 0),
    sampled: parts.reduce((n, p) => n + p.sampled, 0),
    differing: parts.reduce((n, p) => n + p.differing, 0),
    max_prior_diff_ppm: parts.reduce((n, p) => Math.max(n, p.max_prior_diff_ppm), 0),
    target_met: parts.every((p) => p.target_met),
  };
}

/** A game process's divergence record, beside its shard file (only for a run that batches). */
export const DIVERGENCE_FILE = "divergence.json";

/**
 * The operational flags of a GPU-inference run (ADR-030 step 4, #1458), none of them in the run config: `inference_worker_python` is the
 * Python of the unshipped `ml/gpu-env` environment (CUDA torch) the play workers run on; `force_cpu` is the rollback flag, which starts
 * them on their CPU path (the stated setting and the policy id are unchanged; counted in run metrics); `stats_dir` is where each play
 * worker leaves its device counts when its session ends (the driver's scratch directory, never a file a manifest hashes).
 */
export interface InferenceFlags {
  inference_worker_python?: string;
  force_cpu?: boolean;
  stats_dir?: string;
}

/** Why these flags do not fit the run's `generation.inference`, or null: they belong to a GPU run, which needs its worker's Python unless rolled back. */
export function inferenceFlagsProblem(inference: InferenceSettings | undefined, flags: InferenceFlags): string | null {
  const given = flags.inference_worker_python !== undefined || flags.force_cpu === true || flags.stats_dir !== undefined;
  if (inference?.device !== "cuda") return given ? "--inference-worker-python, --inference-cpu and --inference-stats-dir are for a run whose generation.inference says device: cuda" : null;
  if (flags.inference_worker_python === undefined && flags.force_cpu !== true) return "a run whose generation.inference says device: cuda needs --inference-worker-python (the ml/gpu-env Python), or --inference-cpu to roll back to the CPU path";
  return null;
}

/**
 * What a game process does about batched inference (#1195), in one object so the script has no branch for it: `inference` is the
 * run's `generation.inference` (absent: batch 1, and every method here does nothing). A GPU setting (ADR-030 step 4) also picks the play
 * workers' Python and flags (`flags`); the batch-1 control worker of the divergence measure is always the CPU worker.
 */
export class DivergenceRun {
  readonly #inference: InferenceSettings | undefined;
  readonly #flags: InferenceFlags;
  readonly #sampler: DivergenceSampler | undefined;

  constructor(inference: InferenceSettings | undefined, flags: InferenceFlags = {}) {
    this.#inference = inference;
    this.#flags = flags;
    this.#sampler = inference === undefined ? undefined : new DivergenceSampler(inference.max_batch, DIVERGENCE_KEEP_PPM, DIVERGENCE_KEEP_FLOOR, inference.device);
  }

  /** The Python a worker runs on: the GPU environment's for a GPU run's play workers (when given), `python` (the CPU one) otherwise. */
  command(batch1: boolean, python: string): string {
    if (batch1 || this.#inference?.device === undefined) return python;
    return this.#flags.inference_worker_python ?? python;
  }

  /** The worker flags of a play client; none for the batch-1 control worker (`batch1`). */
  workerArgs(batch1: boolean): string[] {
    const inference = this.#inference;
    if (inference === undefined || batch1) return [];
    const batching = ["--inference-max-batch", String(inference.max_batch)];
    if (inference.device === undefined) return batching;
    return [
      ...batching,
      "--inference-device",
      inference.device,
      ...(this.#flags.force_cpu === true ? ["--inference-force-cpu"] : []),
      ...(this.#flags.stats_dir === undefined ? [] : ["--inference-stats-dir", this.#flags.stats_dir]),
    ];
  }

  /** The `PolicyWorkerOptions` part of a play client. */
  clientOptions(batch1: boolean): { inference?: InferenceSettings } {
    return this.#inference === undefined || batch1 ? {} : { inference: this.#inference };
  }

  /** The `GenerationOptions` part that observes the pair's decisions. */
  get observer(): { on_answer?: (query: PolicyQueryInput, choice: string) => void } {
    return this.#sampler === undefined ? {} : { on_answer: this.#sampler.record };
  }

  /** A further game process leaves its record (or `null`) beside its shard file. */
  async leave(dir: string, batched: EvaluationClient, make: (batch1: boolean) => SingleWorker): Promise<void> {
    if (this.#sampler === undefined) return;
    writeFileSync(join(dir, DIVERGENCE_FILE), `${jcs((await this.#sampler.measure(batched, () => make(true))) as unknown as Json)}\n`);
  }

  /**
   * Print the play script's divergence line, if any: this process's record summed with the further game processes' (their scratch
   * directories are `scratch/1` to `scratch/<processes - 1>`; no scratch directory means no further process).
   */
  async report(batched: EvaluationClient, make: (batch1: boolean) => SingleWorker, scratch: string | undefined, processes: number, print: (line: string) => void): Promise<void> {
    if (this.#sampler === undefined) return;
    const own = await this.#sampler.measure(batched, () => make(true));
    const dirs = scratch === undefined ? [] : Array.from({ length: processes - 1 }, (_, i) => join(scratch, String(i + 1)));
    const parts = [own, ...dirs.map((dir) => readDivergenceFile(join(dir, DIVERGENCE_FILE)))].flatMap((part) => (part === null ? [] : [part]));
    const total = sumDivergence(parts);
    if (total !== null) print(jcs({ inference_divergence: total } as unknown as Json));
  }
}
