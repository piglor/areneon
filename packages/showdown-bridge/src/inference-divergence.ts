// How far batched self-play inference strayed from batch 1 (ADR-028 section 2 F2, #1195). A diagnostic of the training play phase:
// nothing here is read by the loop, the loader or the learner, written to a hashed file, or part of any identity.
//
// The pair's own seats report each decision they play (`on_answer`). A deterministic part of them is kept (a hash of the decision's
// own sampling draw and request, so the same decisions are kept whatever the order the games finished in), and when the games are
// done the kept queries are scored again at batch 1 with each decision's own `u`: the share whose choice differs from the played
// one, and the largest difference between the batched and the batch-1 prior (the policy's probabilities at temperature 1, in ppm).

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type Json, jcs } from "./canonical-json.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import type { EvaluateQueryInput, InferenceSettings, PolicyQueryInput } from "./policy-protocol.ts";

/** The share of the decisions the sample must reach (ppm), and the least number of decisions when that many were played. */
export const DIVERGENCE_SHARE_PPM = 10_000;
export const DIVERGENCE_MIN_SAMPLE = 256;
/** The share kept while the games run (ppm): a little over the target, so the final sample (the smallest keys) is a subset of what was kept. */
export const DIVERGENCE_KEEP_PPM = 15_000;

/** The record of run metrics (`RunMetrics.inference_divergence`): integers, counts of decisions. */
export interface InferenceDivergence {
  max_batch: number;
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

const TWO_64 = 2n ** 64n;

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

const below = (key: bigint, ppm: number): boolean => key * 1_000_000n < BigInt(ppm) * TWO_64;

/** How many of `decisions` the sample holds: 1% of them, at least 256 (all of them when fewer were played). */
export function sampleTarget(decisions: number): number {
  return Math.min(decisions, Math.max(DIVERGENCE_MIN_SAMPLE, Math.ceil((decisions * DIVERGENCE_SHARE_PPM) / 1_000_000)));
}

/** Collects the pair's decisions of one game process and scores a deterministic sample of them again at batch 1. */
export class DivergenceSampler {
  readonly max_batch: number;
  #decisions = 0;
  readonly #kept: Kept[] = [];

  readonly #keep_ppm: number;

  /** `keep_ppm` is the share kept while the games run; the default is the one the sample size needs (a test passes a larger one). */
  constructor(max_batch: number, keep_ppm: number = DIVERGENCE_KEEP_PPM) {
    this.max_batch = max_batch;
    this.#keep_ppm = keep_ppm;
  }

  /** Observe one decision of the pair's own seat: the query it sent and the choice the batched worker answered. Never changes a byte. */
  readonly record = (query: PolicyQueryInput, choice: string): void => {
    this.#decisions += 1;
    const key = divergenceKey(query);
    if (below(key, this.#keep_ppm)) this.#kept.push({ key, query, choice });
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
    const chosen = [...this.#kept].sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0)).slice(0, target);
    const result = (differing: number, largest: number): InferenceDivergence => ({ max_batch: this.max_batch, decisions: this.#decisions, sampled: chosen.length, differing, max_prior_diff_ppm: largest, target_met: chosen.length === target });
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

/** The sum of several processes' records (one `max_batch`): counts add, the largest difference is the largest, the target is met by all. */
export function sumDivergence(parts: readonly InferenceDivergence[]): InferenceDivergence | null {
  const first = parts[0];
  if (first === undefined) return null;
  if (parts.some((part) => part.max_batch !== first.max_batch)) throw new RangeError("the divergence records are of different batchings");
  return {
    max_batch: first.max_batch,
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
 * What a game process does about batched inference (#1195), in one object so the script has no branch for it: `inference` is the
 * run's `generation.inference` (absent: batch 1, and every method here does nothing).
 */
export class DivergenceRun {
  readonly #inference: InferenceSettings | undefined;
  readonly #sampler: DivergenceSampler | undefined;

  constructor(inference: InferenceSettings | undefined) {
    this.#inference = inference;
    this.#sampler = inference === undefined ? undefined : new DivergenceSampler(inference.max_batch);
  }

  /** The worker flag of a play client; none for the batch-1 control worker (`batch1`). */
  workerArgs(batch1: boolean): string[] {
    return this.#inference === undefined || batch1 ? [] : ["--inference-max-batch", String(this.#inference.max_batch)];
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
