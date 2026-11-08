// Batched self-play inference (ADR-028 section 2 F2, #1195): the policy id, the config rules, the client's handshake, the sample of
// decisions scored again at batch 1, and the rule that only the training self-play clients ever batch.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { DIVERGENCE_FILE, DIVERGENCE_MIN_SAMPLE, DivergenceRun, DivergenceSampler, divergenceKey, readDivergenceFile, sampleTarget, type SingleWorker, sumDivergence } from "../src/inference-divergence.ts";
import { PolicyProtocolError, type PolicyQueryInput, policyId } from "../src/policy-protocol.ts";
import { PolicyHandshakeError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${name}.json`, "utf8")) as Record<string, unknown>;
const { query_id: _id, ...INPUT } = sample("policy-query/valid/battle") as unknown as PolicyQueryInput & { query_id: number };
const BATCH2 = { kind: "batched", max_batch: 2 } as const;

describe("the policy id of a batched run", () => {
  const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };

  it("matches the vectors computed with Python hashlib, and is the old id without the setting", () => {
    expect(policyId(pair, "sample", {}, undefined, undefined, undefined, { kind: "batched", max_batch: 4 })).toBe("policy:c1cd82a47b9ef4532ae97297f0b913625474a39e410224d496ac9b2b007c1fb0");
    expect(policyId(pair, "sample", { temperature_ppm: 500000 }, undefined, undefined, undefined, BATCH2)).toBe("policy:93259e1f4d24200d9a36b721420a844599b303fe28f205534ca388db4c15c25d");
    expect(policyId(pair, "sample")).toBe("policy:27d845e5bfc0f2d3abfec63197dd7c491189bf77578348114f087f3f532d31c2");
    expect(policyId(pair, "sample", {}, undefined, undefined, undefined, undefined)).toBe(policyId(pair, "sample"));
  });

  it("refuses a setting it cannot encode", () => {
    for (const bad of [{ kind: "batched", max_batch: 1 }, { kind: "batched", max_batch: 2.5 }, { kind: "async", max_batch: 2 }]) {
      expect(() => policyId(pair, "sample", {}, undefined, undefined, undefined, bad as never)).toThrow(PolicyProtocolError);
    }
  });
});

describe("the run config reader", () => {
  it("takes generation.inference and applies its rules", () => {
    expect(readTrainingRunConfig(sample("training-run-config/valid/inference-batched")).generation.inference).toEqual(BATCH2);
    const tiny = sample("training-run-config/valid/tiny") as { generation: Record<string, unknown> };
    const above = { ...tiny, generation: { ...tiny.generation, inference: { kind: "batched", max_batch: 3 } } };
    expect(() => readTrainingRunConfig(above)).toThrow(/cannot exceed generation.n_parallel/);
  });
});

describe("the client's handshake", () => {
  const opened: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(opened.splice(0).map((client) => client.close()));
  });
  const client = (inference: string | undefined, change: Partial<PolicyWorkerOptions> = {}): PolicyWorkerClient => {
    const made = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, "ok"],
      env: inference === undefined ? {} : { FAKE_INFERENCE: inference },
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 2,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
      ...change,
    });
    opened.push(made);
    return made;
  };

  it("sends the batching in its hello and takes a worker that states exactly it", async () => {
    const worker = client("echo", { inference: BATCH2 });
    await worker.start();
    expect((await worker.ask("g1", INPUT)).identity).toEqual(INPUT.identity);
  });

  it("takes a worker that states none from a client that asks for none", async () => {
    await client(undefined).start();
  });

  it("refuses a batching worker when it asks for batch 1, and a plain worker when it asks for batching", async () => {
    await expect(client(JSON.stringify(BATCH2)).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(JSON.stringify(BATCH2)).start()).rejects.toThrow(/states the batching .*max_batch.*not the expected none \(batch 1\)/);
    await expect(client(undefined, { inference: BATCH2 }).start()).rejects.toThrow(/states the batching none \(batch 1\), not the expected/);
  });

  it("refuses a worker that states another max_batch", async () => {
    await expect(client(JSON.stringify({ kind: "batched", max_batch: 3 }), { inference: BATCH2 }).start()).rejects.toThrow(PolicyHandshakeError);
  });

  it("refuses an inference setting that is not 2 to n_parallel", () => {
    expect(() => client(undefined, { inference: { kind: "batched", max_batch: 1 } })).toThrow(RangeError);
    expect(() => client(undefined, { inference: { kind: "batched", max_batch: 3 } })).toThrow(RangeError);
  });
});

describe("only the training self-play clients batch (source scan)", () => {
  const sources = (dir: string): string[] =>
    readdirSync(dir).flatMap((name) => {
      const path = join(dir, name);
      if (name === "node_modules" || name === "test") return [];
      if (statSync(path).isDirectory()) return sources(path);
      return path.endsWith(".ts") ? [path] : [];
    });
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const files = readdirSync(root).flatMap((pkg) => ["src", "scripts"].flatMap((sub) => (statSync(join(root, pkg)).isDirectory() ? (() => { try { return sources(join(root, pkg, sub)); } catch { return []; } })() : [])));

  it("asks a worker to batch in one place, and only the generation script uses it", () => {
    const asking = files.filter((file) => readFileSync(file, "utf8").includes("--inference-max-batch"));
    expect(asking.map((file) => file.slice(root.length))).toEqual(["showdown-bridge/src/inference-divergence.ts"]);
    const users = files.filter((file) => readFileSync(file, "utf8").includes("new DivergenceRun("));
    expect(users.map((file) => file.slice(root.length))).toEqual(["showdown-bridge/src/generation-cli.ts"]);
  });

  it("gives a client the inference option in one place", () => {
    const options = files.filter((file) => /new PolicyWorkerClient\(/.test(readFileSync(file, "utf8")) && /\binference\b/.test(readFileSync(file, "utf8")));
    expect(options.map((file) => file.slice(root.length))).toEqual(["showdown-bridge/src/generation-cli.ts"]);
  });
});

describe("the divergence sample", () => {
  const query = (n: number): PolicyQueryInput => ({ ...INPUT, identity: { ...INPUT.identity, request_seq: n }, sampling: { kind: "sample", u: n * 1000 + 7 } });
  const choiceOf = (q: PolicyQueryInput): string => (q.candidates[0] as { choice: string }).choice;
  const prior = (q: PolicyQueryInput, bump: number): number[] => q.candidates.map((_, i) => (i === 0 ? 900_000 + bump : 1000));

  /** A stand-in worker: `answers` are what it picks, `bump` what it adds to the first prior entry. */
  function worker(pick: (q: PolicyQueryInput) => string, bump: number): SingleWorker & { started: number; closed: number } {
    const made = {
      started: 0,
      closed: 0,
      start: () => {
        made.started += 1;
        return Promise.resolve();
      },
      close: () => {
        made.closed += 1;
        return Promise.resolve();
      },
      ask: (_battle: string, q: PolicyQueryInput) => Promise.resolve({ schema_version: 1, identity: q.identity, choice: pick(q), source_policy_id: q.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" }),
      evaluate: (_battle: string, q: PolicyQueryInput) => Promise.resolve({ identity: q.identity, prior_ppm: prior(q, bump), value_ppm: 500000 }),
    };
    return made as unknown as SingleWorker & { started: number; closed: number };
  }

  it("keys a decision by its own request and draw, whatever the order", () => {
    expect(divergenceKey(query(1))).toBe(divergenceKey(query(1)));
    expect(divergenceKey(query(1))).not.toBe(divergenceKey(query(2)));
    expect(divergenceKey({ ...query(1), sampling: { kind: "argmax" } })).not.toBe(divergenceKey(query(1)));
  });

  it("targets 1 percent of the decisions and at least 256, or all of them when fewer were played", () => {
    expect(sampleTarget(100)).toBe(100);
    expect(sampleTarget(10_000)).toBe(DIVERGENCE_MIN_SAMPLE);
    expect(sampleTarget(100_000)).toBe(1000);
    expect(sampleTarget(25_601)).toBe(257);
  });

  it("scores the kept decisions again at batch 1 and counts the choices that differ and the largest prior difference", async () => {
    const sampler = new DivergenceSampler(2, 1_000_000);
    const queries = Array.from({ length: 10 }, (_, i) => query(i + 1));
    for (const q of queries) sampler.record(q, choiceOf(q));
    expect(sampler.decisions).toBe(10);
    // Batch 1 picks another candidate for two decisions, and its first prior is 37 ppm higher than the batched one.
    const other = (q: PolicyQueryInput): string => ((q.candidates[1] ?? q.candidates[0]) as { choice: string }).choice;
    const differs = new Set([3, 7].map((n) => query(n).identity.request_seq));
    const single = worker((q) => (differs.has(q.identity.request_seq) ? other(q) : choiceOf(q)), 37);
    const found = await sampler.measure(worker(choiceOf, 0), () => single);
    expect(found).toEqual({ max_batch: 2, decisions: 10, sampled: 10, differing: 2, max_prior_diff_ppm: 37, target_met: true });
    expect(single.started).toBe(1);
    expect(single.closed).toBe(1);
  });

  it("takes the smallest keys when more were kept than the target, and says when it kept fewer", async () => {
    const few = new DivergenceSampler(2, 0);
    few.record(query(1), choiceOf(query(1)));
    const quiet = worker(choiceOf, 0);
    expect(await few.measure(worker(choiceOf, 0), () => quiet)).toEqual({ max_batch: 2, decisions: 1, sampled: 0, differing: 0, max_prior_diff_ppm: 0, target_met: false });
    expect(quiet.started).toBe(0); // nothing to score: no worker is started
    expect(await new DivergenceSampler(2).measure(worker(choiceOf, 0), () => quiet)).toBeNull();
    const many = new DivergenceSampler(2, 1_000_000);
    for (let n = 1; n <= 300; n++) many.record(query(n), choiceOf(query(n)));
    const found = await many.measure(worker(choiceOf, 0), () => worker(choiceOf, 0));
    expect(found?.sampled).toBe(256);
    expect(found?.target_met).toBe(true);
  });

  it("refuses priors that list other candidates", async () => {
    const sampler = new DivergenceSampler(2, 1_000_000);
    sampler.record(query(1), choiceOf(query(1)));
    const short = worker(choiceOf, 0);
    (short as unknown as { evaluate: unknown }).evaluate = (_b: string, q: PolicyQueryInput) => Promise.resolve({ identity: q.identity, prior_ppm: [1_000_000], value_ppm: 0 });
    await expect(sampler.measure(worker(choiceOf, 0), () => short)).rejects.toThrow(/different candidates/);
    expect(short.closed).toBe(1);
  });

  it("sums the records of several processes", () => {
    const part = { max_batch: 2, decisions: 10, sampled: 4, differing: 1, max_prior_diff_ppm: 5, target_met: true };
    expect(sumDivergence([])).toBeNull();
    expect(sumDivergence([part, { ...part, differing: 0, max_prior_diff_ppm: 9, target_met: false }])).toEqual({ max_batch: 2, decisions: 20, sampled: 8, differing: 1, max_prior_diff_ppm: 9, target_met: false });
    expect(() => sumDivergence([part, { ...part, max_batch: 3 }])).toThrow(/different batchings/);
  });

  it("reads a process's record, and nothing from a missing or odd file", () => {
    const dir = mkdtempSync(join(tmpdir(), "areneon-divergence-"));
    try {
      const part = { max_batch: 2, decisions: 10, sampled: 4, differing: 1, max_prior_diff_ppm: 5, target_met: true };
      writeFileSync(join(dir, "a.json"), `${JSON.stringify(part)}\n`);
      writeFileSync(join(dir, "null.json"), "null\n");
      writeFileSync(join(dir, "odd.json"), "{");
      expect(readDivergenceFile(join(dir, "a.json"))).toEqual(part);
      expect(readDivergenceFile(join(dir, "null.json"))).toBeNull();
      expect(readDivergenceFile(join(dir, "odd.json"))).toBeNull();
      expect(readDivergenceFile(join(dir, "missing.json"))).toBeNull();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("what a game process does about batching (DivergenceRun)", () => {
  const stand = { evaluate: (_b: string, q: PolicyQueryInput) => Promise.resolve({ identity: q.identity, prior_ppm: q.candidates.map(() => 1000) as [number, ...number[]], value_ppm: 0 }) };
  const single = (): SingleWorker =>
    ({ start: () => Promise.resolve(), close: () => Promise.resolve(), ask: (_b: string, q: PolicyQueryInput) => Promise.resolve({ choice: (q.candidates[0] as { choice: string }).choice }), evaluate: stand.evaluate }) as unknown as SingleWorker;

  it("does nothing for a run that does not batch", async () => {
    const none = new DivergenceRun(undefined);
    expect(none.workerArgs(false)).toEqual([]);
    expect(none.clientOptions(false)).toEqual({});
    expect(none.observer).toEqual({});
    await none.leave("/nonexistent", stand, single);
    const printed: string[] = [];
    await none.report(stand, single, undefined, 1, (line) => void printed.push(line));
    expect(printed).toEqual([]);
  });

  it("asks the play clients, and not the batch-1 control worker, to batch", () => {
    const run = new DivergenceRun(BATCH2);
    expect(run.workerArgs(false)).toEqual(["--inference-max-batch", "2"]);
    expect(run.workerArgs(true)).toEqual([]);
    expect(run.clientOptions(false)).toEqual({ inference: BATCH2 });
    expect(run.clientOptions(true)).toEqual({});
    expect(typeof run.observer.on_answer).toBe("function");
  });

  it("leaves its record for the process that started it, and sums it with its own", async () => {
    const dir = mkdtempSync(join(tmpdir(), "areneon-divergence-run-"));
    try {
      const mine = new DivergenceRun(BATCH2);
      const printed: string[] = [];
      const print = (line: string): void => void printed.push(line);
      await mine.report(stand, single, undefined, 1, print); // nothing played, nothing left
      expect(printed).toEqual([]);
      const other = new DivergenceRun(BATCH2);
      other.observer.on_answer?.(INPUT as PolicyQueryInput, (INPUT.candidates[0] as { choice: string }).choice);
      const scratch = join(dir, "scratch");
      mkdirSync(join(scratch, "1"), { recursive: true });
      await other.leave(join(scratch, "1"), stand, single);
      expect(readDivergenceFile(join(scratch, "1", DIVERGENCE_FILE))).toMatchObject({ max_batch: 2, decisions: 1 });
      await mine.report(stand, single, scratch, 2, print);
      expect(JSON.parse(printed[0] as string)).toEqual({ inference_divergence: { max_batch: 2, decisions: 1, sampled: 0, differing: 0, max_prior_diff_ppm: 0, target_met: false } });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
