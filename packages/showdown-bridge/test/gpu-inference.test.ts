// GPU policy inference for training self-play (ADR-030 step 4, #1458): the device setting of `generation.inference` in the policy id,
// the config reader and the handshake, the refusal of a GPU worker by every batch-1 (evaluation) client, the worker command line of a
// game process (the GPU worker's Python, the rollback flag, the stats directory) and the divergence record that names the device.
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { DivergenceRun, inferenceFlagsProblem, sumDivergence } from "../src/inference-divergence.ts";
import { forwardedFlags } from "../src/generation-cli.ts";
import { PolicyProtocolError, type PolicyQueryInput, policyId } from "../src/policy-protocol.ts";
import { PolicyHandshakeError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${name}.json`, "utf8")) as Record<string, unknown>;
const { query_id: _id, ...INPUT } = sample("policy-query/valid/battle") as unknown as PolicyQueryInput & { query_id: number };
const GPU2 = { kind: "batched", max_batch: 2, device: "cuda" } as const;
const CPU2 = { kind: "batched", max_batch: 2 } as const;

describe("the policy id of a GPU-inference run", () => {
  const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };

  it("matches the vectors computed with Python hashlib, and the device makes another id than the CPU batching", () => {
    expect(policyId(pair, "sample", {}, undefined, undefined, undefined, { kind: "batched", max_batch: 4, device: "cuda" })).toBe("policy:32ffe17f08e8e31c890a6c461a811d6928f7168bf3fec8bb76f04baefca252fb");
    expect(policyId(pair, "sample", { temperature_ppm: 500000 }, undefined, undefined, undefined, GPU2)).toBe("policy:a12c36fe31f5fcff65500cfb3696c3d842c23c94ce748c7567d5323af9fcd721");
    // The CPU batching keeps the id #1195 pinned: the device enters the preimage only when it is the GPU.
    expect(policyId(pair, "sample", {}, undefined, undefined, undefined, { kind: "batched", max_batch: 4 })).toBe("policy:c1cd82a47b9ef4532ae97297f0b913625474a39e410224d496ac9b2b007c1fb0");
    expect(policyId(pair, "sample", {}, undefined, undefined, undefined, GPU2)).not.toBe(policyId(pair, "sample", {}, undefined, undefined, undefined, CPU2));
  });

  it("refuses a device it cannot encode", () => {
    for (const bad of [{ ...CPU2, device: "cpu" }, { ...CPU2, device: "rocm" }, { ...CPU2, device: "" }]) {
      expect(() => policyId(pair, "sample", {}, undefined, undefined, undefined, bad as never)).toThrow(PolicyProtocolError);
    }
  });
});

describe("the run config reader", () => {
  it("takes generation.inference.device", () => {
    const tiny = sample("training-run-config/valid/tiny") as { generation: Record<string, unknown> };
    const gpu = { ...tiny, generation: { ...tiny.generation, inference: GPU2 } };
    expect(readTrainingRunConfig(gpu).generation.inference).toEqual(GPU2);
    expect(readTrainingRunConfig(sample("training-run-config/valid/inference-gpu")).generation.inference).toEqual(GPU2);
  });

  it("refuses a device other than cuda (the CPU batching states none)", () => {
    const tiny = sample("training-run-config/valid/tiny") as { generation: Record<string, unknown> };
    for (const device of ["cpu", "mps"]) {
      expect(() => readTrainingRunConfig({ ...tiny, generation: { ...tiny.generation, inference: { ...CPU2, device } } })).toThrow();
    }
  });
});

describe("the handshake of a GPU worker", () => {
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

  it("a training client that asks for the GPU sends the device and takes a worker that states it", async () => {
    const worker = client("echo", { inference: GPU2 });
    await worker.start();
    expect((await worker.ask("g1", INPUT)).identity).toEqual(INPUT.identity);
  });

  it("every evaluation client (batch 1) refuses a GPU worker at the handshake", async () => {
    await expect(client(JSON.stringify(GPU2)).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(JSON.stringify(GPU2)).start()).rejects.toThrow(/states the batching .*cuda.*not the expected none \(batch 1\)/);
  });

  it("a CPU-batching client refuses a GPU worker, and a GPU client refuses a CPU-batching or a plain worker", async () => {
    await expect(client(JSON.stringify(GPU2), { inference: CPU2 }).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(JSON.stringify(CPU2), { inference: GPU2 }).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(undefined, { inference: GPU2 }).start()).rejects.toThrow(PolicyHandshakeError);
  });

  it("refuses an unknown device in its own options", () => {
    expect(() => client(undefined, { inference: { ...CPU2, device: "rocm" } as never })).toThrow(RangeError);
  });
});

describe("the worker command line of a GPU-inference game process", () => {
  const python = "/cpu/python";
  const gpuPython = "/gpu-env/python";

  it("starts the play workers with the GPU worker's Python and the device, and the batch-1 control worker on the CPU", () => {
    const run = new DivergenceRun(GPU2, { inference_worker_python: gpuPython, stats_dir: "/stats" });
    expect(run.command(false, python)).toBe(gpuPython);
    expect(run.workerArgs(false)).toEqual(["--inference-max-batch", "2", "--inference-device", "cuda", "--inference-stats-dir", "/stats"]);
    expect(run.clientOptions(false)).toEqual({ inference: GPU2 });
    expect(run.command(true, python)).toBe(python);
    expect(run.workerArgs(true)).toEqual([]);
    expect(run.clientOptions(true)).toEqual({});
  });

  it("the rollback flag keeps the stated setting and starts the worker on its CPU path, from either Python", () => {
    const forced = new DivergenceRun(GPU2, { force_cpu: true });
    expect(forced.command(false, python)).toBe(python);
    expect(forced.workerArgs(false)).toEqual(["--inference-max-batch", "2", "--inference-device", "cuda", "--inference-force-cpu"]);
    expect(forced.clientOptions(false)).toEqual({ inference: GPU2 });
    expect(new DivergenceRun(GPU2, { force_cpu: true, inference_worker_python: gpuPython }).command(false, python)).toBe(gpuPython);
  });

  it("a CPU-batching run and a batch-1 run are what they were", () => {
    const cpu = new DivergenceRun(CPU2);
    expect(cpu.command(false, python)).toBe(python);
    expect(cpu.workerArgs(false)).toEqual(["--inference-max-batch", "2"]);
    const none = new DivergenceRun(undefined);
    expect(none.command(false, python)).toBe(python);
    expect(none.workerArgs(false)).toEqual([]);
  });

  it("refuses the GPU flags where the config does not ask for the GPU, and a GPU run without its worker's Python", () => {
    expect(inferenceFlagsProblem(GPU2, { inference_worker_python: gpuPython })).toBeNull();
    expect(inferenceFlagsProblem(GPU2, { force_cpu: true })).toBeNull();
    expect(inferenceFlagsProblem(GPU2, {})).toMatch(/--inference-worker-python/);
    expect(inferenceFlagsProblem(CPU2, { inference_worker_python: gpuPython })).toMatch(/device/);
    expect(inferenceFlagsProblem(undefined, { force_cpu: true })).toMatch(/device/);
    expect(inferenceFlagsProblem(undefined, { stats_dir: "/stats" })).toMatch(/device/);
    expect(inferenceFlagsProblem(undefined, {})).toBeNull();
    expect(inferenceFlagsProblem(CPU2, {})).toBeNull();
  });

  it("forwards the rollback flag and the GPU worker's Python to the further game processes", () => {
    const flags = forwardedFlags({ config: "c.json", "inference-cpu": true, "inference-worker-python": gpuPython, "inference-stats-dir": "/stats", out: "o" });
    expect(flags).toEqual(["--config", "c.json", "--inference-cpu", "--inference-worker-python", gpuPython, "--inference-stats-dir", "/stats"]);
    expect(forwardedFlags({ config: "c.json", "inference-cpu": false })).toEqual(["--config", "c.json"]);
  });
});

describe("the divergence record of a GPU run", () => {
  it("names the device, so the share of differing choices is the GPU worker's against the CPU at batch 1", async () => {
    const dir = mkdtempSync(join(tmpdir(), "areneon-gpu-divergence-"));
    try {
      const run = new DivergenceRun(GPU2, { inference_worker_python: "/gpu" });
      run.observer.on_answer?.(INPUT as PolicyQueryInput, (INPUT.candidates[0] as { choice: string }).choice);
      const stand = { evaluate: (_b: string, q: PolicyQueryInput) => Promise.resolve({ identity: q.identity, prior_ppm: q.candidates.map(() => 1000) as [number, ...number[]], value_ppm: 0 }) };
      const single = () =>
        ({ start: () => Promise.resolve(), close: () => Promise.resolve(), ask: (_b: string, q: PolicyQueryInput) => Promise.resolve({ choice: (q.candidates[0] as { choice: string }).choice }), evaluate: stand.evaluate }) as never;
      const printed: string[] = [];
      await run.report(stand, single, undefined, 1, (line) => void printed.push(line));
      expect(JSON.parse(printed[0] as string)).toEqual({ inference_divergence: { max_batch: 2, device: "cuda", decisions: 1, sampled: 1, differing: 0, max_prior_diff_ppm: 0, target_met: true } });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("refuses to sum records of different devices", () => {
    const part = { max_batch: 2, decisions: 10, sampled: 4, differing: 1, max_prior_diff_ppm: 5, target_met: true };
    expect(sumDivergence([{ ...part, device: "cuda" }, { ...part, device: "cuda" }])).toMatchObject({ device: "cuda", decisions: 20 });
    expect(() => sumDivergence([part, { ...part, device: "cuda" }])).toThrow(/different/);
  });
});

describe("only the training self-play clients ask for the GPU (source scan)", () => {
  const sources = (dir: string): string[] =>
    readdirSync(dir).flatMap((name) => {
      const path = join(dir, name);
      if (name === "node_modules" || name === "test") return [];
      if (statSync(path).isDirectory()) return sources(path);
      return path.endsWith(".ts") ? [path] : [];
    });
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const files = readdirSync(root).flatMap((pkg) => ["src", "scripts"].flatMap((sub) => (statSync(join(root, pkg)).isDirectory() ? (() => { try { return sources(join(root, pkg, sub)); } catch { return []; } })() : [])));

  it("names the GPU device for a worker in one place", () => {
    const asking = files.filter((file) => readFileSync(file, "utf8").includes("--inference-device"));
    expect(asking.map((file) => file.slice(root.length))).toEqual(["showdown-bridge/src/inference-divergence.ts"]);
  });

  it("reads the GPU worker's Python only in the generation script", () => {
    const reading = files.filter((file) => readFileSync(file, "utf8").includes('"inference-worker-python"'));
    expect(reading.map((file) => file.slice(root.length))).toEqual(["showdown-bridge/src/generation-cli.ts"]);
  });
});
