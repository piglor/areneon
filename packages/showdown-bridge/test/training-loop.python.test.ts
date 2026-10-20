// The training loop end to end on CPU (ADR-010 §1, Wave 4 gate): the Python driver plays A1 generations through
// Node and the real policy worker, trains, freezes and commits. It needs the project's Python environment
// (`make setup`, uv sync); without it the file is skipped, unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript
// CI job sets it). The config is the tiny sample of the contract corpus: a few games, two generations, a tiny network.
import { createHash } from "node:crypto";
import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { jcs } from "../src/canonical-json.ts";
import { runGenerationCli } from "../src/generation-cli.ts";
import { GOLDEN_DIR } from "../src/golden.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(GOLDEN_DIR, "engine-data.json");
const TINY = join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json");

interface RunSummary {
  resumed_from: number | null;
  discarded: string[];
  generation: number;
  checkpoints: { battle: string; preview: string };
  learner_state_hash: string;
  policy_id: string;
  played: number[];
}

const args = (config: string, runDir: string, generations: number): string[] => [
  "-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA,
  "--generations", String(generations), "--node", process.execPath, "--worker-python", PYTHON,
];
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };
const run = (config: string, runDir: string, generations: number): RunSummary =>
  JSON.parse(execFileSync(PYTHON, args(config, runDir, generations), { encoding: "utf8", env })) as RunSummary;
const read = (runDir: string, ...path: string[]): string => readFileSync(join(runDir, ...path), "utf8");
interface Manifest {
  checkpoints: Record<string, string>;
  training_set_hash: string | null;
}
interface Doc {
  role: string;
  parent: string;
  training_set_hashes: string[];
  behaviour_policy_id: string;
  learner_ready: boolean;
  episode_hashes: string[];
  provenance: { behaviour_policy_id: string };
}
// The files are the driver's own JSON; the assertions below name the fields they read.
const json = <T = Doc>(runDir: string, ...path: string[]): T => JSON.parse(read(runDir, ...path)) as T;
const manifest = (runDir: string, g: number): Manifest => json<Manifest>(runDir, "generations", gen(g), "manifest.json");
const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;

/** Start the driver in its own process group and SIGKILL the whole group once `ready` holds. */
async function killWhen(config: string, runDir: string, generations: number, ready: () => boolean): Promise<void> {
  const child: ChildProcess = spawn(PYTHON, args(config, runDir, generations), { env, detached: true, stdio: "ignore" });
  const exited = new Promise<number | null>((resolve) => child.once("close", (code) => resolve(code)));
  const deadline = Date.now() + 240_000;
  while (!ready()) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error("the driver ended or stalled before the kill point");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  process.kill(-(child.pid as number), "SIGKILL");
  expect(await exited).toBeNull();
}

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the training loop, tiny and on CPU", () => {
  let dir: string;
  let uninterrupted: string;
  let killed: string;
  let first: RunSummary;
  let resumed: RunSummary;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "areneon-loop-"));
    uninterrupted = join(dir, "uninterrupted");
    killed = join(dir, "killed");
    first = run(TINY, uninterrupted, 2);
    // A real kill: SIGKILL the driver, its Node child and the worker as soon as generation 2 is under way.
    await killWhen(TINY, killed, 2, () => existsSync(join(killed, "generations", `.${gen(2)}.partial`)));
    resumed = run(TINY, killed, 2);
  }, 600_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays, trains and commits two generations, each with records, a learner state and a manifest", () => {
    expect(first).toMatchObject({ resumed_from: null, discarded: [], generation: 2, played: [1, 2] });
    for (const g of [0, 1, 2]) {
      const files = readdirSync(join(uninterrupted, "generations", gen(g)));
      expect(files).toEqual(expect.arrayContaining([`learner-${g}.safetensors`, `learner-${g}.json`, "manifest.json", "records", "team-set.json"]));
    }
    expect(readdirSync(join(uninterrupted, "generations")).filter((n) => n.endsWith(".partial"))).toEqual([]);
    const records = ["battle", "preview"].map((role) => json(uninterrupted, "generations", gen(2), "records", `${role}.json`));
    for (const record of records) {
      expect(record).toMatchObject({ schema_version: 1, lineage_id: "tiny-main", data_class: "synthetic", learner_state_hash: first.learner_state_hash });
      expect(record.parent).toBe(manifest(uninterrupted, 1).checkpoints[record.role]);
      expect(record.training_set_hashes).toEqual([manifest(uninterrupted, 2).training_set_hash]);
    }
  });

  it("trains each generation on the frozen pair's own seats only: the baseline's Episodes are kept but never loaded", () => {
    for (const g of [1, 2]) {
      const set = json(uninterrupted, "generations", gen(g), "training-set.json");
      const summary = json(uninterrupted, "generations", gen(g), "generation.json");
      const pair = manifest(uninterrupted, g - 1).checkpoints;
      expect(set.behaviour_policy_id).toMatch(/^policy:[0-9a-f]{64}$/);
      expect(set.learner_ready).toBe(true);
      const behaviour = (hash: string): string => json(uninterrupted, "generations", gen(g), "episodes", `${hash}.json`).provenance.behaviour_policy_id;
      const all = summary.episode_hashes;
      expect(all).toHaveLength(12); // 4 self-play games x 2 seats + 2 baseline games x 2 seats
      expect(new Set(all.map(behaviour))).toEqual(new Set([set.behaviour_policy_id, "random-legal@tiny"]));
      expect(set.episode_hashes).toEqual(all.filter((hash) => behaviour(hash) === set.behaviour_policy_id).sort());
      expect(all.filter((hash) => behaviour(hash) === "random-legal@tiny")).toHaveLength(2);
      // ADR-010 §2: the behaviour policy id is the hash of the frozen pair and the inference settings.
      const identity = { preview: pair.preview as string, battle: pair.battle as string, mode: "sample", temperature: 1, precision: "float32" };
      expect(set.behaviour_policy_id).toBe(`policy:${createHash("sha256").update(jcs(identity)).digest("hex")}`);
    }
  });

  it("reproduces: the same config gives the same generation 1, byte for byte, in an independent run", () => {
    expect(read(killed, "generations", gen(1), "manifest.json")).toBe(read(uninterrupted, "generations", gen(1), "manifest.json"));
  });

  it("resumes a SIGKILLed run from its last complete generation and ends with the same learner-state hash", () => {
    expect(resumed).toMatchObject({ resumed_from: 1, discarded: [`.${gen(2)}.partial`], generation: 2, played: [2] });
    expect(resumed.learner_state_hash).toBe(first.learner_state_hash);
    expect(resumed.checkpoints).toEqual(first.checkpoints);
    expect(resumed.policy_id).toBe(first.policy_id);
    expect(read(killed, "generations", gen(2), "manifest.json")).toBe(read(uninterrupted, "generations", gen(2), "manifest.json"));
    for (const role of ["battle", "preview"]) {
      expect(read(killed, "generations", gen(2), "records", `${role}.json`)).toBe(read(uninterrupted, "generations", gen(2), "records", `${role}.json`));
    }
  });

  it("is a no-op when the target is already committed, and refuses another config in the same directory", () => {
    expect(run(TINY, uninterrupted, 2)).toMatchObject({ resumed_from: 2, played: [], generation: 2, learner_state_hash: first.learner_state_hash });
    const other = JSON.parse(readFileSync(TINY, "utf8")) as { ppo: { epochs: number } };
    other.ppo.epochs += 1;
    const file = join(dir, "other.json");
    writeFileSync(file, JSON.stringify(other));
    const refused = spawnSync(PYTHON, args(file, uninterrupted, 2), { encoding: "utf8", env });
    expect(refused.status).toBe(3);
    expect(refused.stderr).toContain("another TrainingRunConfig");
  }, 120_000); // each call starts Python and imports torch: seconds on a loaded CI runner, where the default 5 s was exceeded

  it("leaves records the C1/C6 checker reads: the first pair is in the second pair's ancestry", () => {
    const out = join(dir, "provenance");
    execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "provenance", "--run-dir", uninterrupted, "--out", out], { env });
    const [parent, child] = [1, 2].map((g) => manifest(uninterrupted, g).checkpoints.battle as string) as [string, string];
    const classify = (candidate: string, opponent: string): { label: string } =>
      JSON.parse(
        execFileSync(PYTHON, ["-I", "-m", "areneon_ml.checkpoint", "classify", "--records", join(out, "records"), "--sources", join(out, "sources.json"), "--candidate", candidate, "--opponent", opponent], { encoding: "utf8", env }),
      ) as { label: string };
    expect(classify(child, parent).label).toBe("seen");
  }, 120_000);
});

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("play-generation with the real worker, in this process", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-play-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays one generation of an untrained pair, writes its files and prints one canonical line", async () => {
    const tiny = JSON.parse(readFileSync(TINY, "utf8")) as { model: unknown };
    const modelFile = join(dir, "model.json");
    writeFileSync(modelFile, JSON.stringify(tiny.model));
    const init = (role: string, seed: number): { checkpoint_id: string; encoder_version: string } =>
      JSON.parse(
        execFileSync(PYTHON, ["-I", "-m", "areneon_ml.checkpoint", "init", "--role", role, "--seed", String(seed), "--engine-data", DATA, "--model-config", modelFile, "--run-dir", dir], { encoding: "utf8", env }),
      ) as { checkpoint_id: string; encoder_version: string };
    const preview = init("preview", 1);
    const battle = init("battle", 2);
    const out = join(dir, "out");
    const lines: string[] = [];
    const code = await runGenerationCli(
      [
        "--config", TINY, "--generation", "1", "--policy-id", `policy:${"e".repeat(64)}`, "--preview", preview.checkpoint_id, "--battle", battle.checkpoint_id,
        "--run-dir", dir, "--engine-data", DATA, "--encoder-version", preview.encoder_version, "--worker-python", PYTHON, "--out", out,
      ],
      (line) => lines.push(line),
    );
    expect(code).toBe(0);
    expect(lines).toHaveLength(1);
    const printed = JSON.parse(lines[0] as string) as { generation: number; run_config_hash: string; training_set_hash: string; examples: number };
    const summary = JSON.parse(readFileSync(join(out, "generation.json"), "utf8")) as { training_set_hash: string; examples: number; forfeits: number; games: unknown[] };
    expect(printed).toMatchObject({ generation: 1, training_set_hash: summary.training_set_hash, examples: summary.examples });
    expect(printed.run_config_hash).toBe(createHash("sha256").update(jcs(JSON.parse(readFileSync(TINY, "utf8")))).digest("hex"));
    expect(summary.games).toHaveLength(6);
    expect(summary.forfeits).toBe(0);
    expect(readdirSync(join(out, "episodes"))).toHaveLength(12);
  }, 180_000);
});
