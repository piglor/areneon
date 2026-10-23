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

const args = (config: string, runDir: string, generations: number, ...extra: string[]): string[] => [
  "-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA,
  "--generations", String(generations), "--node", process.execPath, "--worker-python", PYTHON, ...extra,
];
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };
const run = (config: string, runDir: string, generations: number, ...extra: string[]): RunSummary =>
  JSON.parse(execFileSync(PYTHON, args(config, runDir, generations, ...extra), { encoding: "utf8", env })) as RunSummary;
const read = (runDir: string, ...path: string[]): string => readFileSync(join(runDir, ...path), "utf8");
interface Manifest {
  checkpoints: Record<string, string>;
  training_set_hash: string | null;
  files: Record<string, string>;
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

/** Command lines of the live processes that mention `runDir` (Linux): the Node play script and the policy worker of that run. */
function liveProcesses(runDir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync("/proc")) {
    if (!/^\d+$/.test(entry)) continue;
    try {
      const command = readFileSync(`/proc/${entry}/cmdline`, "utf8").replaceAll("\0", " ");
      if (command.includes(runDir) && /play-generation\.ts|areneon_ml\.worker/.test(command)) found.push(command);
    } catch {
      // the process ended while it was being read
    }
  }
  return found;
}

/** Start the driver in its own process group and SIGKILL the whole group once `ready` holds. */
async function killWhen(config: string, runDir: string, generations: number, ready: () => boolean): Promise<void> {
  const child: ChildProcess = spawn(PYTHON, args(config, runDir, generations), { env, detached: true, stdio: "ignore" });
  const exited = new Promise<number | null>((resolve) => child.once("close", (code) => resolve(code)));
  const deadline = Date.now() + 240_000;
  while (!ready()) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error("the driver ended or stalled before the kill point");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  process.kill(-(child.pid as number), "SIGKILL");
  expect(await exited).toBeNull();
}

const blank = (): { dir: string; resumed: RunSummary } => ({ dir: "", resumed: undefined as unknown as RunSummary });
const partial = (runDir: string, ...path: string[]): string => join(runDir, "generations", `.${gen(2)}.partial`, ...path);

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the training loop, tiny and on CPU", () => {
  let dir: string;
  let uninterrupted: string;
  let first: RunSummary;
  let sharded: string;
  let shardedSummary: RunSummary;
  // Three real SIGKILLs of the driver's process group in generation 2, each followed by a resume:
  //  play: the simulator side is alive (Node and the worker), the partial may still be empty;
  //  write: Node has played and is writing its files (generation.json is written last), the worker is still alive;
  //  seal: the driver has trained (train-report.json exists) and is sealing, with no manifest yet and no rename.
  type Point = { dir: string; resumed: RunSummary };
  const kills: Record<"play" | "write" | "seal", Point> = { play: blank(), write: blank(), seal: blank() };

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "areneon-loop-"));
    uninterrupted = join(dir, "uninterrupted");
    first = run(TINY, uninterrupted, 2);
    // The same run with two policy workers in each of two game processes (#957): operational flags, not config.
    sharded = join(dir, "sharded");
    shardedSummary = run(TINY, sharded, 2, "--policy-workers", "2", "--game-processes", "2");
    kills.play.dir = join(dir, "killed-playing");
    await killWhen(TINY, kills.play.dir, 2, () => existsSync(partial(kills.play.dir)) && liveProcesses(kills.play.dir).length >= 2);
    expect(liveProcesses(kills.play.dir)).toEqual([]); // the whole group went down with the driver
    expect(existsSync(partial(kills.play.dir, "manifest.json"))).toBe(false);
    kills.play.resumed = run(TINY, kills.play.dir, 2);
    kills.write.dir = join(dir, "killed-writing");
    await killWhen(TINY, kills.write.dir, 2, () => existsSync(partial(kills.write.dir, "generation.json")) && liveProcesses(kills.write.dir).length >= 1);
    expect(existsSync(partial(kills.write.dir, "manifest.json"))).toBe(false);
    kills.write.resumed = run(TINY, kills.write.dir, 2);
    // The seal window is short: if the signal lands after the commit, try again in a fresh directory.
    for (let attempt = 1; ; attempt++) {
      kills.seal.dir = join(dir, `killed-sealing-${attempt}`);
      const at = kills.seal.dir;
      let killed = false;
      try {
        await killWhen(TINY, at, 2, () => existsSync(partial(at, "train-report.json")) && !existsSync(partial(at, "manifest.json")));
        killed = existsSync(partial(at)) && !existsSync(join(at, "generations", gen(2)));
      } catch {
        // the driver finished before the signal: the window was missed
      }
      if (killed) break;
      if (attempt === 5) throw new Error("the seal window was missed five times");
    }
    expect(existsSync(partial(kills.seal.dir, "train-report.json"))).toBe(true);
    kills.seal.resumed = run(TINY, kills.seal.dir, 2);
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays, trains and commits two generations, each with records, a learner state and a manifest", () => {
    expect(first).toMatchObject({ resumed_from: null, discarded: [], generation: 2, played: [1, 2] });
    for (const g of [0, 1, 2]) {
      const files = readdirSync(join(uninterrupted, "generations", gen(g)));
      expect(files).toEqual(expect.arrayContaining([`learner-${g}.safetensors`, `learner-${g}.json`, "manifest.json", "records", "team-set.json"]));
    }
    expect(readdirSync(join(uninterrupted, "generations")).filter((n) => n.endsWith(".partial"))).toEqual([]);
    // The run metrics (ADR-010 erratum A1) sit beside the generations, outside every manifest.
    expect(readdirSync(join(uninterrupted, "metrics")).sort()).toEqual([0, 1, 2].map((g) => `${gen(g)}.json`));
    for (const g of [0, 1, 2]) {
      expect(json(uninterrupted, "metrics", `${gen(g)}.json`)).toMatchObject({ schema_version: 1, generation: g, device: "cpu", peak_accelerator_memory_bytes: null });
      expect(Object.keys(manifest(uninterrupted, g).files).some((name) => name.includes("metrics"))).toBe(false);
    }
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

  it("writes the files the training-data inspector reads with exactly the key sets it pins (fixtures/loop/inspector-run)", () => {
    // The inspector's strict readers (ml/areneon_ml/inspector/readers.py) are tested against the committed fixture run,
    // whose key sets tests/ml/inspector/test_readers.py ties to the readers. A change to the loop's writer shows here.
    const fixture = join(ROOT, "fixtures/loop/inspector-run");
    const keys = (value: unknown): string[] => Object.keys(value as object).sort();
    const shape = (root: string): Record<string, unknown> => {
      const plan = json<Record<string, unknown>[]>(root, "generations", gen(1), "plan.json");
      const generation = json<{ games: Record<string, unknown>[]; manifest: unknown }>(root, "generations", gen(1), "generation.json");
      return {
        plan: [...new Set(plan.map((entry) => keys(entry).join()))],
        teams: [...new Set(plan.map((entry) => keys(entry.teams).join()))],
        generation: keys(generation),
        games: [...new Set(generation.games.map((game) => keys(game).join()))],
        manifest: keys(generation.manifest),
        teamSet: keys(json(root, "generations", gen(1), "team-set.json")),
        teamSetG0: keys(json(root, "generations", gen(0), "team-set.json")),
      };
    };
    expect(shape(uninterrupted)).toEqual(shape(fixture));
  });

  it("gives the same run with several policy workers and game processes: every generation's manifest and the learner-state hash", () => {
    expect(shardedSummary).toEqual(first);
    for (const g of [0, 1, 2]) {
      expect(read(sharded, "generations", gen(g), "manifest.json")).toBe(read(uninterrupted, "generations", gen(g), "manifest.json"));
    }
    for (const name of ["plan.json", "generation.json", "stream.jsonl", "league-run.json"]) {
      expect(read(sharded, "generations", gen(2), name)).toBe(read(uninterrupted, "generations", gen(2), name));
    }
    expect(json(sharded, "run-config.json")).toEqual(json(uninterrupted, "run-config.json")); // no operational flag reaches the config
    expect(readdirSync(join(sharded, "generations", gen(2))).sort()).toEqual(readdirSync(join(uninterrupted, "generations", gen(2))).sort());
  });

  it("reproduces: the same config gives the same generation 1, byte for byte, in an independent run", () => {
    for (const { dir: killed } of Object.values(kills)) {
      expect(read(killed, "generations", gen(1), "manifest.json")).toBe(read(uninterrupted, "generations", gen(1), "manifest.json"));
    }
  });

  it.each(["play", "write", "seal"] as const)("resumes a run SIGKILLed at the %s point from its last complete generation, to the same learner-state hash", (point) => {
    const { dir: killed, resumed } = kills[point];
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
  });

  it("leaves records the C1/C6 checker reads: the first pair is in the second pair's ancestry", () => {
    const out = join(dir, "provenance");
    execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "provenance", "--run-dir", uninterrupted, "--out", out], { env });
    const [parent, child] = [1, 2].map((g) => manifest(uninterrupted, g).checkpoints.battle as string) as [string, string];
    const classify = (candidate: string, opponent: string): { label: string } =>
      JSON.parse(
        execFileSync(PYTHON, ["-I", "-m", "areneon_ml.checkpoint", "classify", "--records", join(out, "records"), "--sources", join(out, "sources.json"), "--candidate", candidate, "--opponent", opponent], { encoding: "utf8", env }),
      ) as { label: string };
    expect(classify(child, parent).label).toBe("seen");
  });
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

    const scratchBefore = readdirSync(tmpdir()).filter((n) => n.startsWith("areneon-shards-"));
    // Two policy workers in each of two game processes write the very same files (#957), and leave nothing else behind.
    const sharded = join(dir, "sharded");
    const script = join(ROOT, "packages/showdown-bridge/scripts/play-generation.ts");
    const shardedLines: string[] = [];
    const shardedCode = await runGenerationCli(
      [
        "--config", TINY, "--generation", "1", "--policy-id", `policy:${"e".repeat(64)}`, "--preview", preview.checkpoint_id, "--battle", battle.checkpoint_id,
        "--run-dir", dir, "--engine-data", DATA, "--encoder-version", preview.encoder_version, "--worker-python", PYTHON, "--out", sharded,
        "--policy-workers", "2", "--game-processes", "2",
      ],
      (line) => shardedLines.push(line),
      undefined,
      script,
    );
    expect(shardedCode).toBe(0);
    expect(shardedLines).toEqual(lines);
    const names = (root: string): string[] => readdirSync(root, { recursive: true, encoding: "utf8" }).sort();
    expect(names(sharded)).toEqual(names(out));
    for (const name of names(out).filter((n) => n.endsWith(".json") || n.endsWith(".jsonl"))) {
      expect(readFileSync(join(sharded, name), "utf8")).toBe(readFileSync(join(out, name), "utf8"));
    }
    expect(readdirSync(tmpdir()).filter((n) => n.startsWith("areneon-shards-") && !scratchBefore.includes(n))).toEqual([]);
  });
});

// ADR-018 slice 3: the damage-aware baseline in the opponent mix, through the real driver and worker. The config is the
// tiny sample with the baseline's two games in place of random-legal's, under its own run key.
describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the training loop with the damage-aware baseline", () => {
  let dir: string;
  let summary: RunSummary;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-loop-damage-"));
    const tiny = JSON.parse(readFileSync(TINY, "utf8")) as { seeds: { run_key: string }; generation: { baseline_games: unknown[] } };
    tiny.seeds.run_key = "tiny-damage-run";
    tiny.generation.baseline_games = [{ actor: "damage-aware", policy_id: "damage-aware@v1", games: 2 }];
    writeFileSync(join(dir, "config.json"), JSON.stringify(tiny));
    summary = run(join(dir, "config.json"), join(dir, "run"), 1);
  }, 600_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("completes, records the baseline's identity and counts, and trains on the pair's seats only", () => {
    expect(summary).toMatchObject({ generation: 1, played: [1] });
    const record = json<{ opponents: { opponent_policy_id: string; games: number }[]; baselines: { policy_id: string; identity_hash: string; games: number; counts: { decisions: number } }[] }>(
      join(dir, "run"), "generations", gen(1), "league-run.json",
    );
    expect(record.opponents.find((o) => o.opponent_policy_id === "damage-aware@v1")?.games).toBe(2);
    expect(record.baselines).toHaveLength(1);
    expect(record.baselines[0]).toMatchObject({ policy_id: "damage-aware@v1", games: 2 });
    expect(record.baselines[0]?.identity_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(record.baselines[0]?.counts.decisions).toBeGreaterThan(0);
    const set = json(join(dir, "run"), "generations", gen(1), "training-set.json");
    const behaviour = (hash: string): string => json(join(dir, "run"), "generations", gen(1), "episodes", `${hash}.json`).provenance.behaviour_policy_id;
    const all = json<{ episode_hashes: string[] }>(join(dir, "run"), "generations", gen(1), "generation.json").episode_hashes;
    expect(all.filter((hash) => behaviour(hash) === "damage-aware@v1")).toHaveLength(2);
    expect(set.episode_hashes).toEqual(all.filter((hash) => behaviour(hash) === set.behaviour_policy_id).sort());
  });
});
