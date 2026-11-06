// The population loop end to end on CPU (ADR-021 slice 5): the Python driver plays A1 population generations through
// Node and one real policy worker per member, trains the members in turn, freezes every pair and commits one directory
// per generation. It needs the project's Python environment (`make setup`, uv sync); without it the file is skipped,
// unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript CI job sets it). The config is the tiny sample with two members.
import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { checkLadderRecord, type LadderRecord, readLadderRecord } from "../src/ladder-record.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(GOLDEN_DIR, "engine-data.json");
const GENERATIONS = 3;
const LABELS = ["m0", "m1"] as const;

interface MemberSummary {
  checkpoints: { battle: string; preview: string };
  learner_state_hash: string;
  policy_id: string;
}
interface RunSummary {
  resumed_from: number | null;
  discarded: string[];
  generation: number;
  members: Record<string, MemberSummary>;
  played: number[];
}
interface Manifest {
  members: Record<string, { checkpoints: Record<string, string>; learner_state_hash: string; training_set_hash: string | null }>;
  files: Record<string, string>;
}
interface Record_ {
  role: string;
  lineage_id: string;
  parent?: string;
  league_opponents: string[];
  training_set_hashes: string[];
}

const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };
const args = (config: string, runDir: string, generations: number): string[] => [
  "-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA,
  "--generations", String(generations), "--node", process.execPath, "--worker-python", PYTHON,
];
const run = (config: string, runDir: string): RunSummary => JSON.parse(execFileSync(PYTHON, args(config, runDir, GENERATIONS), { encoding: "utf8", env })) as RunSummary;
const read = (runDir: string, ...path: string[]): string => readFileSync(join(runDir, ...path), "utf8");
const json = <T>(runDir: string, ...path: string[]): T => JSON.parse(read(runDir, ...path)) as T;
const manifest = (runDir: string, g: number): Manifest => json<Manifest>(runDir, "generations", gen(g), "manifest.json");
const partial = (runDir: string, ...path: string[]): string => join(runDir, "generations", `.${gen(GENERATIONS)}.partial`, ...path);

/** Command lines of the live processes that mention `runDir` (Linux): the Node play script and the policy workers of that run. */
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
async function killWhen(config: string, runDir: string, ready: () => boolean): Promise<void> {
  const child: ChildProcess = spawn(PYTHON, args(config, runDir, GENERATIONS), { env, detached: true, stdio: "ignore" });
  const exited = new Promise<number | null>((resolve) => child.once("close", (code) => resolve(code)));
  const deadline = Date.now() + 480_000;
  while (!ready()) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error("the driver ended or stalled before the kill point");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  process.kill(-(child.pid as number), "SIGKILL");
  expect(await exited).toBeNull();
}

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the population loop, two members, tiny and on CPU", () => {
  let dir: string;
  let config: string;
  let first: string;
  let firstSummary: RunSummary;
  let second: string;
  // Real SIGKILLs of the driver's process group in generation 3, each followed by a resume:
  //  play: the simulator side is alive (Node and the members' workers), the partial may still be empty;
  //  write: Node has played and is writing its files (generation.json is written last), the workers are still alive;
  //  seal: the driver has trained the last member (its train-report.json exists) and is sealing, with no manifest yet.
  const kills: Record<"play" | "write" | "seal", { dir: string; resumed: RunSummary }> = {
    play: { dir: "", resumed: undefined as unknown as RunSummary },
    write: { dir: "", resumed: undefined as unknown as RunSummary },
    seal: { dir: "", resumed: undefined as unknown as RunSummary },
  };

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "areneon-population-"));
    const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown> & { seeds: { run_key: string } };
    const sample = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/population.json"), "utf8")) as { population: Record<string, unknown> & { explore: { bounds: Record<string, unknown> }; selection: Record<string, unknown> } };
    const population = sample.population;
    population.members = 2;
    population.seats_per_member = 2;
    population.shares = { ladder_ppm: 500000, self_play_ppm: 500000, past_pool_ppm: 0 };
    population.selection.min_roots = 2;
    // The ladder (ADR-021 slice 7): the two anchors of the tiny run, a short calibration and a cheap bootstrap.
    population.rating = { window: 3, calibration_pairs: 2, bootstrap_resamples: 40 };
    (tiny.generation as { baseline_games: unknown }).baseline_games = [
      { actor: "random-legal", policy_id: "random-legal@v1", games: 2 },
      { actor: "actor-rule", policy_id: "actor-rule@v1", games: 2 },
    ];
    population.explore.bounds.self_play_ppm = { min: 450000, max: 550000 };
    population.explore.bounds.past_pool_ppm = { min: 0, max: 0 };
    tiny.seeds.run_key = "tiny-population";
    tiny.population = population;
    config = join(dir, "config.json");
    writeFileSync(config, JSON.stringify(tiny));
    first = join(dir, "first");
    firstSummary = run(config, first);
    second = join(dir, "second");
    run(config, second);
    kills.play.dir = join(dir, "killed-playing");
    await killWhen(config, kills.play.dir, () => existsSync(partial(kills.play.dir)) && liveProcesses(kills.play.dir).length >= 2);
    expect(liveProcesses(kills.play.dir)).toEqual([]); // the whole group went down with the driver
    expect(existsSync(partial(kills.play.dir, "manifest.json"))).toBe(false);
    kills.play.resumed = run(config, kills.play.dir);
    kills.write.dir = join(dir, "killed-writing");
    await killWhen(config, kills.write.dir, () => existsSync(partial(kills.write.dir, "generation.json")) && liveProcesses(kills.write.dir).length >= 1);
    expect(existsSync(partial(kills.write.dir, "manifest.json"))).toBe(false);
    kills.write.resumed = run(config, kills.write.dir);
    // The seal window is short: if the signal lands after the commit, try again in a fresh directory.
    for (let attempt = 1; ; attempt++) {
      kills.seal.dir = join(dir, `killed-sealing-${attempt}`);
      const at = kills.seal.dir;
      let killed = false;
      try {
        await killWhen(config, at, () => existsSync(partial(at, "members", "m1", "train-report.json")) && !existsSync(partial(at, "manifest.json")));
        killed = existsSync(partial(at)) && !existsSync(join(at, "generations", gen(GENERATIONS)));
      } catch {
        // the driver finished before the signal: the window was missed
      }
      if (killed) break;
      if (attempt === 5) throw new Error("the seal window was missed five times");
    }
    kills.seal.resumed = run(config, kills.seal.dir);
  }, 3_000_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays, trains and commits three generations, each member with its own files, records and learner state", () => {
    expect(firstSummary).toMatchObject({ resumed_from: null, discarded: [], generation: GENERATIONS, played: [1, 2, 3] });
    expect(Object.keys(firstSummary.members)).toEqual([...LABELS]);
    for (let g = 0; g <= GENERATIONS; g++) {
      const files = readdirSync(join(first, "generations", gen(g)));
      expect(files).toEqual(expect.arrayContaining(["manifest.json", "members", "team-set.json"]));
      for (const label of LABELS) {
        expect(readdirSync(join(first, "generations", gen(g), "members", label))).toEqual(expect.arrayContaining([`learner-${g}.safetensors`, `learner-${g}.json`, "records"]));
        if (g > 0) expect(readdirSync(join(first, "generations", gen(g), "members", label))).toEqual(expect.arrayContaining(["episodes", "stream.jsonl", "training-set.json", "train-report.json"]));
      }
      if (g > 0) expect(files).toEqual(expect.arrayContaining(["plan.json", "league-run.json", "generation.json"]));
    }
    expect(readdirSync(join(first, "generations")).filter((n) => n.endsWith(".partial"))).toEqual([]);
    expect(readdirSync(join(first, "metrics"))).toEqual(expect.arrayContaining([0, 1, 2, 3].map((g) => `${gen(g)}.json`)));
    expect(json<{ schema_version: number }>(first, "generations", gen(1), "league-run.json").schema_version).toBe(2);
    // Different pairs from generation 0: each member is seeded from its own population-seed/1 draws.
    const zero = manifest(first, 0).members;
    expect(zero.m0?.checkpoints.preview).not.toBe(zero.m1?.checkpoints.preview);
    expect(zero.m0?.checkpoints.battle).not.toBe(zero.m1?.checkpoints.battle);
  });

  it("gives every member's records its lineage, its parent and the other member's pair as league opponent", () => {
    for (const label of LABELS) {
      const other = label === "m0" ? "m1" : "m0";
      for (let g = 1; g <= GENERATIONS; g++) {
        for (const role of ["battle", "preview"]) {
          const record = json<Record_>(first, "generations", gen(g), "members", label, "records", `${role}.json`);
          expect(record).toMatchObject({ role, lineage_id: `tiny-main.${label}` });
          expect(record.parent).toBe(manifest(first, g - 1).members[label]?.checkpoints[role]);
          const opponent = manifest(first, g - 1).members[other]?.checkpoints as Record<string, string>;
          expect(record.league_opponents).toEqual([opponent.battle, opponent.preview].sort());
          expect(record.training_set_hashes).toEqual([manifest(first, g).members[label]?.training_set_hash]);
        }
      }
    }
  });

  it("reproduces: an independent run writes every generation and the checkpoint store byte for byte", () => {
    for (let g = 0; g <= GENERATIONS; g++) expect(read(second, "generations", gen(g), "manifest.json")).toBe(read(first, "generations", gen(g), "manifest.json"));
    expect(readdirSync(join(second, "checkpoints")).sort()).toEqual(readdirSync(join(first, "checkpoints")).sort());
    expect(json(second, "run-config.json")).toEqual(json(first, "run-config.json"));
  });

  it.each(["play", "write", "seal"] as const)("resumes a run SIGKILLed at the %s point from its last complete generation, to the same learner-state hashes", (point) => {
    const { dir: killed, resumed } = kills[point];
    expect(resumed).toMatchObject({ resumed_from: GENERATIONS - 1, discarded: [`.${gen(GENERATIONS)}.partial`], generation: GENERATIONS, played: [GENERATIONS] });
    for (const label of LABELS) {
      expect(resumed.members[label]).toEqual(firstSummary.members[label]);
    }
    for (let g = 0; g <= GENERATIONS; g++) expect(read(killed, "generations", gen(g), "manifest.json")).toBe(read(first, "generations", gen(g), "manifest.json"));
  });

  it("calibrates the anchors in generation 0 and writes a ladder.json in every later generation that the TS reader accepts", async () => {
    const config = readTrainingRunConfig(json(first, "run-config.json"));
    const zero = join(first, "generations", gen(0));
    expect(readdirSync(zero)).toEqual(expect.arrayContaining(["calibration.json", "anchors.json", "manifest.json"]));
    const anchors = json<{ anchors: { anchor_id: string; strength: number }[]; cells: { pairs: number }[] }>(first, "generations", gen(0), "anchors.json");
    expect(anchors.anchors.map((a) => a.anchor_id)).toEqual(["actor-rule@v1", "random-legal@v1"]);
    expect(anchors.anchors[0]?.strength).toBe(1_000_000_000);
    expect(anchors.cells.map((c) => c.pairs)).toEqual([2]);
    expect(manifest(first, 0).files["anchors.json"]).toBeDefined();
    let before: LadderRecord | undefined;
    for (let g = 1; g <= GENERATIONS; g++) {
      const record = readLadderRecord(json(first, "generations", gen(g), "ladder.json"), before);
      await checkLadderRecord(record, config);
      expect(record.anchors).toEqual(anchors.anchors);
      expect(record.selections).toEqual([{ kind: "none", failed: "no_ready_member" }]);
      expect(record.members.map((m) => m.member)).toEqual([...LABELS]);
      expect(manifest(first, g).files["ladder.json"]).toBeDefined();
      before = record;
    }
  });

  it("is a no-op when the target is already committed", () => {
    const again = run(config, first);
    expect(again).toMatchObject({ resumed_from: GENERATIONS, played: [], generation: GENERATIONS });
    expect(again.members).toEqual(firstSummary.members);
  });
});
