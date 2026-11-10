// The population loop with candidate features, a shared trunk and a Battle-only warm start (#1333), end to end on CPU:
// the Python driver plays A1 population generations through Node, the real probe and one real policy worker per member,
// trains the members in turn (value warm-up and the KL anchor on the warm Battle role) and commits one directory per
// generation. The files are the same for one game process and for two game processes with two workers per member, and a
// run killed at the play, write and seal points resumes to the same bytes. Needs the project's Python environment (skipped
// without it, unless ARENEON_REQUIRE_PYTHON_WORKER=1, which the typescript CI job sets).
import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { featureSetup } from "../src/league-generation.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(GOLDEN_DIR, "engine-data.json");
const GENERATIONS = 2;
const LABELS = ["m0", "m1"] as const;
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };

interface RunSummary {
  resumed_from: number | null;
  discarded: string[];
  generation: number;
  members: Record<string, { checkpoints: { battle: string; preview: string }; learner_state_hash: string; policy_id: string }>;
  played: number[];
}

/** What `learner clone` leaves for a warm start, with seeded untrained networks standing in for the cloned ones (the loop reads the files, never the training). */
const CLONE = `
import json, sys
from pathlib import Path
from areneon_contracts.contracts import jcs
from areneon_ml.checkpoint.files import checkpoint_bytes, write_checkpoint
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import read_run_config
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, init_network
root, config_file, data_file, manifest = Path(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
config = read_run_config(json.loads(Path(config_file).read_text()))
game = GameData.from_table(json.loads(Path(data_file).read_text()))
teacher = "c" * 64
ids = {}
(root / "records").mkdir(parents=True, exist_ok=True)
for offset, role in enumerate(ROLES):
    found = write_checkpoint(root, checkpoint_bytes(init_network(role, game, config.model, 100 + offset), game))
    ids[role] = found
    record = read_record({
        "schema_version": 1, "checkpoint_id": found, "role": role, "lineage_id": "tiny-clone", "league_opponents": [],
        "training_set_hashes": [teacher], "team_set_manifest_hash": "e" * 64, "run_config_hash": "f" * 64,
        "learner_state_hash": "d" * 64, "encoder_version": encoder_version(game, config.model.encoder), "data_class": "synthetic", "manifest": manifest,
    })
    (root / "records" / (found + ".json")).write_bytes(jcs(record.to_json()) + b"\\n")
(root / "sources.json").write_bytes(jcs({teacher: []}))
print(json.dumps(ids))
`;

interface TinyConfig {
  format_id: string;
  seeds: { run_key: string };
  generation: { baseline_games: unknown };
  population: unknown;
  model: { encoder: { options: unknown }; options: unknown };
  warm_start?: unknown;
  ppo: { value_warmup?: unknown; anchor?: unknown };
}
interface PopulationBlock {
  members: number;
  seats_per_member: number;
  shares: unknown;
  selection: { min_roots: number };
  rating: unknown;
  explore: { bounds: Record<string, unknown> };
}

const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const args = (config: string, runDir: string, ...extra: string[]): string[] => ["-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA, "--generations", String(GENERATIONS), "--node", process.execPath, "--worker-python", PYTHON, ...extra];
const run = (config: string, runDir: string, ...extra: string[]): RunSummary => JSON.parse(execFileSync(PYTHON, args(config, runDir, ...extra), { encoding: "utf8", env })) as RunSummary;
const read = (runDir: string, ...path: string[]): string => readFileSync(join(runDir, ...path), "utf8");
const partial = (runDir: string, ...path: string[]): string => join(runDir, "generations", `.${gen(GENERATIONS)}.partial`, ...path);

/** The files of a run that a rerun must reproduce: every generation's manifest (which hashes every file of it) and the checkpoint store. */
const files = (runDir: string): Record<string, string> => {
  const out: Record<string, string> = {};
  for (let g = 0; g <= GENERATIONS; g++) out[`manifest-${g}`] = read(runDir, "generations", gen(g), "manifest.json");
  for (const name of readdirSync(join(runDir, "checkpoints")).sort()) out[name] = name;
  return out;
};

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

async function killWhen(config: string, runDir: string, ready: () => boolean): Promise<void> {
  const child: ChildProcess = spawn(PYTHON, args(config, runDir), { env, detached: true, stdio: "ignore" });
  const exited = new Promise<number | null>((resolve) => child.once("close", (code) => resolve(code)));
  const deadline = Date.now() + 900_000;
  while (!ready()) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error("the driver ended or stalled before the kill point");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  process.kill(-(child.pid as number), "SIGKILL");
  expect(await exited).toBeNull();
}

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the population loop with candidate features, shared_trunk and a Battle-only warm start, tiny and on CPU", () => {
  let dir: string;
  let config: string;
  let whole: string;
  let wholeSummary: RunSummary;
  let sharded: string;
  let shardedSummary: RunSummary;
  const kills: Record<"play" | "write" | "seal", { dir: string; resumed: RunSummary }> = {
    play: { dir: "", resumed: undefined as unknown as RunSummary },
    write: { dir: "", resumed: undefined as unknown as RunSummary },
    seal: { dir: "", resumed: undefined as unknown as RunSummary },
  };
  const prepared = (name: string, ids: Record<string, string>, cloneDir: string): string => {
    const target = join(dir, name);
    execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "import-clone", "--clone-dir", cloneDir, "--run-dir", target, "--roles", "battle"], { encoding: "utf8", env });
    expect(ids.battle).toMatch(/^[0-9a-f]{64}$/);
    return target;
  };

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "areneon-population-features-"));
    const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as TinyConfig;
    const sample = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/population.json"), "utf8")) as { population: PopulationBlock };
    const population = sample.population;
    population.members = 2;
    population.seats_per_member = 2;
    population.shares = { ladder_ppm: 500000, self_play_ppm: 500000, past_pool_ppm: 0 };
    population.selection.min_roots = 2;
    population.rating = { window: 3, calibration_pairs: 2, bootstrap_resamples: 40 };
    population.explore.bounds.self_play_ppm = { min: 450000, max: 550000 };
    population.explore.bounds.past_pool_ppm = { min: 0, max: 0 };
    tiny.generation.baseline_games = [
      { actor: "random-legal", policy_id: "random-legal@v1", games: 2 },
      { actor: "actor-rule", policy_id: "actor-rule@v1", games: 2 },
    ];
    tiny.seeds.run_key = "tiny-population-features";
    tiny.population = population;
    // The feature set this checkout computes, a shared trunk, and a Battle-only warm start with a value warm-up and the anchor.
    tiny.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: "0".repeat(64) } };
    let hash = "";
    try {
      featureSetup(loadEngine(), readTrainingRunConfig(tiny));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    tiny.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
    tiny.model.options = { shared_trunk: true };
    config = join(dir, "config.json");
    // The clone: seeded, untrained networks with clone records, made by the Python test helpers for this model section.
    const cloneDir = join(dir, "clone");
    mkdirSync(cloneDir, { recursive: true });
    writeFileSync(config, JSON.stringify(tiny));
    const manifest = JSON.stringify(resolveFormat(loadEngine(), tiny.format_id).manifest);
    const ids = JSON.parse(execFileSync(PYTHON, ["-I", "-c", CLONE, cloneDir, config, DATA, manifest], { encoding: "utf8", env })) as Record<string, string>;
    tiny.warm_start = { battle: ids.battle };
    tiny.ppo.value_warmup = { generations: 1 };
    tiny.ppo.anchor = { kl_micro: 100000, schedule: { kind: "constant" } };
    writeFileSync(config, JSON.stringify(tiny));
    whole = prepared("whole", ids, cloneDir);
    wholeSummary = run(config, whole);
    // Two game processes, two workers per member in each: the same files.
    sharded = prepared("sharded", ids, cloneDir);
    shardedSummary = run(config, sharded, "--game-processes", "2", "--policy-workers", "2");
    kills.play.dir = prepared("killed-playing", ids, cloneDir);
    await killWhen(config, kills.play.dir, () => existsSync(partial(kills.play.dir)) && liveProcesses(kills.play.dir).length >= 2);
    expect(liveProcesses(kills.play.dir)).toEqual([]);
    kills.play.resumed = run(config, kills.play.dir);
    kills.write.dir = prepared("killed-writing", ids, cloneDir);
    await killWhen(config, kills.write.dir, () => existsSync(partial(kills.write.dir, "generation.json")) && liveProcesses(kills.write.dir).length >= 1);
    expect(existsSync(partial(kills.write.dir, "manifest.json"))).toBe(false);
    kills.write.resumed = run(config, kills.write.dir);
    for (let attempt = 1; ; attempt++) {
      kills.seal.dir = prepared(`killed-sealing-${attempt}`, ids, cloneDir);
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
  }, 3_600_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays, trains and commits two generations whose pair ids carry the feature hash and whose streams hold the rows", () => {
    expect(wholeSummary).toMatchObject({ resumed_from: null, discarded: [], generation: GENERATIONS, played: [1, 2] });
    const named = (JSON.parse(read(whole, "run-config.json")) as { model: { encoder: { options: { candidate_features: { feature_set_hash: string } } } } }).model.encoder.options.candidate_features.feature_set_hash;
    for (const label of LABELS) {
      expect(wholeSummary.members[label]?.policy_id).toMatch(/^policy:/);
      const stream = read(whole, "generations", gen(GENERATIONS), "members", label, "stream.jsonl");
      expect(stream).toContain(`"feature_set_hash":"${named}"`);
    }
    expect(wholeSummary.members.m0?.checkpoints.preview).not.toBe(wholeSummary.members.m1?.checkpoints.preview);
  });

  it("writes the same generations for two game processes with two workers per member as for one", () => {
    expect(shardedSummary.members).toEqual(wholeSummary.members);
    expect(files(sharded)).toEqual(files(whole));
    for (const label of LABELS) {
      expect(read(sharded, "generations", gen(GENERATIONS), "members", label, "stream.jsonl")).toBe(read(whole, "generations", gen(GENERATIONS), "members", label, "stream.jsonl"));
    }
  });

  it.each(["play", "write", "seal"] as const)("resumes a run SIGKILLed at the %s point from its last complete generation, to the same bytes", (point) => {
    const { dir: killed, resumed } = kills[point];
    expect(resumed).toMatchObject({ resumed_from: GENERATIONS - 1, discarded: [`.${gen(GENERATIONS)}.partial`], generation: GENERATIONS, played: [GENERATIONS] });
    for (const label of LABELS) expect(resumed.members[label]).toEqual(wholeSummary.members[label]);
    expect(files(killed)).toEqual(files(whole));
  });
});
