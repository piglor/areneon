import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { combineUsage, featuresDirOf, interimPoolSource, optionalFlag, readShardFile, runGenerationCli, runManifestCli, startShard, UsageError, writeShardFile } from "../src/generation-cli.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, readTrainingRunConfig } from "../src/index.ts";
import { type PlayedEntry, trainingTeamPool } from "../src/league-generation.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const DATA = join(GOLDEN_DIR, "engine-data.json");
const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-cli-"));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const flags = (config: string, change: Record<string, string | undefined> = {}): string[] =>
  Object.entries({
    config, generation: "1", "policy-id": `policy:${"a".repeat(64)}`, preview: "p".repeat(64), battle: "b".repeat(64),
    "run-dir": dir, "engine-data": DATA, "encoder-version": "enc1:x", "worker-python": join(dir, "no-python"), out: join(dir, "out"),
    ...change,
  }).flatMap(([name, value]) => (value === undefined ? [] : [`--${name}`, value]));

describe("format-manifest", () => {
  it("prints the version manifest of a supported format as one canonical line", () => {
    const lines: string[] = [];
    expect(runManifestCli(["--format", GOLDEN_FORMAT], (line) => lines.push(line))).toBe(0);
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ format_id: GOLDEN_FORMAT, showdown_commit: loadEngine().commit });
  });

  it("refuses a missing flag, an unknown flag and an unsupported format", () => {
    expect(() => runManifestCli([])).toThrow(/--format is required/);
    expect(() => runManifestCli(["--nope"])).toThrow(UsageError);
    expect(() => runManifestCli(["--format", "gen9doesnotexist"])).toThrow();
  });
});

describe("play-generation", () => {
  it("refuses what it was not given, before it starts anything", async () => {
    const config = join(dir, "tiny.json");
    writeFileSync(config, JSON.stringify(tiny));
    await expect(runGenerationCli(flags(config, { out: undefined }))).rejects.toThrow(/--out is required/);
    await expect(runGenerationCli(flags(config, { generation: "0" }))).rejects.toThrow(/positive integer/);
    await expect(runGenerationCli(flags(config, { generation: "x" }))).rejects.toThrow(UsageError);
    await expect(runGenerationCli([...flags(config), "--unknown", "1"])).rejects.toThrow(UsageError);
    writeFileSync(join(dir, "bad.json"), JSON.stringify({ ...tiny, notes: "x" }));
    await expect(runGenerationCli(flags(join(dir, "bad.json")))).rejects.toThrow(/invalid TrainingRunConfig/);
  });

  it("refuses a config that names a team dataset instead of playing it on the interim pool", async () => {
    const named = { ...tiny, team_dataset: { path: "somewhere", dataset_hash: "a".repeat(64) } };
    const file = join(dir, "named.json");
    writeFileSync(file, JSON.stringify(named));
    await expect(runGenerationCli(flags(file))).rejects.toThrow(/team-dataset\/scripts\/play-generation/);
    expect(() => interimPoolSource(loadEngine(), readTrainingRunConfig(named))).toThrow(UsageError);
    expect(interimPoolSource(loadEngine(), readTrainingRunConfig(tiny)).teams.length).toBeGreaterThan(60);
  });

  it("refuses a worker or process count that is not a small positive integer, and a slice it cannot place", async () => {
    const file = join(dir, "tiny.json");
    writeFileSync(file, JSON.stringify(tiny));
    for (const flag of ["--policy-workers", "--game-processes"]) {
      for (const bad of ["0", "1.5", "x", "", "65", "0x2", "1e0", " 2"]) {
        await expect(runGenerationCli([...flags(file), flag, bad], console.log, undefined, "play.ts")).rejects.toThrow(new RegExp(`${flag} must be an integer from 1 to 64`));
      }
    }
    await expect(runGenerationCli([...flags(file), "--game-processes", "2"])).rejects.toThrow(/needs the script to start again/);
    for (const bad of ["1", "x/2", "2/2", "0/1", "0/65"]) {
      await expect(runGenerationCli([...flags(file), "--shard", bad])).rejects.toThrow(/INDEX\/COUNT/);
    }
    await expect(runGenerationCli([...flags(file), "--shard", "1/2", "--game-processes", "2"], console.log, undefined, "play.ts")).rejects.toThrow(/INDEX\/COUNT/);
  });

  it("refuses a run_key that is still the CHANGE-ME placeholder", async () => {
    const file = join(dir, "placeholder.json");
    writeFileSync(file, JSON.stringify({ ...tiny, seeds: { ...(tiny.seeds as object), run_key: "CHANGE-ME-PER-RUN" } }));
    await expect(runGenerationCli(flags(file))).rejects.toThrow(/CHANGE-ME/);
  });

  it("takes its teams from the pool source it is given", async () => {
    const file = join(dir, "tiny.json");
    writeFileSync(file, JSON.stringify(tiny));
    const refuse = (): never => {
      throw new UsageError("from the port");
    };
    await expect(runGenerationCli(flags(file), console.log, refuse)).rejects.toThrow("from the port");
  });

  it("fails with exit 1 when the config leaves no team to draw, before it starts the worker", async () => {
    const abilities = [...new Set(trainingTeamPool(loadEngine()).teams.flatMap((team) => team.map((set) => set.ability)))];
    const every = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), excluded_abilities: abilities } });
    const file = join(dir, "every.json");
    writeFileSync(file, JSON.stringify(every));
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli(flags(file))).toBe(1);
      expect(error).toHaveBeenCalledWith(expect.stringContaining("the team pool is empty"));
    } finally {
      error.mockRestore();
    }
  });
});

describe("a run config that names a candidate feature set (ADR-022)", () => {
  it("fails with exit 1, before it starts the worker, when this checkout does not compute the named hash", async () => {
    const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
    raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: "0".repeat(64) } };
    const file = join(dir, "features.json");
    writeFileSync(file, JSON.stringify(raw));
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli(flags(file))).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/names the candidate feature set probe-damage\/1 0{64}, which this checkout computes as [0-9a-f]{64}/);
    } finally {
      error.mockRestore();
    }
  });
});

describe("a run config that names an opponent sheet source (ADR-027 slice 4)", () => {
  const named = (hash: string): string => {
    const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
    raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } };
    const file = join(dir, `sheet-${hash.slice(0, 4)}.json`);
    writeFileSync(file, JSON.stringify(raw));
    return file;
  };

  it("fails with exit 1 when the script has no factory, and when the factory's port computes another hash", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli(flags(named("a".repeat(64))))).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/no belief features factory/);
      const factory = () => ({ features_hash: "b".repeat(64), prior_format_id: GOLDEN_FORMAT }) as never;
      expect(await runGenerationCli(flags(named("c".repeat(64))), console.log, interimPoolSource, undefined, undefined, factory)).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/computes as b{64}/);
    } finally {
      error.mockRestore();
    }
  });
});

describe("a further game process", () => {
  const script = (name: string, body: string): string => {
    const file = join(dir, name);
    writeFileSync(file, body);
    return file;
  };

  it("hands back the entries it wrote", async () => {
    const out = join(dir, "shard-ok");
    const ok = script("ok.mjs", `import { mkdirSync, writeFileSync } from "node:fs"; mkdirSync(process.argv[2], { recursive: true }); writeFileSync(process.argv[2] + "/shard.jsonl", '{"game":1}\\n');`);
    await expect(startShard(ok, [out], out, 1).entries).resolves.toEqual([{ game: 1 }]);
  });

  it("hands back the peak memory it left, and none when it left none or garbage", async () => {
    const out = join(dir, "shard-usage");
    const body = (usage: string): string =>
      `import { mkdirSync, writeFileSync } from "node:fs"; mkdirSync(process.argv[2], { recursive: true }); writeFileSync(process.argv[2] + "/shard.jsonl", '{"game":1}\\n'); ${usage}`;
    const withUsage = script("usage.mjs", body(`writeFileSync(process.argv[2] + "/usage.json", '{"game_process_peak_rss_bytes":7}');`));
    await expect(startShard(withUsage, [out], out, 1).usage).resolves.toEqual({ game_process_peak_rss_bytes: 7 });
    const bare = script("bare.mjs", body(""));
    const bareOut = join(dir, "shard-bare");
    await expect(startShard(bare, [bareOut], bareOut, 1).usage).resolves.toEqual({});
    const junk = script("junk.mjs", body(`writeFileSync(process.argv[2] + "/usage.json", 'not json');`));
    const junkOut = join(dir, "shard-junk");
    await expect(startShard(junk, [junkOut], junkOut, 1).usage).resolves.toEqual({});
    // A process that fails has no usage to report either.
    const failing = script("failing-usage.mjs", "process.exit(3);");
    const failingChild = startShard(failing, [], join(dir, "shard-none"), 6);
    await expect(failingChild.usage).rejects.toThrow(/game process 6 failed/);
  });

  it("combines several processes' peaks into the largest of each", () => {
    expect(combineUsage([])).toEqual({});
    expect(combineUsage([{ game_process_peak_rss_bytes: 5 }, { game_process_peak_rss_bytes: 9, policy_worker_peak_rss_bytes: 2 }, {}])).toEqual({
      game_process_peak_rss_bytes: 9,
      policy_worker_peak_rss_bytes: 2,
    });
  });

  it("fails the generation when the process exits non-zero, leaves no result, or does not start", async () => {
    const out = join(dir, "shard-none");
    const failing = script("failing.mjs", `console.error("boom"); process.exit(3);`);
    await expect(startShard(failing, [], out, 2).entries).rejects.toThrow(/game process 2 failed \(exit 3\): boom/);
    const silent = script("silent.mjs", "");
    await expect(startShard(silent, [], out, 3).entries).rejects.toThrow(/game process 3 left no readable result/);
    await expect(startShard(silent, [], out, 4, join(dir, "no-such-node")).entries).rejects.toThrow(/game process 4 did not start/);
    const child = startShard(script("slow.mjs", "setTimeout(() => {}, 60000);"), [], out, 5);
    child.stop();
    await expect(child.entries).rejects.toThrow(/game process 5 failed/);
  });
});

describe("the directory the game processes leave feature rows in (#1185)", () => {
  it("is the one given, else a directory of the scratch when the run reads a feature set, else none", () => {
    expect(featuresDirOf("/given", "/scratch", true)).toBe("/given");
    expect(featuresDirOf(undefined, "/scratch", true)).toBe("/scratch/features");
    expect(featuresDirOf(undefined, "/scratch", false)).toBeUndefined();
    expect(featuresDirOf(undefined, undefined, true)).toBeUndefined();
    expect(optionalFlag("--features-dir", "/d")).toEqual(["--features-dir", "/d"]);
    expect(optionalFlag("--features-dir", undefined)).toEqual([]);
  });
});

describe("the shard handoff file (#998)", () => {
  it("is one canonical entry per line and reads back to the same entries, however many", () => {
    const path = join(mkdtempSync(join(tmpdir(), "shard-file-")), "shard.jsonl");
    const entries = Array.from({ length: 50 }, (_, game) => ({ game, row: { game, note: "ünï".repeat(game) }, episodes: [] })) as unknown as PlayedEntry[];
    writeShardFile(path, entries);
    expect(readFileSync(path, "utf8")).toBe(entries.map((entry) => `${jcs(entry as unknown as Json)}\n`).join(""));
    expect(readShardFile(path)).toEqual(entries);
  });
});
