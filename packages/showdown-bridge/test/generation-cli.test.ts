import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { spawn } from "node:child_process";
import { PassThrough } from "node:stream";
import { combinedLoad, combineUsage, exitWhenParentDies, featuresDirOf, interimPoolSource, MAX_COMBINED_LOAD, optionalFlag, readShardFile, runGenerationCli, runManifestCli, startShard, UsageError, writeShardFile } from "../src/generation-cli.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, readTrainingRunConfig } from "../src/index.ts";
import { runConfigHash } from "../src/training-run-config.ts";
import { type PlayedEntry, trainingTeamPool } from "../src/league-generation.ts";
import { resetProfile } from "../src/profile.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { populationConfig } from "./population-support.ts";

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

  it("makes its profile directory, times the pool load and removes the directory when it fails", async () => {
    const file = join(dir, "profiled.json");
    writeFileSync(file, JSON.stringify(tiny));
    const refuse = (): never => {
      throw new UsageError("from the port");
    };
    const before = new Set(readdirSync(tmpdir()).filter((name) => name.startsWith("areneon-profile-")));
    await expect(runGenerationCli(flags(file, { "policy-workers": "1" }).concat("--profile"), console.log, refuse)).rejects.toThrow("from the port");
    expect(readdirSync(tmpdir()).filter((name) => name.startsWith("areneon-profile-") && !before.has(name))).toEqual([]); // removed again
    resetProfile();
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

describe("a population run (ADR-021 slice 4)", () => {
  const member = (k: number): string[] => ["--member", `m${k},${String(k).repeat(64)},${String(k + 2).repeat(64)},policy:${String(k + 4).repeat(64)}`];
  const perMember = { preview: undefined, battle: undefined, "policy-id": undefined };

  it("is played with --member for each member, in place of --preview, --battle and --policy-id, and only then", async () => {
    const file = join(dir, "population.json");
    writeFileSync(file, JSON.stringify(populationConfig(2, 2)));
    const plain = join(dir, "tiny.json");
    writeFileSync(plain, JSON.stringify(tiny));
    await expect(runGenerationCli(flags(file))).rejects.toThrow(/--member for each member/);
    await expect(runGenerationCli([...flags(plain, perMember), ...member(0)])).rejects.toThrow(/--member for each member/);
    await expect(runGenerationCli(flags(file, perMember))).rejects.toThrow(/is required/);
    await expect(runGenerationCli([...flags(file, perMember), "--member", "m0,x"])).rejects.toThrow(/--member is m<k>/);
    await expect(runGenerationCli([...flags(file, perMember), ...member(0), ...member(1), "--profile"], console.log, undefined, "play.ts")).rejects.toThrow(/--profile does not apply/);
    for (const extra of [["--policy-workers", "2"], ["--game-processes", "2"], ["--shard", "0/2"]]) {
      await expect(runGenerationCli([...flags(file, perMember), ...member(0), ...member(1), ...extra], console.log, undefined, "play.ts")).rejects.toThrow(/one process with one worker per member/);
    }
  });

  it("reaches the population branch with the members it was given, and the pool source", async () => {
    const file = join(dir, "population.json");
    writeFileSync(file, JSON.stringify(populationConfig(2, 2)));
    const refuse = (): never => {
      throw new UsageError("from the port");
    };
    await expect(runGenerationCli([...flags(file, perMember), ...member(0), ...member(1)], console.log, refuse)).rejects.toThrow("from the port");
  });
});

describe("a population run with a past pool (ADR-021 slice 10)", () => {
  const member = (k: number): string[] => ["--member", `m${k},${String(k).repeat(64)},${String(k + 2).repeat(64)},policy:${String(k + 4).repeat(64)}`];
  const perMember = { preview: undefined, battle: undefined, "policy-id": undefined };
  const refuse = (): never => {
    throw new UsageError("from the port");
  };
  const poolFile = async (config: ReturnType<typeof populationConfig>): Promise<unknown> => ({
    schema_version: 1,
    format: "past-pool/1",
    run_id: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    generation: 0,
    snapshots: [0, 1].map((k) => ({ id: `m${k}@g0`, member: `m${k}`, generation: 0, policy_id: `policy:${String(k + 6).repeat(64)}`, preview: "a".repeat(64), battle: "b".repeat(64) })),
    cells: [],
  });

  it("takes --past-pool exactly when the config has a past-pool share, and refuses a bad file", async () => {
    const withShare = populationConfig(2, 4, [500_000, 250_000, 250_000]);
    const file = join(dir, "past-config.json");
    writeFileSync(file, JSON.stringify(withShare));
    const plain = join(dir, "no-past-config.json");
    writeFileSync(plain, JSON.stringify(populationConfig(2, 2)));
    const pool = join(dir, "past-pool.json");
    writeFileSync(pool, JSON.stringify(await poolFile(withShare)));
    const members = [...member(0), ...member(1)];
    await expect(runGenerationCli([...flags(file, perMember), ...members])).rejects.toThrow(/--past-pool is required with a past-pool share/);
    await expect(runGenerationCli([...flags(plain, perMember), ...members, "--past-pool", pool])).rejects.toThrow(/refused without one/);
    const bad = join(dir, "bad-past-pool.json");
    writeFileSync(bad, JSON.stringify({ ...(await poolFile(withShare) as object), run_id: "other" }));
    await expect(runGenerationCli([...flags(file, perMember), ...members, "--past-pool", bad])).rejects.toThrow(/another run or config/);
    writeFileSync(bad, "not json");
    await expect(runGenerationCli([...flags(file, perMember), ...members, "--past-pool", bad])).rejects.toThrow(SyntaxError);
    // A good file reaches the pool source.
    await expect(runGenerationCli([...flags(file, perMember), ...members, "--past-pool", pool], console.log, refuse)).rejects.toThrow("from the port");
  });
});

describe("the anchor calibration of a population run (ADR-021 slice 7)", () => {
  const perMember = { preview: undefined, battle: undefined, "policy-id": undefined };
  const calibrate = (config: string, change: Record<string, string | undefined> = {}): string[] => [...flags(config, { generation: "0", ...perMember, ...change }), "--calibrate"];

  it("is played with --calibrate at generation 0 of a population config, with no member, and nowhere else", async () => {
    const file = join(dir, "calibrate.json");
    writeFileSync(file, JSON.stringify(populationConfig(2, 2)));
    const plain = join(dir, "tiny.json");
    writeFileSync(plain, JSON.stringify(tiny));
    await expect(runGenerationCli(calibrate(file, { generation: "1" }))).rejects.toThrow(/--calibrate plays generation 0/);
    await expect(runGenerationCli(flags(file, { generation: "0" }))).rejects.toThrow(/positive integer/);
    await expect(runGenerationCli([...calibrate(file), "--member", `m0,${"0".repeat(64)},${"2".repeat(64)},policy:${"4".repeat(64)}`])).rejects.toThrow(/takes no --member/);
    await expect(runGenerationCli([...calibrate(file), "--game-processes", "2"], console.log, undefined, "play.ts")).rejects.toThrow(/takes no --member and plays in one process/);
    await expect(runGenerationCli(calibrate(plain))).rejects.toThrow(/needs a population config/);
  });

  it("reaches the calibration branch with the pool source, and refuses a config without the reference anchor", async () => {
    const file = join(dir, "calibrate.json");
    writeFileSync(file, JSON.stringify(populationConfig(2, 2)));
    const refuse = (): never => {
      throw new UsageError("from the port");
    };
    await expect(runGenerationCli(calibrate(file), console.log, refuse)).rejects.toThrow("from the port");
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      // The tiny population config names no actor-rule@v1 baseline.
      expect(await runGenerationCli(calibrate(file))).toBe(1);
      expect(error).toHaveBeenCalledWith(expect.stringContaining("actor-rule@v1"));
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
      // A matching hash gets past the setup: the next thing that fails is the worker that cannot start.
      const matching = () => ({ features_hash: "d".repeat(64), prior_format_id: GOLDEN_FORMAT }) as never;
      await expect(runGenerationCli(flags(named("d".repeat(64))), console.log, interimPoolSource, undefined, undefined, matching)).rejects.toThrow(/cannot start the worker|ENOENT/);
    } finally {
      error.mockRestore();
    }
  });
});

describe("a run config that asks for search labels (ADR-026 erratum E2, #1290)", () => {
  const withLabels = (_generation: number, hash: string, name: string): string => {
    const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/search-distill.json"), "utf8")) as { generation: { search_labels: { search_config_hash: string; start_generation: number } } };
    raw.generation.search_labels.search_config_hash = hash;
    const file = join(dir, `labels-${name}.json`);
    writeFileSync(file, JSON.stringify(raw));
    return file;
  };

  it("fails with exit 1 when the script has no search setup factory, or the config names another label search config", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli(flags(withLabels(3, "e".repeat(64), "a"), { generation: "3" }))).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/no search setup factory/);
      const factory = () => {
        throw new Error("the factory is not reached with a wrong config hash");
      };
      expect(await runGenerationCli(flags(withLabels(3, "e".repeat(64), "b"), { generation: "3" }), console.log, interimPoolSource, undefined, undefined, undefined, factory)).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/which this checkout computes as/);
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
    await expect(child.entries).rejects.toThrow(/game process 5 failed \(signal SIGKILL\)/);
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

describe("a further game process and its parent (#980)", () => {
  it("is told to end when its parent's pipe closes, and only then", () => {
    const stdin = new PassThrough();
    const exits: number[] = [];
    exitWhenParentDies(stdin, (code) => exits.push(code));
    expect(exits).toEqual([]);
    stdin.end();
    return new Promise<void>((resolve) => setImmediate(() => {
      expect(exits).toEqual([1]);
      resolve();
    }));
  });

  it("exits when its parent dies, and the processes it started end with it", async () => {
    const script = join(dir, "orphan.mjs");
    const source = join(process.cwd(), "packages/showdown-bridge/src/generation-cli.ts");
    writeFileSync(
      script,
      `import { exitWhenParentDies } from ${JSON.stringify(source)};
       import { spawn } from "node:child_process";
       const worker = spawn(process.execPath, ["-e", "process.stdin.on('end', () => process.exit(0)); process.stdin.resume(); setTimeout(() => {}, 60000)"], { stdio: ["pipe", "ignore", "ignore"] });
       exitWhenParentDies();
       console.log(worker.pid);
       setTimeout(() => {}, 60000);`,
    );
    const child = spawn(process.execPath, [...process.execArgv, script], { stdio: ["pipe", "pipe", "inherit"] });
    try {
      const pid = await new Promise<number>((resolve, reject) => {
        child.once("error", reject);
        child.stdout.once("data", (chunk: Buffer) => resolve(Number(chunk.toString("utf8").trim())));
      });
      expect(Number.isInteger(pid) && pid > 0).toBe(true);
      const exited = new Promise<number | null>((resolve) => child.once("exit", (code) => resolve(code)));
      child.stdin.end(); // the parent "dies": its end of the pipe closes
      expect(await exited).toBe(1);
      const gone = async (): Promise<boolean> => {
        for (let i = 0; i < 100; i++) {
          try {
            process.kill(pid, 0);
          } catch {
            return true;
          }
          await new Promise((resolve) => setTimeout(resolve, 50));
        }
        return false;
      };
      expect(await gone()).toBe(true);
    } finally {
      child.kill("SIGKILL");
    }
  }, 60_000);

  it("warns about game processes x workers x games in flight past a stated bound, and refuses --pool-file outside a further game process", async () => {
    expect(combinedLoad(2, 3, 4)).toEqual({ games: 24, over: false });
    expect(combinedLoad(MAX_COMBINED_LOAD, 1, 1).over).toBe(false);
    expect(combinedLoad(MAX_COMBINED_LOAD, 1, 2)).toEqual({ games: MAX_COMBINED_LOAD * 2, over: true });
    const file = join(dir, "pool-flag.json");
    writeFileSync(file, JSON.stringify(tiny));
    await expect(runGenerationCli([...flags(file), "--pool-file", join(dir, "x")])).rejects.toThrow(/--pool-file is for a further game process only/);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      // 8 processes x 64 workers is far past the bound; the run is not refused for that (it stops here only for lack of a script to start again).
      await expect(runGenerationCli(flags(file, { "policy-workers": "64", "game-processes": "8" }))).rejects.toThrow(/script to start again/);
      expect(error).toHaveBeenCalledWith(expect.stringContaining("may be oversubscribed"));
    } finally {
      error.mockRestore();
    }
  });
});
