// The defaults of `runPopulationCli` the other tests replace with seams (#1333): the real way a further game process is
// started and linked to its starter (both stubbed here so no process starts), a candidate feature set, and a calibration
// of a config with no population.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { loadEngine } from "../src/engine.ts";
import type { FeatureSetup } from "../src/league-generation.ts";
import { trainingTeamPool } from "../src/league-generation.ts";
import { runCalibrationCli, runPopulationCli, type PopulationCliInput } from "../src/population-cli.ts";
import { populationConfig } from "./population-support.ts";

const hooks = { started: [] as string[], linked: 0, alive: undefined as number | undefined, fake: false };
vi.mock("../src/shard-process.ts", async (original) => ({
  ...(await original<typeof import("../src/shard-process.ts")>()),
  startShard: (script: string) => {
    hooks.started.push(script);
    throw new TypeError("no process in this test");
  },
}));
vi.mock("../src/parent-link.ts", async (original) => ({
  ...(await original<typeof import("../src/parent-link.ts")>()),
  exitWhenParentDies: () => (hooks.linked++, Promise.resolve("a".repeat(64))),
}));

vi.mock("../src/population-generation.ts", async (original) => {
  const real = await original<typeof import("../src/population-generation.ts")>();
  return {
    ...real,
    playPopulationGeneration: (options: Parameters<typeof real.playPopulationGeneration>[0]) => {
      if (!hooks.fake) return real.playPopulationGeneration(options);
      hooks.alive = options.live_children?.();
      return Promise.resolve({ members: [] } as unknown as Awaited<ReturnType<typeof real.playPopulationGeneration>>);
    },
    writePopulationGeneration: (...args: Parameters<typeof real.writePopulationGeneration>) => (hooks.fake ? undefined : real.writePopulationGeneration(...args)),
  };
});

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const scratch = mkdtempSync(join(tmpdir(), "population-cli-defaults-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const config = populationConfig(2, 2);
const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
const features = { statement: { feature_set: "probe-damage/1", feature_set_hash: "a".repeat(64) }, configs: {} } as unknown as FeatureSetup;
const idle = { start: () => Promise.resolve(), close: () => Promise.resolve() } as unknown as ReturnType<NonNullable<PopulationCliInput["make_client"]>>;
const input = (out: string, change: object = {}): PopulationCliInput => ({
  engine, config, generation: 1, pool, out, run_dir: scratch, worker_python: "/nonexistent/python", engine_data: "unused", engine_data_hash: "e".repeat(64), encoder_version: "enc1:x",
  members: [0, 1].map((k) => ({ member: `m${k}`, preview: String(k).repeat(64), battle: String(k + 2).repeat(64), policy_id: `policy:${String(k + 4).repeat(64)}` })),
  ...change,
});

describe("runPopulationCli defaults", () => {
  it("starts further game processes with startShard by default and ends cleanly when it fails", async () => {
    await expect(runPopulationCli(input(join(scratch, "a"), { processes: 2, shard: { index: 0, of: 2 }, shard_script: "play.ts", make_client: () => idle }), () => undefined)).rejects.toThrow("no process");
    expect(hooks.started).toEqual(["play.ts"]);
  });

  it("links a further game process to its starter with exitWhenParentDies by default", async () => {
    // The pool arrives after the keep-alive timer has fired once.
    const slow = async (): Promise<typeof pool> => (await new Promise((resolve) => setTimeout(resolve, 1_100)), pool);
    const run = input(join(scratch, "b"), { pool: undefined, load_pool: slow, further: true, shard: { index: 99, of: 100 }, make_client: () => idle });
    expect(await runPopulationCli(run, () => undefined)).toBe(0);
    expect(hooks.linked).toBe(1);
  }, 120_000);

  it("gives the workers the feature statement and the options the feature configs", async () => {
    await expect(runPopulationCli(input(join(scratch, "c"), { features }), () => undefined)).rejects.toThrow();
    const run = input(join(scratch, "d"), { features, further: true, shard: { index: 99, of: 100 }, make_client: () => idle, parent_link: () => Promise.resolve("a".repeat(64)) });
    expect(await runPopulationCli(run, () => undefined)).toBe(0);
  }, 120_000);
});

describe("the idle watchdog's view of the further processes", () => {
  it("counts the game processes that are alive", async () => {
    hooks.fake = true;
    const child = (alive: boolean) => ({ entries: Promise.resolve([]), usage: Promise.resolve({}), alive: () => alive, tell: () => undefined, stop: () => undefined });
    const start = (_s: string, _a: readonly string[], _o: string, index: number) => child(index === 1) as unknown as ReturnType<NonNullable<PopulationCliInput["start_shard"]>>;
    const run = input(join(scratch, "f"), { processes: 3, shard: { index: 0, of: 3 }, shard_script: "play.ts", start_shard: start, make_client: () => idle });
    try {
      expect(await runPopulationCli(run, () => undefined)).toBe(0);
    } finally {
      hooks.fake = false;
    }
    expect(hooks.alive).toBe(1);
  });
});

describe("runCalibrationCli", () => {
  it("refuses a config with no population section", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const bare = { ...config, population: undefined } as unknown as typeof config;
    expect(await runCalibrationCli({ engine, config: bare, pool, out: join(scratch, "e") }, () => undefined)).toBe(1);
    expect(error.mock.calls.flat().join(" ")).toMatch(/no population section/);
    error.mockRestore();
  });
});
