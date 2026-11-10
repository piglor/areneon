// The population branch of `runGenerationCli` (#1333) with `runPopulationCli` stubbed: the options it spreads for game processes,
// a further game process and a candidate feature set.
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/engine.ts";
import { featureSetup, trainingTeamPool } from "../src/league-generation.ts";
import type { PopulationCliInput } from "../src/population-cli.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { populationConfig } from "./population-support.ts";

const seen: PopulationCliInput[] = [];
const hooks = { failing: false };
vi.mock("../src/league-generation.ts", async (original) => {
  const real = await original<typeof import("../src/league-generation.ts")>();
  return { ...real, featureSetup: (...args: Parameters<typeof real.featureSetup>) => (hooks.failing ? (() => { throw new TypeError("features broke"); })() : real.featureSetup(...args)) };
});
vi.mock("../src/population-cli.ts", async (original) => ({
  ...(await original<typeof import("../src/population-cli.ts")>()),
  runPopulationCli: (input: PopulationCliInput) => (seen.push(input), Promise.resolve(0)),
}));
const { runGenerationCli } = await import("../src/generation-cli.ts");

const dir = mkdtempSync(join(tmpdir(), "areneon-cli-pop-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const member = (k: number): string[] => ["--member", `m${k},${String(k).repeat(64)},${String(k + 2).repeat(64)},policy:${String(k + 4).repeat(64)}`];

describe("the population branch of play-generation", () => {
  it("hands runPopulationCli the shard, the further-process flags, the features and a pool loader", async () => {
    const raw = structuredClone(populationConfig(2, 2)) as unknown as { model: { encoder: { options?: unknown } } };
    const set = (hash: string): void => void (raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } });
    set("0".repeat(64));
    let hash = "";
    try {
      featureSetup(loadEngine(), readTrainingRunConfig(raw));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    set(hash);
    const file = join(dir, "c.json");
    writeFileSync(file, JSON.stringify(raw));
    const base = ["--config", file, "--generation", "1", "--run-dir", dir, "--engine-data", join(GOLDEN_DIR, "engine-data.json"), "--encoder-version", "enc1:x", "--worker-python", "none", ...member(0), ...member(1)];
    let loaded = 0;
    const source = (engine: Parameters<typeof trainingTeamPool>[0]) => (loaded++, trainingTeamPool(engine));
    hooks.failing = true;
    await expect(runGenerationCli([...base, "--out", join(dir, "o0")], () => undefined, source, "play.ts")).rejects.toThrow("features broke");
    hooks.failing = false;
    // The starter of two game processes.
    expect(await runGenerationCli([...base, "--out", join(dir, "o"), "--game-processes", "2", "--policy-workers", "2"], () => undefined, source, "play.ts")).toBe(0);
    const starter = seen[0] as PopulationCliInput;
    expect(starter).toMatchObject({ processes: 2, policy_workers: 2, shard: { index: 0, of: 2 }, shard_script: "play.ts" });
    expect(starter.features?.statement.feature_set_hash).toBe(hash);
    expect(starter.further).toBeUndefined();
    expect(starter.forward).toEqual(expect.arrayContaining(["--config", file, "--member", "--policy-workers", "2"]));
    expect(starter.forward).not.toContain("--out");
    expect((await (starter.load_pool as NonNullable<typeof starter.load_pool>)({ read: undefined })).teams.length).toBeGreaterThan(0);
    expect(loaded).toBe(1);
    // Past the sanity bound of game processes x workers x games in flight, it warns and plays on.
    const warn = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli([...base, "--out", join(dir, "o3"), "--game-processes", "64", "--policy-workers", "64"], () => undefined, source, "play.ts")).toBe(0);
      expect(warn.mock.calls.flat().join(" ")).toMatch(/oversubscribed/);
    } finally {
      warn.mockRestore();
    }
    // A further game process.
    const pool = join(dir, "pool.json");
    expect(await runGenerationCli([...base, "--out", join(dir, "o2"), "--shard", "1/2", "--episodes-dir", join(dir, "e"), "--features-dir", join(dir, "f"), "--pool-file", pool], () => undefined, source, "play.ts")).toBe(0);
    expect(seen[2]).toMatchObject({ processes: 1, shard: { index: 1, of: 2 }, further: true, episodes_dir: join(dir, "e"), features_dir: join(dir, "f"), pool_file: pool });
  }, 120_000);
});
