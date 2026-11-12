// The single-lineage flow of `runGenerationCli` in process, with the worker client and the play functions stubbed: a game
// process that starts further ones, a further one, and the output lines each prints.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/engine.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

vi.mock("../src/policy-worker.ts", async (original) => ({
  ...(await original<typeof import("../src/policy-worker.ts")>()),
  PolicyWorkerClient: class {
    start = (): Promise<void> => Promise.resolve();
    close = (): Promise<void> => Promise.resolve();
    peakRssBytes = (): number => 1234;
  },
}));
/** Makes the named setup function throw something that is no refusal. */
const hooks = { failing: "" };
const unexpected = (name: string): void => {
  if (hooks.failing === name) throw new TypeError(`${name} broke`);
};
vi.mock("../src/league-generation.ts", async (original) => {
  const real = await original<typeof import("../src/league-generation.ts")>();
  return {
  ...real,
  featureSetup: (...args: Parameters<typeof real.featureSetup>) => (unexpected("features"), real.featureSetup(...args)),
  sheetSetup: (...args: Parameters<typeof real.sheetSetup>) => (unexpected("sheet"), real.sheetSetup(...args)),
  labelSetup: (...args: Parameters<typeof real.labelSetup>) => (unexpected("labels"), real.labelSetup(...args)),
  playGeneration: () => Promise.resolve({ training_set_hash: "t".repeat(64), set: { examples: 7 } }),
  playEntries: () => Promise.resolve([{ game: 1, row: {}, episodes: [] }]),
  writeGeneration: () => undefined,
  };
});
vi.mock("../src/parent-link.ts", async (original) => ({ ...(await original<typeof import("../src/parent-link.ts")>()), exitWhenParentDies: () => Promise.resolve("d".repeat(64)) }));
const { runGenerationCli } = await import("../src/generation-cli.ts");
const { featureSetup, trainingTeamPool } = await import("../src/league-generation.ts");

const dir = mkdtempSync(join(tmpdir(), "areneon-cli-single-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("the single-lineage flow of play-generation, in process", () => {
  const file = join(dir, "tiny.json");
  writeFileSync(file, readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8"));
  readTrainingRunConfig(JSON.parse(readFileSync(file, "utf8")));
  const base = (out: string): string[] => ["--config", file, "--generation", "1", "--policy-id", `policy:${"a".repeat(64)}`, "--preview", "p".repeat(64), "--battle", "b".repeat(64), "--run-dir", dir, "--engine-data", join(GOLDEN_DIR, "engine-data.json"), "--encoder-version", "enc1:x", "--worker-python", "none", "--out", out];
  const source = (engine: Parameters<typeof trainingTeamPool>[0]) => trainingTeamPool(engine);

  it("plays a generation, writes it and prints the summary and the peak memory", async () => {
    const lines: string[] = [];
    expect(await runGenerationCli(base(join(dir, "o1")), (line) => lines.push(line), source)).toBe(0);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ generation: 1, training_set_hash: "t".repeat(64), examples: 7 });
    expect(JSON.parse(lines[1] as string).resource_usage).toMatchObject({ policy_worker_peak_rss_bytes: 1234 });
  }, 120_000);

  it("plays a further game process's slice and leaves its entries and its peak memory", async () => {
    const out = join(dir, "o2");
    expect(await runGenerationCli([...base(out), "--shard", "1/2", "--episodes-dir", join(dir, "e")], () => undefined, source, "play.ts")).toBe(0);
    expect(readFileSync(join(out, "shard.jsonl"), "utf8")).toContain('"game":1');
    expect(JSON.parse(readFileSync(join(out, "usage.json"), "utf8"))).toHaveProperty("policy_worker_peak_rss_bytes", 1234);
  }, 120_000);

  it("starts a further game process with the flags it needs and merges what it leaves", async () => {
    const script = join(dir, "fake-shard.mjs");
    writeFileSync(script, `import { mkdirSync, writeFileSync } from "node:fs"; const out = process.argv[process.argv.indexOf("--out") + 1]; mkdirSync(out, { recursive: true }); writeFileSync(out + "/shard.jsonl", JSON.stringify({ game: 1, row: {}, episodes: [] }) + "\\n"); process.stdin.resume(); setTimeout(() => process.exit(0), 50);`);
    const lines: string[] = [];
    expect(await runGenerationCli([...base(join(dir, "o3")), "--game-processes", "2"], (line) => lines.push(line), source, script)).toBe(0);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ examples: 7 });
  }, 120_000);

  it("times its phases when asked to", async () => {
    const lines: string[] = [];
    expect(await runGenerationCli([...base(join(dir, "o4")), "--profile"], (line) => lines.push(line), source)).toBe(0);
    expect(lines.length).toBeGreaterThan(1);
  }, 120_000);

  it("refuses a further game process whose starter's pool file is not there", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      expect(await runGenerationCli([...base(join(dir, "o5")), "--shard", "1/2", "--pool-file", join(dir, "missing-pool.json")], () => undefined, source, "play.ts")).toBe(1);
      expect(error.mock.calls.flat().join(" ")).toMatch(/shared team pool/);
    } finally {
      error.mockRestore();
    }
  }, 120_000);

  it("rethrows what is no refusal from the feature, sheet and label setups", async () => {
    for (const name of ["features", "sheet", "labels"]) {
      hooks.failing = name;
      await expect(runGenerationCli(base(join(dir, `o-${name}`)), () => undefined, source)).rejects.toThrow(`${name} broke`);
    }
    hooks.failing = "";
  }, 120_000);

  it("plays with the candidate feature set the config names, in a process of its own and in a further one", async () => {
    const raw = JSON.parse(readFileSync(file, "utf8")) as { model: { encoder: { options?: unknown } } };
    const set = (hash: string): void => void (raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } });
    set("0".repeat(64));
    let hash = "";
    try {
      featureSetup(loadEngine(), readTrainingRunConfig(raw));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    set(hash);
    const featured = join(dir, "featured.json");
    writeFileSync(featured, JSON.stringify(raw));
    const args = (out: string): string[] => ["--config", featured, ...base(out).slice(2)];
    const lines: string[] = [];
    expect(await runGenerationCli(args(join(dir, "o6")), (line) => lines.push(line), source)).toBe(0);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ examples: 7 });
    expect(await runGenerationCli([...args(join(dir, "o7")), "--shard", "1/2"], () => undefined, source, "play.ts")).toBe(0);
  }, 120_000);
});
