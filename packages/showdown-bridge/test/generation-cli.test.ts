import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { interimPoolSource, runGenerationCli, runManifestCli, UsageError } from "../src/generation-cli.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, readTrainingRunConfig } from "../src/index.ts";
import { trainingTeamPool } from "../src/league-generation.ts";
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
    const named = { ...tiny, team_dataset: { path: "somewhere", manifest_hash: "a".repeat(64) } };
    const file = join(dir, "named.json");
    writeFileSync(file, JSON.stringify(named));
    await expect(runGenerationCli(flags(file))).rejects.toThrow(/team-dataset\/scripts\/play-generation/);
    expect(() => interimPoolSource(loadEngine(), readTrainingRunConfig(named))).toThrow(UsageError);
    expect(interimPoolSource(loadEngine(), readTrainingRunConfig(tiny)).teams.length).toBeGreaterThan(60);
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
