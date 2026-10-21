import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  actorRule,
  GenerationError,
  playGeneration,
  type PolicyClient,
  readTrainingRunConfig,
  stampTeam,
  importPaste,
  resolveFormat,
  teamHash,
  type TrainingRunConfig,
  validateTeam,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { MANIFEST_FILE, readDataset, teamsFile } from "../src/dataset-reader.ts";
import { writeDataset } from "../src/dataset-io.ts";
import { DATASETS_DIR, loadFixtureInputs, readFixtureRun, recordedPilot } from "../src/fixtures.ts";
import { generate } from "../src/generate.ts";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { datasetPoolSource, datasetTeamPool } from "../src/loop-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const inputs = loadFixtureInputs(engine, FORMAT);
const scratch = mkdtempSync(join(tmpdir(), "loop-pool-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const format = resolveFormat(engine, FORMAT);
/** The hash of the team a battle starts with: the paste's sets as the validator normalises them. */
const playedHash = (paste: string): string => {
  const team = importPaste(engine, paste);
  validateTeam(engine, format, team);
  return teamHash(team);
};
const pastes = (file: string): string[] => readFileSync(file, "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l).paste as string);
const manifestHash = (dir: string): string => readDataset(dir, inputs).manifest_hash;
const configFor = (dir: string, change: (c: Record<string, unknown>) => void = () => undefined): TrainingRunConfig => {
  const raw = structuredClone(tiny);
  (raw.generation as Record<string, unknown>).excluded_abilities = [];
  raw.team_dataset = { path: dir, manifest_hash: manifestHash(dir) };
  change(raw);
  return readTrainingRunConfig(raw);
};

/** The dataset directory `generate` writes (records.<split>.jsonl for dev, validation and training), made from the committed fixture run. */
function generatedDataset(): string {
  const dir = join(scratch, "generated");
  mkdirSync(dir);
  const run = readFixtureRun(SOURCE);
  const result = generate({
    engine, format: inputs.format, table: inputs.table, vocabulary: inputs.vocabulary, labelSet: inputs.labelSet, config: inputs.config,
    generationKey: run.generation_key, assignmentKey: run.assignment_key, datasetId: run.dataset_id, planId: run.dataset_id, sizes: run.sizes,
    generatorCommit: run.generator_commit, reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, pilot: recordedPilot(inputs, SOURCE), logDir: dir,
  });
  writeDataset(dir, result);
  return dir;
}

describe("the committed fixtures as a team pool", () => {
  const config = configFor(SOURCE);
  const pool = datasetTeamPool(engine, config);
  const dev = pastes(join(SOURCE, teamsFile("dev"))).map(playedHash);
  const validation = pastes(join(SOURCE, teamsFile("validation"))).map(playedHash);

  it("holds the dev teams only, sorted by team hash, each proven synthetic", () => {
    expect(pool.hashes).toEqual([...new Set(dev)].sort());
    expect(pool.teams.map(teamHash)).toEqual([...pool.hashes]);
    for (const hash of pool.hashes) expect(pool.known_synthetic.has(hash)).toBe(true);
  });

  it("proves nothing else synthetic: not a validation team, not a team outside the dataset", () => {
    for (const hash of validation) expect(pool.known_synthetic.has(hash)).toBe(false);
    expect(pool.known_synthetic.size).toBe(pool.hashes.length);
    // A team outside the pool (here a validation team, as a stand-in for any other) is stamped private, not synthetic.
    const paste = JSON.parse(readFileSync(join(SOURCE, teamsFile("validation")), "utf8").split("\n")[0] as string).paste as string;
    expect(stampTeam(importPaste(engine, paste) as never, pool.known_synthetic).data_class).toBe("private");
    expect(stampTeam(pool.teams[0] as never, pool.known_synthetic).data_class).toBe("synthetic");
  });

  it("leaves out teams with an excluded ability, and refuses a pool that is empty after it", () => {
    const ability = (pool.teams[0] ?? [])[0]?.ability as string;
    const fewer = datasetTeamPool(engine, configFor(SOURCE, (c) => ((c.generation as Record<string, unknown>).excluded_abilities = [ability])));
    expect(fewer.hashes.length).toBeLessThan(pool.hashes.length);
    expect(fewer.teams.every((team) => team.every((set) => set.ability !== ability))).toBe(true);
    const every = [...new Set(pool.teams.flatMap((team) => team.map((set) => set.ability)))];
    expect(() => datasetTeamPool(engine, configFor(SOURCE, (c) => ((c.generation as Record<string, unknown>).excluded_abilities = every)))).toThrow(GenerationError);
  });

  it("refuses a dataset whose manifest is not the one the config names, or a config that names none", () => {
    const wrong = readTrainingRunConfig({ ...tiny, team_dataset: { path: SOURCE, manifest_hash: "0".repeat(64) } });
    expect(() => datasetTeamPool(engine, wrong)).toThrow(/manifest hash/);
    expect(() => datasetTeamPool(engine, readTrainingRunConfig(tiny))).toThrow(/names no team dataset/);
  });

  it("refuses a forged dataset directory: a validation team copied into dev", () => {
    const dir = join(scratch, "forged");
    cpSync(SOURCE, dir, { recursive: true });
    const first = readFileSync(join(SOURCE, teamsFile("validation")), "utf8").split("\n")[0] as string;
    const dev = readFileSync(join(SOURCE, teamsFile("dev")), "utf8").split("\n").filter((l) => l !== "");
    writeFileSync(join(dir, teamsFile("dev")), [first, ...dev.slice(1)].map((l) => `${l}\n`).join(""));
    expect(() => readDataset(dir, inputs)).toThrow();
    writeFileSync(join(dir, MANIFEST_FILE), "{}");
    expect(() => datasetTeamPool(engine, readTrainingRunConfig({ ...tiny, team_dataset: { path: dir, manifest_hash: "0".repeat(64) } }))).toThrow();
  });

  it("is the port's choice only when the config names a dataset", () => {
    const interim = () => ({ teams: [], hashes: [], known_synthetic: new Set<string>() });
    expect(datasetPoolSource(interim)(engine, readTrainingRunConfig(tiny))).toEqual(interim());
    expect(datasetPoolSource(interim)(engine, config).hashes).toEqual(pool.hashes);
  });
});

describe("a generated dataset directory with a training split", () => {
  const dir = generatedDataset();
  const config = configFor(dir);
  const pool = datasetTeamPool(engine, config);
  const lines = (split: string): string[] => pastes(join(dir, `records.${split}.jsonl`)).map(playedHash);

  it("pools the dev and training teams and never the validation teams", () => {
    expect(new Set(pool.hashes)).toEqual(new Set([...lines("dev"), ...lines("training")]));
    expect(lines("training").length).toBeGreaterThan(0);
    for (const hash of lines("validation")) expect(pool.known_synthetic.has(hash)).toBe(false);
  });

  it("plays a generation on those teams and records them", async () => {
    const client = {
      beginGame: () => Promise.resolve(),
      ask: (_battle, query) =>
        Promise.resolve({
          schema_version: 1 as const,
          identity: query.identity,
          choice: actorRule({ choices: query.candidates.filter((c) => !c.pending).map((c) => c.choice), pending: query.candidates.filter((c) => c.pending).map((c) => c.choice) }),
          source_policy_id: query.policy_id,
          elapsed_ms: 0,
          deadline_outcome: "in_budget" as const,
        }),
    } satisfies PolicyClient & { beginGame: () => Promise<void> };
    const small = readTrainingRunConfig({ ...config, generation: { ...config.generation, games: 2, baseline_games: [] } });
    const result = await playGeneration({ engine, config: small, generation: 1, policy_id: `policy:${"c".repeat(64)}`, client, pool });
    expect(result.team_hashes.length).toBeGreaterThan(0);
    for (const hash of result.team_hashes) expect(pool.hashes).toContain(hash);
    expect(result.episodes.every((e) => e.data_class === "synthetic")).toBe(true);
  });
});
