import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { importPaste, ProgressError, readTrainingRunConfig, resolveFormat, teamHash, validateTeam } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { MANIFEST_FILE, teamsFile } from "../src/dataset-reader.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "progress-pool-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const configFor = (dir: string, change: (c: Record<string, unknown>) => void = () => undefined) => {
  const raw = structuredClone(tiny);
  (raw.generation as Record<string, unknown>).excluded_abilities = [];
  raw.team_dataset = { path: dir, dataset_hash: verifiedDatasetHash(engine, dir, FORMAT) };
  change(raw);
  return readTrainingRunConfig(raw);
};
const format = resolveFormat(engine, FORMAT);
const pastes = (file: string): { paste: string; canonical_team_hash: string }[] =>
  readFileSync(file, "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l) as { paste: string; canonical_team_hash: string });
const playedHash = (paste: string): string => {
  const team = importPaste(engine, paste);
  validateTeam(engine, format, team);
  return teamHash(team);
};

describe("the progress evaluation's validation pool", () => {
  const config = configFor(SOURCE);
  const pool = datasetValidationPool(engine, config);
  const validation = pastes(join(SOURCE, teamsFile("validation")));

  it("holds the validation teams only, sorted by canonical hash, each proven synthetic", () => {
    expect(pool.dataset_hash).toBe(config.team_dataset?.dataset_hash);
    expect(pool.teams.map((t) => t.canonical_team_hash)).toEqual(validation.map((r) => r.canonical_team_hash).sort());
    for (const team of pool.teams) expect(pool.known_synthetic.has(teamHash(team.sets))).toBe(true);
    expect(pool.known_synthetic.size).toBe(pool.teams.length);
  });

  it("shares no team with the training pool of the same dataset", () => {
    const training = datasetTeamPool(engine, config);
    const mine = new Set(pool.teams.map((t) => teamHash(t.sets)));
    for (const hash of training.hashes) expect(mine.has(hash)).toBe(false);
    for (const record of pastes(join(SOURCE, teamsFile("dev")))) expect(mine.has(playedHash(record.paste))).toBe(false);
  });

  it("leaves out teams with an excluded ability, and refuses a pool that is empty after it", () => {
    const ability = (pool.teams[0]?.sets ?? [])[0]?.ability as string;
    const fewer = datasetValidationPool(engine, configFor(SOURCE, (c) => ((c.generation as Record<string, unknown>).excluded_abilities = [ability])));
    expect(fewer.teams.length).toBeLessThan(pool.teams.length);
    expect(fewer.teams.every((t) => t.sets.every((set) => set.ability !== ability))).toBe(true);
    const every = [...new Set(pool.teams.flatMap((t) => t.sets.map((set) => set.ability)))];
    expect(() => datasetValidationPool(engine, configFor(SOURCE, (c) => ((c.generation as Record<string, unknown>).excluded_abilities = every)))).toThrow(ProgressError);
  });

  it("refuses a config without a team dataset and a dataset whose hash is not the one named", () => {
    expect(() => datasetValidationPool(engine, readTrainingRunConfig(tiny))).toThrow(/names no team dataset/);
    const wrong = readTrainingRunConfig({ ...tiny, team_dataset: { path: SOURCE, dataset_hash: "0".repeat(64) } });
    expect(() => datasetValidationPool(engine, wrong)).toThrow(/hash is not the one/);
  });

  it("refuses a dataset directory that was changed", () => {
    const dir = join(scratch, "forged");
    cpSync(SOURCE, dir, { recursive: true });
    writeFileSync(join(dir, MANIFEST_FILE), "{}");
    expect(() => datasetValidationPool(engine, readTrainingRunConfig({ ...tiny, team_dataset: { path: dir, dataset_hash: "0".repeat(64) } }))).toThrow();
  });
});
