import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { importPaste, planPopulation, readTrainingRunConfig, resolveFormat, teamHash, type TrainingRunConfig, validateTeam } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { teamsFile } from "../src/dataset-reader.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "population-pool-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const format = resolveFormat(engine, FORMAT);
const playedHash = (paste: string): string => {
  const team = importPaste(engine, paste);
  validateTeam(engine, format, team);
  return teamHash(team);
};
const pastes = (split: string, dir = SOURCE): string[] => readFileSync(join(dir, teamsFile(split)), "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l).paste as string);

/** The population test config (two members, no past pool) on a dataset directory. */
function configFor(dir: string): TrainingRunConfig {
  const base = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/population.json"), "utf8"));
  base.generation.excluded_abilities = [];
  base.generation.baseline_games = [{ actor: "random-legal", policy_id: "random-legal@tiny", games: 2 }];
  base.team_dataset = { path: dir, dataset_hash: verifiedDatasetHash(engine, dir, FORMAT) };
  const population = base.population;
  Object.assign(population, { members: 3, seats_per_member: 8, shares: { ladder_ppm: 500_000, self_play_ppm: 500_000, past_pool_ppm: 0 } });
  population.selection.min_roots = 1;
  for (const key of ["ladder_ppm", "self_play_ppm", "past_pool_ppm"]) population.explore.bounds[key] = { min: 0, max: 1_000_000 };
  return readTrainingRunConfig(base);
}

describe("the population plan on a dataset pool", () => {
  const config = configFor(SOURCE);
  const pool = datasetTeamPool(engine, config);
  const validation = new Set(pastes("validation").map(playedHash));

  it("draws only teams of the pool, which holds no validation team", () => {
    const plan = planPopulation(config, 1, pool.teams.length);
    expect(plan.length).toBeGreaterThan(0);
    const drawn = new Set(plan.flatMap((g) => [pool.hashes[g.teams.p1] as string, pool.hashes[g.teams.p2] as string]));
    expect(drawn.size).toBeGreaterThan(1);
    for (const hash of drawn) {
      expect(pool.known_synthetic.has(hash)).toBe(true);
      expect(validation.has(hash)).toBe(false);
    }
    for (const g of plan) {
      expect(g.teams.p1).toBeLessThan(pool.teams.length);
      expect(g.teams.p2).toBeLessThan(pool.teams.length);
    }
  });

  it("cannot be given a validation team: a dataset with one copied into dev is refused before any plan exists", () => {
    const dir = join(scratch, "forged");
    cpSync(SOURCE, dir, { recursive: true });
    const first = readFileSync(join(SOURCE, teamsFile("validation")), "utf8").split("\n")[0] as string;
    const dev = readFileSync(join(SOURCE, teamsFile("dev")), "utf8").split("\n").filter((l) => l !== "");
    writeFileSync(join(dir, teamsFile("dev")), [first, ...dev.slice(1)].map((l) => `${l}\n`).join(""));
    const forged = readTrainingRunConfig({ ...config, team_dataset: { path: dir, dataset_hash: config.team_dataset?.dataset_hash } });
    expect(() => datasetTeamPool(engine, forged)).toThrow();
  });
});
