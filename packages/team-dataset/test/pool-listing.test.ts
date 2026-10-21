import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Json, jcs, readTrainingRunConfig, type TrainingRunConfig } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetPoolDetail, datasetTeamPool } from "../src/loop-pool.ts";
import { listPool } from "../src/pool-listing.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const RUN = join(ROOT, "fixtures/loop/inspector-run");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const datasetConfig = (): TrainingRunConfig =>
  readTrainingRunConfig({ ...tiny, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });

describe("the pool listing the training-data inspector reads", () => {
  it("is, for the interim pool, the committed listing of the inspector's fixture run, byte for byte", () => {
    const config = readTrainingRunConfig(JSON.parse(readFileSync(join(RUN, "run-config.json"), "utf8")));
    expect(`${jcs(listPool(engine, config) as unknown as Json)}\n`).toBe(readFileSync(join(RUN, "pool.json"), "utf8"));
  });

  it("lists the dataset pool the loop builds: same teams in the same order, with each record's labels and paste", () => {
    const config = datasetConfig();
    const listing = listPool(engine, config);
    const pool = datasetTeamPool(engine, config);
    const detail = datasetPoolDetail(engine, config);
    expect(listing.source).toBe("dataset");
    expect(listing.dataset_hash).toBe(config.team_dataset?.dataset_hash);
    expect(listing.teams.map((team) => team.hash)).toEqual([...pool.hashes]);
    expect(listing.records_before).toBe(detail.records_before);
    expect(listing.excluded_abilities).toEqual([...config.generation.excluded_abilities]);
    for (const team of listing.teams) {
      const record = detail.records.get(team.hash);
      expect(team.paste).toBe(record?.paste);
      expect(team.canonical_team_hash).toBe(record?.canonical_team_hash);
      expect(team.labels.length).toBeGreaterThan(0);
      expect(team.split).toBe(record?.origin.split);
      expect(team.sets.length).toBe(team.paste.split("\n\n").filter((block) => block.trim() !== "").length);
    }
  });

  it("marks a set as a Mega Stone holder only when the engine's data says its item is one", () => {
    const sets = listPool(engine, datasetConfig()).teams.flatMap((team) => team.sets);
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(FORMAT).mod);
    expect(sets.some((set) => set.mega_stone)).toBe(true);
    expect(sets.some((set) => !set.mega_stone)).toBe(true);
    for (const set of sets) expect(set.mega_stone).toBe(dex.items.get(set.item).megaStone !== undefined);
  });

  it("applies the config's excluded abilities, and counts the pool before them", () => {
    const config = readTrainingRunConfig(JSON.parse(readFileSync(join(RUN, "run-config.json"), "utf8")));
    const listing = listPool(engine, config);
    expect(listing.excluded_abilities).toEqual(["Illusion"]);
    expect(listing.records_before).toBeGreaterThan(listing.teams.length - 1);
    expect(listing.teams.every((team) => team.sets.every((set) => set.ability !== "Illusion"))).toBe(true);
  });
});
