import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  canonicalTeamHash,
  exportPaste,
  importPaste,
  jcs,
  type PokemonSet,
  stampTeam,
  type TeamRecord,
  teamHash,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import {
  COMMITTED_SPLITS,
  DatasetReadError,
  datasetSyntheticHashes,
  drawTrainingTeam,
  HASHES_FILE,
  MANIFEST_FILE,
  readDataset,
  teamsFile,
  trainingPool,
  type TrainingPool,
  type VerifiedDataset,
} from "../src/dataset-reader.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";
import { cellOf, labelTeam } from "../src/labeller.ts";
import { engine, FORMAT } from "./support.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;
const inputs = loadFixtureInputs(engine, FORMAT);
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "dataset-reader-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

let n = 0;
/** A copy of the committed dataset that `change` may edit file by file. */
function copy(change: (dir: string) => void): string {
  const dir = join(scratch, `d${n++}`);
  cpSync(SOURCE, dir, { recursive: true });
  change(dir);
  return dir;
}
const lines = (dir: string, split: string): string[] => readFileSync(join(dir, teamsFile(split)), "utf8").split("\n").filter((l) => l !== "");
const putLines = (dir: string, split: string, rows: string[]): void => writeFileSync(join(dir, teamsFile(split)), rows.map((r) => `${r}\n`).join(""));
const edit = (dir: string, file: string, change: (value: Json) => void): void => {
  const value = JSON.parse(readFileSync(join(dir, file), "utf8"));
  change(value);
  writeFileSync(join(dir, file), `${JSON.stringify(value, null, 2)}\n`);
};
const refused = (dir: string): string => {
  try {
    readDataset(dir, inputs);
  } catch (error) {
    expect(error).toBeInstanceOf(DatasetReadError);
    return (error as Error).message;
  }
  throw new Error("the dataset was accepted");
};
const stratumOf = (record: TeamRecord): number => {
  const cell = cellOf(inputs.labelSet, record.labels);
  const group = (JSON.parse(readFileSync(join(SOURCE, MANIFEST_FILE), "utf8")).merged_cells ?? []).find((g: { cells: number[] }) => g.cells.includes(cell));
  return group === undefined ? cell : group.stratum;
};

const dataset = readDataset(SOURCE, inputs);

describe("the committed dataset", () => {
  it("reads: every team is valid and round-trips, and the lists match the records", () => {
    expect(dataset.manifest.dataset_id).toBe("fixtures-v1");
    for (const split of COMMITTED_SPLITS) {
      expect(dataset.splits[split].length).toBe(dataset.manifest.splits.find((s) => s.split === split)?.teams);
      expect(dataset.hashes.splits[split].record_hashes).toHaveLength(dataset.splits[split].length);
    }
  });

  it("keeps dev and validation disjoint (L1) and holds only synthetic records", () => {
    const dev = new Set(dataset.splits.dev.map((r) => r.canonical_team_hash));
    expect(dataset.splits.validation.some((r) => dev.has(r.canonical_team_hash))).toBe(false);
    for (const record of [...dataset.splits.dev, ...dataset.splits.validation]) expect(record.data_class).toBe("synthetic");
  });

  it("has at least n_stratum_min teams of every stratum in each split (E1)", () => {
    for (const plan of dataset.manifest.splits) for (const cell of plan.cells) expect(cell.quota).toBeGreaterThanOrEqual(inputs.config.n_stratum_min);
  });

  it("lists a public key only", () => {
    expect(dataset.manifest.key.kind).toBe("public");
  });
});

describe("refusals", () => {
  it("a missing manifest", () => {
    expect(refused(copy((d) => rmSync(join(d, MANIFEST_FILE))))).toMatch(/manifest\.json/);
  });
  it("a manifest that is not a TeamDatasetManifest", () => {
    expect(refused(copy((d) => edit(d, MANIFEST_FILE, (m) => (m.surprise = 1))))).toMatch(/manifest\.json: invalid/);
  });
  it("another version manifest than the engine's", () => {
    expect(refused(copy((d) => edit(d, MANIFEST_FILE, (m) => (m.generator.manifest.ruleset_hash = "0".repeat(64)))))).toMatch(/version manifest/);
  });
  it("a missing teams file, or one with no final newline", () => {
    expect(refused(copy((d) => rmSync(join(d, teamsFile("dev")))))).toMatch(/teams-dev\.jsonl/);
    expect(refused(copy((d) => writeFileSync(join(d, teamsFile("dev")), lines(d, "dev").join("\n"))))).toMatch(/newline/);
  });
  it("a line that is not JSON, or not in canonical form", () => {
    expect(refused(copy((d) => putLines(d, "dev", ["{", ...lines(d, "dev").slice(1)])))).toMatch(/line 1/);
    expect(refused(copy((d) => putLines(d, "dev", [(lines(d, "dev")[0] as string).replace('","', '", "'), ...lines(d, "dev").slice(1)])))).toMatch(/canonical/);
  });
  it("a record the bridge refuses", () => {
    const d = copy((dir) => {
      const record = JSON.parse(lines(dir, "dev")[0] as string);
      record.paste = record.paste.replace("Ability:", "Ability :");
      putLines(dir, "dev", [jcs(record), ...lines(dir, "dev").slice(1)]);
    });
    expect(refused(d)).toMatch(/line 1/);
  });
  it("a record of another split or dataset, or out of order", () => {
    const first = (dir: string): Json => JSON.parse(lines(dir, "dev")[0] as string);
    const put = (dir: string, record: unknown): void => putLines(dir, "dev", [jcs(record as never), ...lines(dir, "dev").slice(1)]);
    expect(refused(copy((d) => put(d, { ...first(d), origin: { ...first(d).origin, split: "validation" } })))).toMatch(/origin is/);
    expect(refused(copy((d) => put(d, { ...first(d), origin: { ...first(d).origin, dataset_id: "other" } })))).toMatch(/origin is/);
    expect(refused(copy((d) => put(d, { ...first(d), origin: { ...first(d).origin, index: 5 } })))).toMatch(/origin index/);
  });
  it("a team the pinned validator refuses", () => {
    const illegal = importPaste(engine, readFileSync(join(DATASETS_DIR(FORMAT), "..", "illegal-restricted-legendary.txt"), "utf8"));
    const d = copy((dir) => {
      const base = JSON.parse(lines(dir, "dev")[0] as string);
      const labels = labelTeam(inputs.labelSet, illegal);
      const record = { ...base, paste: exportPaste(engine, illegal), team_hash: teamHash(illegal), canonical_team_hash: canonicalTeamHash(illegal), labels };
      putLines(dir, "dev", [jcs(record), ...lines(dir, "dev").slice(1)]);
    });
    expect(refused(d)).toMatch(/pinned validator refuses/);
  });
  it("a split with the wrong number of teams", () => {
    expect(refused(copy((d) => putLines(d, "dev", lines(d, "dev").slice(0, -1))))).toMatch(/holds 119 teams/);
  });
  it("a split whose strata do not hold their quotas", () => {
    // Replace a dev record by a validation record of another stratum.
    const d = copy((dir) => {
      const dev = lines(dir, "dev").map((l) => JSON.parse(l) as TeamRecord);
      const other = lines(dir, "validation").map((l) => JSON.parse(l) as TeamRecord).find((r) => stratumOf(r) !== stratumOf(dev[0] as TeamRecord)) as TeamRecord;
      const swapped = { ...other, origin: { ...other.origin, split: "dev", index: 0 } };
      putLines(dir, "dev", [jcs(swapped as never), ...lines(dir, "dev").slice(1)]);
    });
    expect(refused(d)).toMatch(/stratum \d+ holds/);
  });
  it("a team in both splits (L1)", () => {
    const d = copy((dir) => {
      const dev = lines(dir, "dev").map((l) => JSON.parse(l) as TeamRecord);
      const val = lines(dir, "validation").map((l) => JSON.parse(l) as TeamRecord);
      const k = val.findIndex((r) => stratumOf(r) === stratumOf(dev[0] as TeamRecord));
      const copyOfDev = { ...dev[0], origin: { ...(dev[0] as TeamRecord).origin, split: "validation", index: k } };
      putLines(dir, "validation", val.map((r, i) => jcs((i === k ? copyOfDev : r) as never)));
    });
    expect(refused(d)).toMatch(/is in dev and in validation/);
  });
  it("hashes.json that is missing, malformed, unsorted or not the records' lists", () => {
    expect(refused(copy((d) => rmSync(join(d, HASHES_FILE))))).toMatch(/hashes\.json/);
    expect(refused(copy((d) => writeFileSync(join(d, HASHES_FILE), "null\n")))).toMatch(/not an object/);
    expect(refused(copy((d) => edit(d, HASHES_FILE, (h) => delete h.splits.dev)))).toMatch(/not a list/);
    expect(refused(copy((d) => edit(d, HASHES_FILE, (h) => (h.splits.dev.record_hashes[0] = "xyz"))))).toMatch(/not a list/);
    expect(refused(copy((d) => edit(d, HASHES_FILE, (h) => h.splits.dev.record_hashes.reverse())))).toMatch(/not sorted and unique/);
    expect(refused(copy((d) => edit(d, HASHES_FILE, (h) => (h.manifest_hash = "0".repeat(64)))))).toMatch(/does not list the hashes/);
  });
});

describe("SyntheticTeamHashes from the dataset", () => {
  const known = datasetSyntheticHashes([dataset]);
  const sets = (record: TeamRecord): PokemonSet[] => importPaste(engine, record.paste);

  it("stamps a generated team synthetic", () => {
    for (const split of COMMITTED_SPLITS) {
      const team = stampTeam(sets(dataset.splits[split][0] as TeamRecord) as never, known);
      expect(team.data_class).toBe("synthetic");
    }
  });
  it("stamps a non-member private, a generated team with one changed value included", () => {
    const changed = sets(dataset.splits.dev[0] as TeamRecord);
    (changed[0] as PokemonSet).level = 51;
    expect(stampTeam(changed as never, known).data_class).toBe("private");
    expect(stampTeam([] as never, known).data_class).toBe("private");
  });
  it("holds exactly the committed teams", () => {
    expect(known.size).toBe(dataset.splits.dev.length + dataset.splits.validation.length);
  });
});

describe("the training pool", () => {
  const fresh = (): VerifiedDataset => readDataset(SOURCE, inputs);
  const validation = dataset.splits.validation;

  it("draws dev teams", () => {
    const pool = trainingPool([dataset], engine);
    const dev = dataset.splits.dev;
    expect(pool.records).toHaveLength(dev.length);
    expect(drawTrainingTeam(pool, 0)).toBe(dev[0]);
    expect(drawTrainingTeam(pool, dev.length - 1)).toBe(dev[dev.length - 1]);
  });
  it("holds no validation team", () => {
    const blocked = new Set(validation.map((r) => r.canonical_team_hash));
    expect(trainingPool([dataset], engine).records.some((r) => blocked.has(r.canonical_team_hash) || r.origin.split !== "dev")).toBe(false);
  });
  it("refuses a dataset readDataset did not return (a validation record relabelled training)", () => {
    const v = validation[0] as TeamRecord;
    const forged = { ...dataset, splits: { dev: [{ ...v, origin: { ...v.origin, split: "training" } }], validation: [] } } as unknown as VerifiedDataset;
    expect(() => trainingPool([forged], engine)).toThrow(/did not return/);
  });
  it("recomputes hashes from the paste: a validation team under a forged hash is still refused", () => {
    const own = fresh();
    const v = validation[0] as TeamRecord;
    own.splits.dev[0] = { ...v, canonical_team_hash: "0".repeat(64), origin: { ...v.origin, split: "dev" } };
    expect(() => trainingPool([own], engine)).toThrow(/duplicates a validation team/);
  });
  it("refuses a validation record relabelled dev, and a dev team twice", () => {
    const own = fresh();
    const v = validation[0] as TeamRecord;
    own.splits.dev[0] = { ...v, origin: { ...v.origin, split: "validation" } };
    expect(() => trainingPool([own], engine)).toThrow(/not a dev record/);
    const twice = fresh();
    twice.splits.dev[1] = twice.splits.dev[0] as TeamRecord;
    expect(() => trainingPool([twice], engine)).toThrow(/twice/);
  });
  it("refuses no dataset and an empty pool", () => {
    expect(() => trainingPool([], engine)).toThrow(/no dataset/);
    const own = fresh();
    own.splits.dev.length = 0;
    expect(() => trainingPool([own], engine)).toThrow(/empty/);
  });
  it("draws only from a pool trainingPool made, and only inside it", () => {
    const pool = trainingPool([dataset], engine);
    for (const bad of [-1, pool.records.length, 1.5, Number.NaN]) expect(() => drawTrainingTeam(pool, bad)).toThrow(/outside the pool/);
    const v = validation[0] as TeamRecord;
    const handBuilt = { records: [{ ...v, origin: { ...v.origin, split: "dev" } }] } as unknown as TrainingPool;
    expect(() => drawTrainingTeam(handBuilt, 0)).toThrow(/did not make/);
  });
});

describe("what readDataset recomputes from its context", () => {
  it("refuses another vocabulary or reference pool", () => {
    expect(() => readDataset(SOURCE, { ...inputs, vocabulary: { ...inputs.vocabulary, roster: inputs.vocabulary.roster.slice(1) } })).toThrow(/vocabulary_hash/);
    expect(() => readDataset(SOURCE, { ...inputs, referenceHashes: ["1".repeat(64)] })).toThrow(/reference_pool_hash/);
  });
});
