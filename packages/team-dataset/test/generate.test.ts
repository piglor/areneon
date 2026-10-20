import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  canonicalTeamHash,
  importPaste,
  readTeamDatasetManifest,
  readTeamRecord,
  type TeamRecord,
  teamVocabularyHash,
  recordHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { recordHashes, recordsFile, writeDataset } from "../src/dataset-io.ts";
import { KnownHashes, referencePoolHash, sortedUnique } from "../src/dedupe.ts";
import { checkRecord, generate, GenerateError, type GenerateInput } from "../src/generate.ts";
import { cellOf } from "../src/labeller.ts";
import { type Pilot, runPilot } from "../src/pilot.ts";
import { ASSIGNMENT_KEY, config, drawerFor, engine, format, KEY, light as labelSet, small, table, vocabulary } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "generate-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
let counter = 0;
const fresh = (): string => {
  const dir = join(scratch, `run-${counter++}`);
  mkdtempSync(`${scratch}/x`);
  return dir;
};

const cfg = small();
const identity = { config_hash: recordHash(cfg), label_set_hash: recordHash(labelSet), vocabulary_hash: teamVocabularyHash(vocabulary) };
const pilot: Pilot = runPilot(drawerFor(cfg, KEY, labelSet), identity, labelSet, cfg.pilot_draws);
const empty = { hash: referencePoolHash([]), known: new KnownHashes([]) };
const sizes = { dev: 2, validation: 2, training: 5 };

function input(over: Partial<GenerateInput> = {}): GenerateInput {
  const logDir = join(scratch, `log-${counter++}`);
  mkdtempSync(`${scratch}/y`);
  return {
    engine,
    format,
    table,
    vocabulary,
    labelSet,
    config: cfg,
    generationKey: KEY,
    assignmentKey: ASSIGNMENT_KEY,
    datasetId: "test-1",
    planId: "test-plan-1",
    sizes,
    generatorCommit: "a".repeat(40),
    reference: empty,
    pilot,
    logDir: prepared(logDir),
    ...over,
  };
}
const prepared = (dir: string): string => (mkdirSync(dir, { recursive: true }), dir);

const run = generate(input());
const sha256 = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");

describe("a generated dataset (ADR-013 section 2)", () => {
  it("holds teams that all pass the pinned validator and the paste round trip, as records the bridge's reader accepts", () => {
    const all = Object.values(run.records).flat() as TeamRecord[];
    expect(all.length).toBe(run.manifest.splits.reduce((n, s) => n + s.teams, 0));
    for (const record of all) {
      readTeamRecord(engine, record, labelSet);
      const sets = importPaste(engine, record.paste);
      expect(canonicalTeamHash(sets)).toBe(record.canonical_team_hash);
      // the validator normalises the sets it is given in place
      expect(validateTeam(engine, format, sets).valid).toBe(true);
      expect(record.data_class).toBe("synthetic");
    }
    expect(new Set(all.map((r) => r.canonical_team_hash)).size).toBe(all.length);
  });

  it("numbers each split's teams in order and keeps the split, dataset and draw counter in the origin", () => {
    for (const split of run.plan.splits) {
      const records = run.records[split] as TeamRecord[];
      expect(records.map((r) => r.origin.index)).toEqual(records.map((_, k) => k));
      expect(records.every((r) => r.origin.split === split && r.origin.dataset_id === "test-1")).toBe(true);
      expect(records.every((r) => Number.isSafeInteger(r.origin.draw_counter) && r.origin.draw_counter >= 0)).toBe(true);
    }
  });

  it("fills every stratum's quota in every split, with at least n_stratum_min", () => {
    for (const split of run.plan.splits) {
      for (const stratum of run.plan.strata) {
        const held = (run.records[split] as TeamRecord[]).filter((r) => run.plan.stratumOf[cellOf(labelSet, r.labels)] === stratum.stratum);
        expect(held).toHaveLength(stratum.quotas[split] as number);
        expect(held.length).toBeGreaterThanOrEqual(cfg.n_stratum_min);
      }
    }
  });

  it("records a manifest the reader accepts against its config and label set, with the D20 merge and the pilot counts", () => {
    const { manifest } = run;
    expect(readTeamDatasetManifest(manifest, { config: cfg, labelSet })).toEqual(manifest);
    expect(manifest).toMatchObject({ data_class: "synthetic", dataset_id: "test-1", plan_id: "test-plan-1", d_max: cfg.d_max, pilot_draws: cfg.pilot_draws });
    expect(manifest.generator).toMatchObject({ generator_commit: "a".repeat(40), config_hash: identity.config_hash, label_set_hash: identity.label_set_hash, vocabulary_hash: identity.vocabulary_hash, manifest: format.manifest });
    expect(manifest.key).toEqual({ kind: "public", generation_key: KEY, assignment_key: ASSIGNMENT_KEY });
    expect(manifest.reference_pool_hash).toBe(empty.hash);
    for (const plan of manifest.splits) expect(plan.cells.reduce((n, c) => n + c.pilot_hits, 0)).toBe(pilot.cell_hits.reduce((a, b) => a + b, 0));
  });

  it("logs every rejection, duplicate and overflow, and names the log by its hash", () => {
    const { rejection_log: log } = run.manifest;
    expect(sha256(run.logPath)).toBe(log.log_sha256);
    const lines = readFileSync(run.logPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { k: number; reason: string; canonical_team_hash: string; stratum?: number; detail?: string });
    expect(lines).toHaveLength(log.rejections + log.duplicates + log.overflows);
    expect(lines.filter((l) => l.reason === "overflow")).toHaveLength(log.overflows);
    expect(log.overflows).toBeGreaterThan(0);
    expect(log.rejections).toBeGreaterThan(0);
    expect(lines.filter((l) => l.reason === "invalid").every((l) => typeof l.detail === "string")).toBe(true);
    expect(lines.filter((l) => l.reason === "overflow").every((l) => l.stratum !== undefined)).toBe(true);
    expect(lines.map((l) => l.k)).toEqual([...lines.map((l) => l.k)].sort((a, b) => a - b));
    expect(log.draws).toBe(run.coverage.stream.draws);
    expect(log.draws).toBeLessThanOrEqual(cfg.d_max);
  });

  it("reports the coverage: every cell, the pilot counts and the merged strata, per split", () => {
    const { coverage } = run;
    expect(coverage.cells).toHaveLength(8);
    expect(coverage.cells[0]).toMatchObject({ cell: 0, values: ["yes", "yes", "yes"] });
    expect(coverage.cells.reduce((n, c) => n + c.pilot_hits, 0) + coverage.pilot.rejected).toBe(cfg.pilot_draws);
    expect(coverage.labels.map((l) => l.label_id)).toEqual(labelSet.labels.map((l) => l.label_id));
    expect(coverage.strata.map((s) => s.stratum)).toEqual(run.plan.strata.map((s) => s.stratum));
    expect(coverage.splits.map((s) => s.split)).toEqual(["dev", "validation", "training"]);
    expect(coverage.stream.accepted).toBeGreaterThan(coverage.stream.draws * 0.8);
    expect(coverage.provisioning.provisioned_draws).toBeGreaterThanOrEqual(coverage.provisioning.planned_draws);
    expect(coverage.dedupe).toMatchObject({ reference_hashes: 0, prefix_bytes: 0, exact_checks: 0 });
    expect(coverage.dedupe.prefix_misses).toBeGreaterThan(0);
    expect(coverage.attempts).toEqual([{ plan_id: "test-plan-1", dataset_id: "test-1", short: [] }]);
  });
});

describe("the last gate on a record", () => {
  it("accepts the generated records and refuses one whose labels are not the labeller's output", () => {
    const record = (run.records.dev as TeamRecord[])[0] as TeamRecord;
    expect(() => checkRecord(engine, record, labelSet)).not.toThrow();
    // a value the label set allows, but not the one the labeller gives this team
    const first = record.labels[0] as { label_id: string; value: string };
    const other = first.value === "yes" ? "no" : "yes";
    const wrong = { ...record, labels: [{ label_id: first.label_id, value: other }, ...record.labels.slice(1)] };
    expect(() => checkRecord(engine, wrong, labelSet)).toThrow(GenerateError);
    expect(() => checkRecord(engine, wrong, labelSet)).toThrow(/not the labeller's output/);
  });
});

describe("determinism", () => {
  it("gives byte-identical files from the same inputs, and different teams from another key", () => {
    const [a, b, c] = [fresh(), fresh(), fresh()].map((dir) => prepared(dir));
    const first = generate(input({ logDir: a as string }));
    const second = generate(input({ logDir: b as string }));
    const other = generate(input({ logDir: c as string, generationKey: "33".repeat(32) }));
    const files = writeDataset(a as string, first);
    expect(writeDataset(b as string, second)).toEqual(files);
    writeDataset(c as string, other);
    expect(files).toEqual(["records.dev.jsonl", "records.validation.jsonl", "records.training.jsonl", "manifest.json", "coverage.json", "pilot.json", "rejections.jsonl"]);
    for (const file of files) expect(readFileSync(join(b as string, file))).toEqual(readFileSync(join(a as string, file)));
    expect(readFileSync(join(c as string, "records.dev.jsonl"), "utf8")).not.toBe(readFileSync(join(a as string, "records.dev.jsonl"), "utf8"));
  });

  it("runs the config's pilot when none is given, and gets the pilot a separate run recorded", () => {
    const { pilot: _recorded, ...unrecorded } = input();
    const withPilot = generate(unrecorded);
    expect(withPilot.pilotDrawsRun).toBe(cfg.pilot_draws);
    expect(run.pilotDrawsRun).toBe(0);
    expect(withPilot.pilot).toEqual(pilot);
    expect(withPilot.manifest).toEqual(run.manifest);
  });
});

describe("dedupe across datasets", () => {
  it("skips every team of an earlier dataset, logs them as duplicates, and keeps the rest of the stream order", () => {
    const earlier = sortedUnique(Object.values(run.records).flat().map((r) => (r as TeamRecord).canonical_team_hash));
    const known = new KnownHashes(earlier);
    // the same plan id gives the same stream: every team the first run binned is now a known duplicate
    const again = generate(input({ reference: { hash: referencePoolHash(earlier), known } }));
    const hashes = Object.values(again.records).flat().map((r) => (r as TeamRecord).canonical_team_hash);
    expect(hashes.some((h) => earlier.includes(h))).toBe(false);
    expect(again.manifest.rejection_log.duplicates).toBeGreaterThanOrEqual(earlier.length);
    expect(again.manifest.reference_pool_hash).toBe(referencePoolHash(earlier));
    expect(again.coverage.dedupe.reference_hashes).toBe(earlier.length);
    expect(again.coverage.dedupe.exact_checks).toBeGreaterThanOrEqual(earlier.length);
    expect(again.coverage.dedupe.prefix_bytes).toBe(8 * earlier.length);
    readTeamDatasetManifest(again.manifest, { config: cfg, labelSet });
  });

  it("skips a repeat within the run", () => {
    // with a one-stratum label space every candidate competes for the same bin; a repeated team would be binned twice without the in-run check
    const hashes = Object.values(run.records).flat().map((r) => (r as TeamRecord).canonical_team_hash);
    expect(new Set(hashes).size).toBe(hashes.length);
  });
});

describe("the dataset directory", () => {
  it("is read back: records parse, the manifest reads, and the hashes feed a later dedupe", () => {
    const dir = prepared(fresh());
    const result = generate(input({ logDir: dir }));
    writeDataset(dir, result);
    expect(readdirSync(dir).sort()).toEqual(["coverage.json", "manifest.json", "pilot.json", "records.dev.jsonl", "records.training.jsonl", "records.validation.jsonl", "rejections.jsonl"]);
    expect(readTeamDatasetManifest(JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")), { config: cfg, labelSet }).dataset_id).toBe("test-1");
    const hashes = recordHashes([join(dir, recordsFile("dev")), join(dir, recordsFile("training"))]);
    expect(hashes).toHaveLength((result.records.dev?.length ?? 0) + (result.records.training?.length ?? 0));
    const bare = join(dir, "bare.jsonl");
    writeFileSync(bare, '{"paste":"x"}\n');
    expect(() => recordHashes([bare])).toThrow(/line 1 has no canonical_team_hash/);
  });

  it("writes only the splits that were asked for", () => {
    const dir = prepared(fresh());
    const result = generate(input({ logDir: dir, sizes: { training: 8 } }));
    expect(writeDataset(dir, result)).toEqual(["records.training.jsonl", "manifest.json", "coverage.json", "pilot.json", "rejections.jsonl"]);
    expect(result.manifest.splits.map((s) => s.split)).toEqual(["training"]);
  });
});

describe("a shortfall re-plans with one more merge under a fresh plan id (ADR-013 section 2)", () => {
  /** A pilot that claims every cell is equally common is wrong about the proposal: the rare cells never fill. */
  const lyingPilot = (forConfig: typeof cfg): Pilot => {
    const each = Math.floor(forConfig.pilot_draws / 8);
    return { ...pilot, config_hash: recordHash(forConfig), cell_hits: new Array<number>(8).fill(each), rejected: forConfig.pilot_draws - 8 * each };
  };

  it("coarsens the short strata, logs the failed attempt and finishes under <plan_id>-r<n>", () => {
    const dir = prepared(fresh());
    const lying = small({ d_max: 1500 });
    const result = generate(input({ logDir: dir, pilot: lyingPilot(lying), config: lying, sizes: { dev: 4, validation: 2, training: 4 } }));
    expect(result.attempts.length).toBeGreaterThan(1);
    expect(result.attempts[0]?.short.length).toBeGreaterThan(0);
    expect(result.attempts.at(-1)).toMatchObject({ short: [] });
    expect(result.planId).toBe(`test-plan-1-r${result.attempts.length - 1}`);
    expect(result.datasetId).toBe(`test-1-r${result.attempts.length - 1}`);
    expect(result.manifest.plan_id).toBe(result.planId);
    expect(readdirSync(dir)).toContain("rejections.attempt-0.jsonl");
    for (const record of Object.values(result.records).flat()) expect((record as TeamRecord).origin.dataset_id).toBe(result.datasetId);
  });

  it("gives up when even the coarsest plan cannot fill within d_max", () => {
    // 24 draws cannot make 24 valid teams that are all new: some draws are rejected
    const tight = small({ d_max: 24 });
    const attempt = (): unknown => generate(input({ config: tight, pilot: lyingPilot(tight), sizes: { dev: 24 } }));
    expect(attempt).toThrow(GenerateError);
    expect(attempt).toThrow(/still short after/);
  });
});

describe("generate refuses inputs that do not belong together", () => {
  it("a pilot of another length, another config, or another vocabulary", () => {
    expect(() => generate(input({ pilot: { ...pilot, draws: pilot.draws + 1 } }))).toThrow(/the pilot has 601 draws, the config says 600/);
    expect(() => generate(input({ pilot: { ...pilot, config_hash: "0".repeat(64) } }))).toThrow(/another config, label set or vocabulary/);
    expect(() => generate(input({ pilot: { ...pilot, vocabulary_hash: "0".repeat(64) } }))).toThrow(GenerateError);
  });

  it("a bad key", () => {
    expect(() => generate(input({ generationKey: "abc" }))).toThrow(/generation key/);
    expect(() => generate(input({ assignmentKey: "abc" }))).toThrow(/assignment key/);
  });

  it("reports progress as it draws", () => {
    const seen: number[] = [];
    generate(input({ progressEvery: 500, onProgress: ({ draws }) => seen.push(draws) }));
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((n) => n % 500 === 0)).toBe(true);
  });

  it("the committed config is the one the tests shrink", () => {
    expect(config.pilot_draws).toBe(40_000);
    expect(cfg.pilot_draws).toBe(600);
  });
});
