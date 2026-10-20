import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { HASHES_FILE, MANIFEST_FILE, teamsFile } from "../src/dataset-reader.ts";
import { committedDatasetHashes, COVERAGE_FILE, DATASETS_DIR, differing, FixtureError, loadFixtureInputs, PILOT_FILE, readFixtureRun, recordedPilot, regenerate, RUN_FILE, writeFixtures } from "../src/fixtures.ts";
import { engine, FORMAT } from "./support.ts";

const DIR = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "fixtures-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const inputs = loadFixtureInputs(engine, FORMAT);
const COMMITTED = [COVERAGE_FILE, HASHES_FILE, MANIFEST_FILE, PILOT_FILE, teamsFile("dev"), teamsFile("validation")];

describe("the committed fixtures", () => {
  // The generator is run again from the committed config, label set, vocabulary,
  // run record and recorded pilot; `fixtures.ts check --full` also reruns the pilot.
  const files = regenerate(inputs, readFixtureRun(DIR), recordedPilot(inputs, DIR));

  it("are reproduced byte for byte by the generator", () => {
    expect(Object.keys(files).sort()).toEqual(COMMITTED);
    expect(differing(DIR, files)).toEqual([]);
  });

  it("do not include the training split", () => {
    expect(Object.keys(files).some((name) => name.includes("training"))).toBe(false);
  });

  it("round-trip through writeFixtures", () => {
    const out = join(scratch, "out");
    expect(writeFixtures(out, files)).toEqual(COMMITTED);
    expect(differing(out, files)).toEqual([]);
    writeFileSync(join(out, MANIFEST_FILE), "{}");
    expect(differing(out, files)).toEqual([MANIFEST_FILE]);
    rmSync(join(out, HASHES_FILE));
    expect(differing(out, files).sort()).toEqual([HASHES_FILE, MANIFEST_FILE]);
  });
});

describe("committedDatasetHashes", () => {
  it("holds every committed team once and nothing else", () => {
    const known = committedDatasetHashes(inputs, FORMAT);
    const hashes = [teamsFile("dev"), teamsFile("validation")].flatMap((f) => readFileSync(join(DIR, f), "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l).team_hash as string));
    expect(known.size).toBe(hashes.length);
    expect(hashes.every((h) => known.has(h))).toBe(true);
  });
});

describe("the run record", () => {
  it("holds public keys only and names the generator commit", () => {
    const run = readFixtureRun(DIR);
    expect(run.generator_commit).toMatch(/^[0-9a-f]{40}$/);
    expect(JSON.parse(readFileSync(join(DIR, MANIFEST_FILE), "utf8")).key).toEqual({ kind: "public", generation_key: run.generation_key, assignment_key: run.assignment_key });
  });
  it("is refused when malformed", () => {
    const dir = join(scratch, "bad-run");
    cpSync(DIR, dir, { recursive: true });
    const run = JSON.parse(readFileSync(join(DIR, RUN_FILE), "utf8"));
    for (const broken of [{ ...run, schema_version: 2 }, { ...run, dataset_id: 5 }, { ...run, generation_key: "ab" }, { ...run, assignment_key: "AB".repeat(32) }]) {
      writeFileSync(join(dir, RUN_FILE), JSON.stringify(broken));
      expect(() => readFixtureRun(dir)).toThrow(FixtureError);
    }
  });
  it("is refused when its pilot was recorded for another config", () => {
    const dir = join(scratch, "bad-pilot");
    cpSync(DIR, dir, { recursive: true });
    const pilot = JSON.parse(readFileSync(join(DIR, PILOT_FILE), "utf8"));
    writeFileSync(join(dir, PILOT_FILE), JSON.stringify({ ...pilot, config_hash: "0".repeat(64) }));
    expect(() => recordedPilot(inputs, dir)).toThrow();
  });
});
