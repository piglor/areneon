// The evaluation-only dataset (ADR-020 section 7): teams within d_team of a reference team are dropped and logged,
// the manifest marks the dataset, evaluation readers accept it and the training pool and the loop's team pool refuse it.

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { importPaste, readTrainingRunConfig, readTeamDatasetManifest, type PokemonSet } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { Binner } from "../src/binner.ts";
import { recordPastes, writeDataset } from "../src/dataset-io.ts";
import { readDataset, TrainingDrawError, trainingPool } from "../src/dataset-reader.ts";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR, loadFixtureInputs, readFixtureRun, recordedPilot } from "../src/fixtures.ts";
import { generate } from "../src/generate.ts";
import { NearIndex } from "../src/near-index.ts";
import { roleSetsSource } from "../src/role-set-proposal.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { runCli } from "../src/command.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const inputs = loadFixtureInputs(engine, FORMAT);
const scratch = mkdtempSync(join(tmpdir(), "evaluation-only-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;

describe("the Binner's near-reference check", () => {
  const draw = (hash: string, stratum = 0) => ({ status: "accepted", k: 0, canonical_team_hash: hash, team_hash: hash, paste: hash, labels: [], cell: stratum }) as never;
  it("drops exactly the candidates the predicate flags, after the full-bin and duplicate checks", () => {
    const near = new Set(["n1", "n2"]);
    const binner = new Binner(new Map([[0, 3]]), { has: (h: string) => h === "known" }, (d) => near.has(d.canonical_team_hash));
    const offers = ["a", "n1", "known", "b", "n2", "c", "d", "n1"].map((h) => [h, binner.offer(draw(h), 0)]);
    expect(offers).toEqual([
      ["a", "binned"],
      ["n1", "near_reference"],
      ["known", "duplicate"],
      ["b", "binned"],
      ["n2", "near_reference"],
      ["c", "binned"],
      ["d", "overflow"], // full bin first
      ["n1", "overflow"],
    ]);
    expect(binner.bin(0).map((d) => d.canonical_team_hash)).toEqual(["a", "b", "c"]);
  });
  it("is not asked without a predicate", () => {
    const binner = new Binner(new Map([[0, 1]]), { has: () => false });
    expect(binner.offer(draw("a"), 0)).toBe("binned");
  });
});

/** The first set of a team with one field changed: one differing field from the original. */
function nudged(sets: PokemonSet[]): PokemonSet[] {
  const [first, ...rest] = sets as [PokemonSet, ...PokemonSet[]];
  return [{ ...first, nature: first.nature === "Jolly" ? "Adamant" : "Jolly" }, ...rest];
}

describe("an evaluation-only generation from the committed config", () => {
  const run = readFixtureRun(SOURCE);
  const refPastes = [...recordPastes([join(SOURCE, "teams-dev.jsonl"), join(SOURCE, "teams-validation.jsonl")])];
  // The same stream as the fixtures run: its first teams are the committed ones. The references are those teams with one field
  // changed, so none is equal (L1 would not see them) and every one is within 1 of its twin.
  const index = new NearIndex(refPastes.map((paste) => nudged(importPaste(engine, paste))), 1);
  const dir = join(scratch, "eval");
  mkdirSync(dir);
  const result = generate({
    engine, format: inputs.format, table: inputs.table, vocabulary: inputs.vocabulary, labelSet: inputs.labelSet, config: inputs.config,
    generationKey: run.generation_key, assignmentKey: run.assignment_key, datasetId: "eval-only-1", planId: run.dataset_id, sizes: run.sizes,
    generatorCommit: run.generator_commit, reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, pilot: recordedPilot(inputs, SOURCE),
    logDir: dir, exclude: { d_team: 1, index },
  });
  writeDataset(dir, result);

  const log = readFileSync(join(dir, "rejections.jsonl"), "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l) as { reason: string; canonical_team_hash: string });
  const fixtureHashes = new Set(
    [...readFileSync(join(SOURCE, "teams-dev.jsonl"), "utf8").split("\n"), ...readFileSync(join(SOURCE, "teams-validation.jsonl"), "utf8").split("\n")].filter((l) => l !== "").map((l) => (JSON.parse(l) as { canonical_team_hash: string }).canonical_team_hash),
  );

  it("drops the near teams, logs each as near_reference and counts them in the manifest", () => {
    const dropped = log.filter((e) => e.reason === "near_reference");
    expect(dropped.length).toBeGreaterThan(0);
    expect(result.manifest.evaluation_only).toEqual({ d_team: 1, near_references: dropped.length });
    // Only the twins of the references are dropped: every dropped team is one of the committed teams the references were made from.
    for (const e of dropped) expect(fixtureHashes.has(e.canonical_team_hash)).toBe(true);
    expect(readTeamDatasetManifest(JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")), { config: inputs.config, labelSet: inputs.labelSet }).evaluation_only).toBeDefined();
  });

  it("keeps no team within d_team of a reference, and the stream's other teams stay", () => {
    for (const split of ["dev", "validation", "training"] as const) {
      for (const record of result.records[split] ?? []) {
        expect(index.near(importPaste(engine, record.paste))).toBe(false);
        expect(fixtureHashes.has(record.canonical_team_hash)).toBe(false);
      }
    }
    expect(result.records.validation).toHaveLength(result.coverage.splits.find((s) => s.split === "validation")?.teams as number);
    expect(result.coverage.population).toContain("evaluation-only: teams within 1 differing fields of a reference team were dropped");
  });

  it("is read by the evaluation readers and refused as training data", () => {
    const dataset = readDataset(dir, inputs);
    expect(dataset.manifest.evaluation_only).toEqual({ d_team: 1, near_references: result.manifest.evaluation_only?.near_references });
    expect(() => trainingPool([dataset], engine)).toThrow(TrainingDrawError);
    expect(() => trainingPool([dataset], engine)).toThrow(/evaluation-only/);
    const config = readTrainingRunConfig({ ...structuredClone(tiny), team_dataset: { path: dir, dataset_hash: verifiedDatasetHash(engine, dir, FORMAT) } });
    expect(() => datasetTeamPool(engine, config)).toThrow(/evaluation-only/);
    const validation = datasetValidationPool(engine, config);
    expect(validation.teams.length).toBeGreaterThan(0);
  });

  it("is a different dataset hash than the same run without the exclusion", () => {
    expect(verifiedDatasetHash(engine, dir, FORMAT)).not.toBe(verifiedDatasetHash(engine, SOURCE, FORMAT));
  });
}, 600_000); // a 300-team generation through the pinned validator, with an engine import per candidate

describe("the refusals of an evaluation-only run", () => {
  const index = new NearIndex([], 1);
  const base = {
    engine, format: inputs.format, table: inputs.table, vocabulary: inputs.vocabulary, labelSet: inputs.labelSet, config: inputs.config,
    generationKey: "11".repeat(32), assignmentKey: "22".repeat(32), datasetId: "x", planId: "x", sizes: { dev: 1 }, generatorCommit: "a".repeat(40),
    reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, pilot: recordedPilot(inputs, SOURCE), logDir: scratch,
  };
  it("needs a pure component config", () => {
    const roleSets = roleSetsSource(engine, inputs.vocabulary, inputs.table, JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json"), "utf8")));
    const mixed = { ...inputs.config, proposal: { kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: roleSets.hash, share_permille: 700 }, { kind: "uniform", share_permille: 300 }] } } as never;
    expect(() => generate({ ...base, config: mixed, roleSets, exclude: { d_team: 1, index } })).toThrow(/pure component/);
  });
  it("the manifest reader refuses a mixed config, and a command without --d-team or a --reference", () => {
    const manifest = JSON.parse(readFileSync(join(SOURCE, "manifest.json"), "utf8"));
    const mixed = { ...inputs.config, proposal: { kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: "b".repeat(64), share_permille: 700 }, { kind: "uniform", share_permille: 300 }] } } as never;
    expect(() => readTeamDatasetManifest({ ...manifest, evaluation_only: { d_team: 1, near_references: 0 } }, { config: mixed })).toThrow(/pure component|config_hash/);
    const err: string[] = [];
    const code = runCli(["generate", "--config", "c", "--labels", "l", "--format", FORMAT, "--out", scratch, "--dataset-id", "x", "--generation-key", "11".repeat(32), "--assignment-key", "22".repeat(32), "--evaluation-only"], { out: () => undefined, err: (l) => err.push(l), engine });
    expect(code).toBe(2);
  });
});
