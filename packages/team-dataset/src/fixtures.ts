// The committed dev and validation fixtures (ADR-013 slice (c)): how they are
// made and how they are checked. `regenerate` runs the generator from the
// committed config, label set, vocabulary and the run record (`run.json`),
// and returns the bytes of every committed file; `writeFixtures` puts them
// on disk and a test compares them with what is committed, so the fixtures
// cannot be edited by hand and cannot drift from the generator.
//
// Only the public keys are ever in the run record. The training split is
// generated (it is part of the plan: the pool is assigned once) but is not
// returned, so it is never written.

import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type Engine,
  exportEngineData,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  readTeamVocabulary,
  resolveFormat,
  recordHash,
  type SyntheticTeamHashes,
  teamVocabularyHash,
  type TeamRecord,
} from "@areneon/showdown-bridge";
import { COMMITTED_SPLITS, type CommittedSplit, type DatasetContext, datasetSyntheticHashes, HASHES_FILE, hashIndex, MANIFEST_FILE, readDataset, teamsFile } from "./dataset-reader.ts";
import { pretty, recordsFile, writeDataset } from "./dataset-io.ts";
import { KnownHashes, referencePoolHash } from "./dedupe.ts";
import { generate } from "./generate.ts";
import { type Pilot, readPilot } from "./pilot.ts";
import type { SplitSizes } from "./plan.ts";

export const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));
export const DATASETS_DIR = (formatId: string): string => join(REPO_ROOT, "fixtures/teams", formatId, "datasets");
export const RUN_FILE = "run.json";
export const PILOT_FILE = "pilot.json";
export const COVERAGE_FILE = "coverage.json";

/** The inputs of a fixture run, all public. */
export interface FixtureRun {
  schema_version: 1;
  dataset_id: string;
  /** The sizes asked for (the quotas, with their floors, may deliver more). */
  sizes: Required<SplitSizes>;
  /** The commit whose generator the fixtures are made with. */
  generator_commit: string;
  generation_key: string;
  assignment_key: string;
}

export class FixtureError extends Error {
  override name = "FixtureError";
}

const json = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));

export function readFixtureRun(dir: string): FixtureRun {
  const run = json(join(dir, RUN_FILE)) as FixtureRun;
  const keys = [run.generation_key, run.assignment_key];
  if (run.schema_version !== 1 || typeof run.dataset_id !== "string" || !keys.every((k) => /^[0-9a-f]{64}$/.test(k))) throw new FixtureError(`${RUN_FILE} is not a fixture run record`);
  return run;
}

/**
 * The committed generator config a dataset directory was generated with: the `generator*.json` of the format whose
 * hash is the manifest's `config_hash` (a mixture or role-set dataset names its own file), else `generator.json`.
 */
function configFileOf(formatId: string, datasetDir: string | undefined): string {
  const dir = join(REPO_ROOT, "config/teams", formatId);
  const default_ = join(dir, "generator.json");
  if (datasetDir === undefined || !existsSync(join(datasetDir, MANIFEST_FILE))) return default_;
  const named = (json(join(datasetDir, MANIFEST_FILE)) as { generator?: { config_hash?: unknown } }).generator?.config_hash;
  for (const name of readdirSync(dir).filter((f) => /^generator.*\.json$/.test(f)).sort()) {
    if (recordHash(json(join(dir, name)) as never) === named) return join(dir, name);
  }
  return default_;
}

/** The committed inputs of a format, read against the pinned engine. With a dataset directory, the config it names is used. */
export function loadFixtureInputs(engine: Engine, formatId: string, datasetDir?: string) {
  const table = exportEngineData(engine, formatId);
  const config = readTeamGeneratorConfig(json(configFileOf(formatId, datasetDir)), table);
  const labelSet = readTeamLabelSet(json(join(REPO_ROOT, "config/teams", formatId, "labels.json")), table);
  const vocabulary = readTeamVocabulary(json(join(REPO_ROOT, "fixtures/showdown", formatId, "team-vocabulary.json")), table);
  const format = resolveFormat(engine, formatId);
  const context: DatasetContext = { engine, format, table, config, labelSet, vocabulary };
  return context;
}

export type FixtureInputs = ReturnType<typeof loadFixtureInputs>;

/**
 * Run the generator and return the committed files by name. With `pilot`, the
 * recorded pilot is used (cheap); without, the config's pilot is run first and
 * returned as `pilot.json` (about 40 s).
 */
export function regenerate(inputs: FixtureInputs, run: FixtureRun, pilot?: Pilot): Record<string, string> {
  const scratch = mkdtempSync(join(tmpdir(), "team-fixtures-"));
  try {
    const result = generate({
      engine: inputs.engine,
      format: inputs.format,
      table: inputs.table,
      vocabulary: inputs.vocabulary,
      labelSet: inputs.labelSet,
      config: inputs.config,
      generationKey: run.generation_key,
      assignmentKey: run.assignment_key,
      datasetId: run.dataset_id,
      planId: run.dataset_id,
      sizes: run.sizes,
      generatorCommit: run.generator_commit,
      reference: { hash: referencePoolHash([]), known: new KnownHashes([]) },
      pilot,
      logDir: scratch,
    });
    writeDataset(scratch, result);
    const out: Record<string, string> = {};
    for (const split of COMMITTED_SPLITS) out[teamsFile(split)] = readFileSync(join(scratch, recordsFile(split)), "utf8");
    for (const name of [MANIFEST_FILE, COVERAGE_FILE, PILOT_FILE]) {
      out[name] = readFileSync(join(scratch, name), "utf8");
    }
    const { dev, validation } = result.records as Record<CommittedSplit, TeamRecord[]>;
    out[HASHES_FILE] = pretty(hashIndex(result.manifest, { dev, validation }));
    return out;
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** The recorded pilot of a fixture directory, checked against the inputs. */
export function recordedPilot(inputs: FixtureInputs, dir: string): Pilot {
  const identity = { config_hash: recordHash(inputs.config), label_set_hash: recordHash(inputs.labelSet), vocabulary_hash: teamVocabularyHash(inputs.vocabulary) };
  return readPilot(json(join(dir, PILOT_FILE)), identity, inputs.labelSet);
}

/** Write the committed files into `dir` (created if needed). */
export function writeFixtures(dir: string, files: Record<string, string>): string[] {
  mkdirSync(dir, { recursive: true });
  for (const [name, text] of Object.entries(files)) writeFileSync(join(dir, name), text);
  return Object.keys(files).sort();
}

/** Names of committed files whose bytes differ from `files` (or are missing). */
export function differing(dir: string, files: Record<string, string>): string[] {
  return Object.entries(files)
    .filter(([name, text]) => {
      try {
        return readFileSync(join(dir, name), "utf8") !== text;
      } catch {
        return true;
      }
    })
    .map(([name]) => name);
}


/**
 * The teams of every committed dataset under `fixtures/teams/<format>/datasets/`,
 * each dataset read and verified first, as a `SyntheticTeamHashes`. A team is
 * synthetic only if its hash is in the set a caller passes as `known_synthetic`
 * (`readEpisode`, `stampTeam`, the training loader); a caller adds this set to
 * the sets it already trusts and nothing else.
 */
export function committedDatasetHashes(inputs: FixtureInputs, formatId: string): SyntheticTeamHashes {
  const root = DATASETS_DIR(formatId);
  const dirs = readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => join(root, e.name)).sort();
  const datasets = dirs.map((dir) => readDataset(dir, inputs));
  return datasetSyntheticHashes(datasets);
}
