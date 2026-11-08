// The run inputs of a rented machine (ADR-024 section 3.5, slice S-b): the owner
// puts a run's public inputs (the run config and the synthetic team dataset
// directory) in the private inputs bucket under `runs/<run_key>/inputs/`, with
// a `RunInputManifest` that lists every file's path, size and SHA-256 and names
// the frozen commit, the run config hash and the dataset hash. The machine
// pulls first, then runs: `fetchRunInputs` reads the manifest, downloads each
// object, checks size and SHA-256, writes the files under a fresh local
// directory (a temporary sibling renamed when everything verified), and runs the
// dataset reader's hash check against the config's `dataset_hash`. Any mismatch
// stops before the loop and leaves no directory at the output path. The loop
// then runs on local, hashed files with no network I/O.
//
// Everything in git (engine data, the Showdown pin, lockfiles, code) is not
// copied: it comes from the repository at the manifest's commit, which the
// fetch checks against the checked-out repository's HEAD.
//
// Keys: `runs/<run_key>/inputs/manifest.json` and
// `runs/<run_key>/inputs/files/<relative path>`. The layout handed to the
// shared client admits those and nothing else, and no overwrite.
//
// Every refusal is a `RunInputError` with a code and a fixed message: no path,
// hash, key or credential is ever in one. A store failure keeps its own fixed
// `ObjectStoreError`.

import { createHash } from "node:crypto";
import { closeSync, fsyncSync, lstatSync, mkdirSync, mkdtempSync, openSync, readdirSync, readFileSync, renameSync, rmSync, writeSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { MAX_OBJECT_BYTES, type ObjectLayout, type ObjectStore } from "@areneon/object-store";
import {
  type Engine,
  isSafeRelativePath,
  readRunInputManifest,
  readTrainingRunConfig,
  type RunInputManifest,
  runConfigHash,
  runInputManifestBytes,
  runInputManifestHash,
} from "@areneon/showdown-bridge";
import { verifiedDatasetHash } from "./dataset-hash.ts";

export const RUN_INPUT_REFUSALS = [
  "run_key_invalid",
  "output_exists",
  "manifest_missing",
  "manifest_not_synthetic",
  "manifest_invalid",
  "manifest_not_canonical",
  "manifest_other_run",
  "manifest_hash_mismatch",
  "commit_mismatch",
  "object_missing",
  "object_extra",
  "size_mismatch",
  "hash_mismatch",
  "config_invalid",
  "config_hash_mismatch",
  "config_other_run",
  "dataset_hash_mismatch",
  "dataset_invalid",
  "input_unsupported",
  "input_too_large",
  "write_failed",
] as const;
export type RunInputRefusal = (typeof RUN_INPUT_REFUSALS)[number];

const MESSAGES: Readonly<Record<RunInputRefusal, string>> = {
  run_key_invalid: "the run key is not a valid run key",
  output_exists: "the output path already exists; the inputs are written to a fresh directory only",
  manifest_missing: "the inputs bucket holds no run-input manifest for this run",
  manifest_not_synthetic: "the run-input manifest is not labelled synthetic",
  manifest_invalid: "the run-input manifest is not a valid RunInputManifest",
  manifest_not_canonical: "the run-input manifest is not stored as its canonical (JCS) bytes",
  manifest_other_run: "the run-input manifest names another run",
  manifest_hash_mismatch: "the run-input manifest is not the one whose hash was given",
  commit_mismatch: "the manifest's commit is not the checked-out repository's commit",
  object_missing: "an object the manifest lists is not in the bucket",
  object_extra: "the bucket holds an object the manifest does not list",
  size_mismatch: "an object's size is not the size the manifest lists",
  hash_mismatch: "an object's SHA-256 is not the hash the manifest lists",
  config_invalid: "the run config is not a valid TrainingRunConfig with a team dataset",
  config_hash_mismatch: "the run config does not hash to the manifest's run_config_hash",
  config_other_run: "the run config names another run than the manifest",
  dataset_hash_mismatch: "the dataset hash of the config, the manifest and the dataset directory are not all equal",
  dataset_invalid: "the dataset directory does not pass the dataset reader",
  input_unsupported: "an input is not a plain file or a plain directory",
  input_too_large: "an input is larger than one object may be",
  write_failed: "an input could not be written to the local disk",
};

export class RunInputError extends Error {
  override name = "RunInputError";
  readonly code: RunInputRefusal;

  constructor(code: RunInputRefusal) {
    super(`run inputs refused: ${MESSAGES[code]}`);
    this.code = code;
  }
}

const RUN_KEY = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const HASH = /^[0-9a-f]{64}$/;

/** The default names of the two things the manifest points at. */
export const RUN_CONFIG_FILE = "run-config.json";
export const DATASET_DIR = "dataset";

const checkRunKey = (runKey: string): string => {
  if (!RUN_KEY.test(runKey)) throw new RunInputError("run_key_invalid");
  return runKey;
};

export const inputsPrefix = (runKey: string): string => `runs/${checkRunKey(runKey)}/inputs/`;
export const manifestKey = (runKey: string): string => `${inputsPrefix(runKey)}manifest.json`;
export const filesPrefix = (runKey: string): string => `${inputsPrefix(runKey)}files/`;
export const fileKey = (runKey: string, path: string): string => `${filesPrefix(runKey)}${path}`;

/** What this run's inputs admit through the shared client: the manifest, the files under `files/`, one listing prefix, no overwrite. */
export function runInputLayout(runKey: string): ObjectLayout {
  const files = filesPrefix(runKey);
  const manifest = manifestKey(runKey);
  return {
    allowsKey: (key) => key === manifest || (key.startsWith(files) && isSafeRelativePath(key.slice(files.length))),
    allowsListPrefix: (prefix) => prefix === files,
    overwriteKey: null,
  };
}

const sha256Of = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

// ---------------------------------------------------------------------------
// The owner's side: build the manifest from a verified dataset and a run config.

/** The inputs of a run as the owner holds them: the manifest and where each listed file is read from. */
export interface RunInputs {
  readonly manifest: RunInputManifest;
  /** Listed path to the absolute local file it comes from. */
  readonly sources: ReadonlyMap<string, string>;
}

export interface BuildRunInputsInput {
  readonly engine: Engine;
  /** The run config file (JSON); its bytes travel as `run-config.json`. */
  readonly configFile: string;
  /** The dataset directory the config's `team_dataset` names (verified by the dataset reader). */
  readonly datasetDir: string;
  /** The frozen commit (40 lower-case hex) the run is started from. */
  readonly commit: string;
}

/** Every plain file under `dir`, as `/`-separated relative paths; a symbolic link or special file is refused. */
function walk(dir: string, prefix = ""): string[] {
  const found: string[] = [];
  for (const name of readdirSync(dir).sort()) {
    const path = join(dir, name);
    const stat = lstatSync(path);
    if (stat.isDirectory()) found.push(...walk(path, `${prefix}${name}/`));
    else if (stat.isFile()) found.push(`${prefix}${name}`);
    else throw new RunInputError("input_unsupported");
  }
  return found;
}

function fileEntry(path: string, source: string): { path: string; size: number; sha256: string } {
  const stat = lstatSync(source);
  if (!stat.isFile()) throw new RunInputError("input_unsupported");
  if (stat.size > MAX_OBJECT_BYTES) throw new RunInputError("input_too_large");
  const bytes = readFileSync(source);
  return { path, size: bytes.byteLength, sha256: sha256Of(bytes) };
}

/**
 * Build the manifest of a run's inputs. The dataset directory goes through the
 * dataset reader and must hash to the config's `dataset_hash`; the config
 * through `readTrainingRunConfig`. Nothing is written or sent.
 */
export async function buildRunInputs(input: BuildRunInputsInput): Promise<RunInputs> {
  let configBytes: Buffer;
  let config;
  try {
    configBytes = readFileSync(input.configFile);
    config = readTrainingRunConfig(JSON.parse(configBytes.toString("utf8")));
  } catch {
    throw new RunInputError("config_invalid");
  }
  if (config.team_dataset === undefined) throw new RunInputError("config_invalid");
  let datasetHash: string;
  try {
    datasetHash = verifiedDatasetHash(input.engine, input.datasetDir, config.format_id);
  } catch {
    throw new RunInputError("dataset_invalid");
  }
  if (datasetHash !== config.team_dataset.dataset_hash) throw new RunInputError("dataset_hash_mismatch");
  const sources = new Map<string, string>([[RUN_CONFIG_FILE, resolve(input.configFile)]]);
  for (const path of walk(input.datasetDir)) sources.set(`${DATASET_DIR}/${path}`, resolve(input.datasetDir, path));
  const files = [...sources.keys()].sort().map((path) => fileEntry(path, sources.get(path) as string));
  let manifest: RunInputManifest;
  try {
    manifest = readRunInputManifest({
      schema_version: 1,
      data_class: "synthetic",
      run_key: config.seeds.run_key,
      commit: input.commit,
      run_config_hash: await runConfigHash(config),
      dataset_hash: datasetHash,
      config_file: RUN_CONFIG_FILE,
      dataset_dir: DATASET_DIR,
      files,
    });
  } catch {
    throw new RunInputError("manifest_invalid");
  }
  return { manifest, sources };
}

/** Put the inputs in the bucket: every file, then the manifest last (an interrupted upload leaves no manifest). Nothing is overwritten. Returns the manifest's hash. */
export async function uploadRunInputs(store: ObjectStore, inputs: RunInputs): Promise<string> {
  const { manifest } = inputs;
  for (const entry of manifest.files) {
    await store.putIfAbsent(fileKey(manifest.run_key, entry.path), readFileSync(inputs.sources.get(entry.path) as string));
  }
  await store.putIfAbsent(manifestKey(manifest.run_key), runInputManifestBytes(manifest));
  return runInputManifestHash(manifest);
}

// ---------------------------------------------------------------------------
// The machine's side: pull, check every hash, then (and only then) the loop.

export interface FetchRunInputsInput {
  readonly store: ObjectStore;
  readonly engine: Engine;
  readonly runKey: string;
  /** A fresh path: it must not exist. The inputs appear there complete or not at all. */
  readonly outDir: string;
  /** The checked-out repository's HEAD commit (40 hex). */
  readonly headCommit: () => string;
  /** Optional pin of the manifest's hash (SHA-256 of its JCS bytes), given by the owner out of band. */
  readonly expectedManifestHash?: string;
}

export interface FetchedRunInputs {
  /** The directory the inputs were written to. */
  readonly dir: string;
  /** The verified dataset directory inside it (the run config's `team_dataset.path` must name it). */
  readonly datasetDir: string;
  /** The run config file inside it. */
  readonly configFile: string;
  readonly manifestHash: string;
  readonly runConfigHash: string;
  readonly datasetHash: string;
  readonly files: number;
}

function parseManifest(bytes: Uint8Array): RunInputManifest {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new RunInputError("manifest_invalid");
  }
  if (typeof value === "object" && value !== null && "data_class" in value && (value as { data_class: unknown }).data_class !== "synthetic") {
    throw new RunInputError("manifest_not_synthetic");
  }
  let manifest: RunInputManifest;
  try {
    manifest = readRunInputManifest(value);
  } catch {
    throw new RunInputError("manifest_invalid");
  }
  if (!sameBytes(bytes, runInputManifestBytes(manifest))) throw new RunInputError("manifest_not_canonical");
  return manifest;
}

const sameBytes = (a: Uint8Array, b: Uint8Array): boolean => Buffer.from(a).equals(Buffer.from(b));

/** Write one file durably; whatever the file system says is dropped. */
function writeFile(path: string, bytes: Uint8Array): void {
  try {
    mkdirSync(dirname(path), { recursive: true });
    const fd = openSync(path, "wx", 0o644);
    try {
      writeSync(fd, bytes);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  } catch {
    throw new RunInputError("write_failed");
  }
}

function occupied(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Pull a run's inputs and verify them. Order: the manifest (its canonical
 * bytes, optional hash pin, run key, commit against HEAD); the bucket listing
 * against the manifest (no missing or extra object); every object's size and
 * SHA-256 as it arrives; the run config against the manifest's run key and
 * config hash and its `dataset_hash`; the dataset reader's hash of the written
 * directory. The files are written to a temporary sibling of `outDir` and
 * renamed to it only after every check passed; on any failure the sibling is
 * removed and `outDir` does not exist.
 */
export async function fetchRunInputs(input: FetchRunInputsInput): Promise<FetchedRunInputs> {
  const { store, runKey } = input;
  checkRunKey(runKey);
  if (occupied(input.outDir)) throw new RunInputError("output_exists");
  const manifestBytes = await store.get(manifestKey(runKey));
  if (manifestBytes === null) throw new RunInputError("manifest_missing");
  const manifest = parseManifest(manifestBytes);
  const manifestHash = sha256Of(manifestBytes);
  if (input.expectedManifestHash !== undefined && (!HASH.test(input.expectedManifestHash) || input.expectedManifestHash !== manifestHash)) throw new RunInputError("manifest_hash_mismatch");
  if (manifest.run_key !== runKey) throw new RunInputError("manifest_other_run");
  if (input.headCommit() !== manifest.commit) throw new RunInputError("commit_mismatch");

  const wanted = new Set(manifest.files.map((entry) => fileKey(runKey, entry.path)));
  const listed = new Set(await store.list(filesPrefix(runKey)));
  for (const key of listed) if (!wanted.has(key)) throw new RunInputError("object_extra");
  for (const key of wanted) if (!listed.has(key)) throw new RunInputError("object_missing");

  const out = resolve(input.outDir);
  try {
    mkdirSync(dirname(out), { recursive: true });
  } catch {
    throw new RunInputError("write_failed");
  }
  let partial: string;
  try {
    partial = mkdtempSync(join(dirname(out), `.${basename(out)}.partial-`));
  } catch {
    throw new RunInputError("write_failed");
  }
  let done = false;
  try {
    for (const entry of manifest.files) {
      const bytes = await store.get(fileKey(runKey, entry.path));
      if (bytes === null) throw new RunInputError("object_missing");
      if (bytes.byteLength !== entry.size) throw new RunInputError("size_mismatch");
      if (sha256Of(bytes) !== entry.sha256) throw new RunInputError("hash_mismatch");
      writeFile(join(partial, entry.path), bytes);
    }
    const config = readConfig(join(partial, manifest.config_file));
    if ((await runConfigHash(config)) !== manifest.run_config_hash) throw new RunInputError("config_hash_mismatch");
    if (config.seeds.run_key !== manifest.run_key) throw new RunInputError("config_other_run");
    if (config.team_dataset?.dataset_hash !== manifest.dataset_hash) throw new RunInputError("dataset_hash_mismatch");
    let found: string;
    try {
      found = verifiedDatasetHash(input.engine, join(partial, manifest.dataset_dir), config.format_id);
    } catch {
      throw new RunInputError("dataset_invalid");
    }
    if (found !== manifest.dataset_hash) throw new RunInputError("dataset_hash_mismatch");
    if (occupied(out)) throw new RunInputError("output_exists");
    try {
      renameSync(partial, out);
    } catch {
      throw new RunInputError("write_failed");
    }
    done = true;
    return {
      dir: out,
      datasetDir: join(out, manifest.dataset_dir),
      configFile: join(out, manifest.config_file),
      manifestHash,
      runConfigHash: manifest.run_config_hash,
      datasetHash: manifest.dataset_hash,
      files: manifest.files.length,
    };
  } finally {
    if (!done) rmSync(partial, { recursive: true, force: true });
  }
}

function readConfig(path: string): ReturnType<typeof readTrainingRunConfig> {
  try {
    return readTrainingRunConfig(JSON.parse(readFileSync(path, "utf8")));
  } catch {
    throw new RunInputError("config_invalid");
  }
}

