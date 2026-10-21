// Reading a committed dataset back (ADR-013 section 4, reader rules) and the
// training draw that must never see validation (ADR-010 E2).
//
// A committed dataset directory holds
//   teams-dev.jsonl, teams-validation.jsonl   one TeamRecord per line, in JCS form
//   manifest.json                             the TeamDatasetManifest
//   hashes.json                               the sorted record_hash and canonical_team_hash
//                                             lists of each committed split, and the manifest hash
// (the training split is generated with the others but never committed).
// A dataset directory as `generate` wrote it (`records.<split>.jsonl` for dev,
// validation and training, `manifest.json`; no `hashes.json`, which belongs to
// the committed fixtures) is read the same way and also yields its training
// split. The layout is told by the presence of `records.dev.jsonl`.
// `readDataset` trusts none of it: the manifest is read against the config and
// label set, every record is read through the bridge (the engine's paste round
// trip, both hashes), judged by the pinned validator, and its labels must be
// the labeller's output; a split holds exactly its quota in every stratum; no
// team is in two splits (L1); the hash lists are recomputed and must match
// (the same lists a Parquet shard of the split must reproduce, so the
// fixtures and the shards cannot drift apart). Whether a team is legal is the
// engine's call (`validateTeam`), never this file's.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type Engine,
  type EngineData,
  importPaste,
  jcs,
  readTeamDatasetManifest,
  recordHash,
  type ResolvedFormat,
  type SyntheticTeamHashes,
  syntheticTeamHashes,
  type TeamDatasetManifest,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamRecord,
  type TeamVocabulary,
  teamVocabularyHash,
  canonicalTeamHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { referencePoolHash, sortedUnique } from "./dedupe.ts";
import { recordsFile } from "./dataset-io.ts";
import { checkRecord } from "./generate.ts";
import { cellOf } from "./labeller.ts";

export class DatasetReadError extends Error {
  override name = "DatasetReadError";
}

const bad = (message: string): never => {
  throw new DatasetReadError(message);
};

/** The splits a dataset directory commits. Training is never one of them. */
export const COMMITTED_SPLITS = ["dev", "validation"] as const;
export type CommittedSplit = (typeof COMMITTED_SPLITS)[number];
type ReadSplit = CommittedSplit | "training";

export const teamsFile = (split: string): string => `teams-${split}.jsonl`;
export const MANIFEST_FILE = "manifest.json";
export const HASHES_FILE = "hashes.json";

/** The hash lists of one committed split: both sorted and unique. */
export interface SplitHashes {
  record_hashes: string[];
  canonical_team_hashes: string[];
}

export interface HashIndex {
  schema_version: 1;
  dataset_id: string;
  /** SHA-256 of the manifest's JCS bytes: the dataset hash a snapshot lists. */
  manifest_hash: string;
  splits: Record<CommittedSplit, SplitHashes>;
}

/** What a dataset is read against. */
export interface DatasetContext {
  engine: Engine;
  format: ResolvedFormat;
  table: EngineData;
  config: TeamGeneratorConfig;
  labelSet: TeamLabelSet;
  /** The vocabulary the dataset was drawn from: its hash must be the manifest's `vocabulary_hash`. */
  vocabulary: TeamVocabulary;
  /** The canonical hashes of the earlier datasets it deduplicated against (none for the first dataset): `reference_pool_hash` must be their hash. */
  referenceHashes?: readonly string[];
}

/** A dataset that `readDataset` returned. Only such an object is accepted by `trainingPool`. */
export type VerifiedDataset = VerifiedShape & { readonly [VERIFIED]: true };
declare const VERIFIED: unique symbol;
const verifiedDatasets = new WeakSet<object>();

interface VerifiedShape {
  manifest: TeamDatasetManifest;
  manifest_hash: string;
  splits: Record<CommittedSplit, TeamRecord[]>;
  /** The training split of a dataset directory as `generate` wrote it; null for the committed layout. */
  training: TeamRecord[] | null;
  hashes: HashIndex;
}

const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (values: readonly string[]): string[] => [...values].sort(codeUnits);

/** Sorted, unique hex lists. */
function checkList(list: unknown, what: string): string[] {
  if (!Array.isArray(list) || !list.every((h) => typeof h === "string" && /^[0-9a-f]{64}$/.test(h))) return bad(`${what} is not a list of SHA-256 hex digests`);
  const hashes = list as string[];
  hashes.forEach((h, k) => {
    if (k > 0 && codeUnits(hashes[k - 1] as string, h) >= 0) bad(`${what} is not sorted and unique`);
  });
  return hashes;
}

/** The hash lists of a split's records. */
export function splitHashes(records: readonly TeamRecord[]): SplitHashes {
  return { record_hashes: sorted(records.map(recordHash)), canonical_team_hashes: sorted(records.map((r) => r.canonical_team_hash)) };
}

/** The hash index of the committed splits, as `hashes.json` holds it. */
export function hashIndex(manifest: TeamDatasetManifest, splits: Record<CommittedSplit, readonly TeamRecord[]>): HashIndex {
  return {
    schema_version: 1,
    dataset_id: manifest.dataset_id,
    manifest_hash: recordHash(manifest),
    splits: { dev: splitHashes(splits.dev), validation: splitHashes(splits.validation) },
  };
}

function readJson(dir: string, file: string): unknown {
  try {
    return JSON.parse(readFileSync(join(dir, file), "utf8"));
  } catch (error) {
    return bad(`${file}: ${(error as Error).message}`);
  }
}

/** The stratum of every base cell the manifest names: a merged stratum for its members, else the cell. */
function stratumOf(manifest: TeamDatasetManifest, cell: number): number {
  const group = (manifest.merged_cells ?? []).find((g) => g.cells.includes(cell));
  return group === undefined ? cell : group.stratum;
}

function readSplit(dir: string, split: ReadSplit, file: string, manifest: TeamDatasetManifest, context: DatasetContext): TeamRecord[] {
  const { engine, format, labelSet } = context;
  let text: string;
  try {
    text = readFileSync(join(dir, file), "utf8");
  } catch (error) {
    return bad(`${file}: ${(error as Error).message}`);
  }
  if (text !== "" && !text.endsWith("\n")) bad(`${file} does not end with a newline`);
  const plan = manifest.splits.find((s) => s.split === split) ?? bad(`the manifest has no ${split} split`);
  const lines = text === "" ? [] : text.slice(0, -1).split("\n");
  const records = lines.map((line, n) => {
    const where = `${file} line ${n + 1}`;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      return bad(`${where}: ${(error as Error).message}`);
    }
    if (jcs(value as never) !== line) bad(`${where} is not in canonical (JCS) form`);
    try {
      checkRecord(engine, value as TeamRecord, labelSet);
    } catch (error) {
      return bad(`${where}: ${(error as Error).message}`);
    }
    const record = value as TeamRecord;
    if (record.origin.split !== split || record.origin.dataset_id !== manifest.dataset_id) bad(`${where}: its origin is ${record.origin.dataset_id}/${record.origin.split}`);
    if (record.origin.index !== n) bad(`${where}: its origin index is ${record.origin.index}`);
    const verdict = validateTeam(engine, format, importPaste(engine, record.paste));
    if (!verdict.valid) bad(`${where}: the pinned validator refuses the team: ${verdict.problems[0]}`);
    return record;
  });
  if (records.length !== plan.teams) bad(`${file} holds ${records.length} teams, the manifest says ${plan.teams}`);
  // Each stratum holds exactly its quota.
  const held = new Map<number, number>();
  for (const record of records) {
    const stratum = stratumOf(manifest, cellOf(labelSet, record.labels));
    held.set(stratum, (held.get(stratum) ?? 0) + 1);
  }
  for (const cell of plan.cells) {
    if ((held.get(cell.stratum) ?? 0) !== cell.quota) bad(`${file}: stratum ${cell.stratum} holds ${held.get(cell.stratum) ?? 0} teams, its quota is ${cell.quota}`);
  }
  return records;
}

/**
 * Read and verify the committed dataset in `dir`. Every refusal is a
 * `DatasetReadError`. The manifest's config, label set, vocabulary and
 * reference pool hashes and its version manifest are recomputed from the
 * context. `generator_commit` cannot be verified here (it names code, not
 * data); the byte-for-byte regeneration test is what ties it to the files.
 */
export function readDataset(dir: string, context: DatasetContext): VerifiedDataset {
  let manifest: TeamDatasetManifest;
  try {
    manifest = readTeamDatasetManifest(readJson(dir, MANIFEST_FILE), { config: context.config, labelSet: context.labelSet });
  } catch (error) {
    if (error instanceof DatasetReadError) throw error;
    return bad(`${MANIFEST_FILE}: ${(error as Error).message}`);
  }
  if (jcs(manifest.generator.manifest as never) !== jcs(context.format.manifest as never)) bad("the dataset was generated under another version manifest than the pinned engine's");
  if (manifest.generator.vocabulary_hash !== teamVocabularyHash(context.vocabulary)) bad("vocabulary_hash is not the hash of the given vocabulary");
  if (manifest.reference_pool_hash !== referencePoolHash(sortedUnique([...(context.referenceHashes ?? [])]))) bad("reference_pool_hash is not the hash of the given reference hashes");
  const generated = existsSync(join(dir, recordsFile("dev")));
  const fileOf = (split: ReadSplit): string => (generated ? recordsFile(split) : teamsFile(split));
  const splits = { dev: readSplit(dir, "dev", fileOf("dev"), manifest, context), validation: readSplit(dir, "validation", fileOf("validation"), manifest, context) };
  const training = generated ? readSplit(dir, "training", fileOf("training"), manifest, context) : null;
  // L1: no team in two splits, and none twice.
  const seen = new Map<string, string>();
  for (const [split, records] of [["dev", splits.dev], ["validation", splits.validation], ["training", training ?? []]] as const) {
    for (const record of records) {
      const earlier = seen.get(record.canonical_team_hash);
      if (earlier !== undefined) bad(`team ${record.canonical_team_hash} is in ${earlier} and in ${split}`);
      seen.set(record.canonical_team_hash, split);
    }
  }
  const recomputed = hashIndex(manifest, splits);
  const hashes = generated ? recomputed : readHashIndex(readJson(dir, HASHES_FILE), recomputed);
  const verified = { manifest, manifest_hash: hashes.manifest_hash, splits, training, hashes } as VerifiedDataset;
  verifiedDatasets.add(verified);
  return verified;
}

/** The index in `hashes.json`, which must equal the one recomputed from the records. */
function readHashIndex(value: unknown, expected: HashIndex): HashIndex {
  const given = value as Partial<HashIndex> | null;
  if (typeof given !== "object" || given === null) return bad(`${HASHES_FILE} is not an object`);
  for (const split of COMMITTED_SPLITS) {
    const lists = given.splits?.[split];
    checkList(lists?.record_hashes, `${HASHES_FILE}: ${split} record_hashes`);
    checkList(lists?.canonical_team_hashes, `${HASHES_FILE}: ${split} canonical_team_hashes`);
  }
  if (jcs(given as never) !== jcs(expected as never)) bad(`${HASHES_FILE} does not list the hashes of the committed records and manifest`);
  return expected;
}

/**
 * The dataset hash a run config pins: SHA-256 of the JCS of the manifest's own
 * hash and, per split, the sorted `record_hash` list of the records the reader
 * verified (computed here from the records, never read from a file; the
 * committed layout has no training split, an empty list). A swapped, added or
 * removed record changes it.
 */
export function datasetHash(dataset: VerifiedDataset): string {
  const list = (records: readonly TeamRecord[]): string[] => splitHashes(records).record_hashes;
  return recordHash({
    manifest_hash: recordHash(dataset.manifest),
    splits: { dev: list(dataset.splits.dev), validation: list(dataset.splits.validation), training: list(dataset.training ?? []) },
  } as never);
}

/** The positional `team_hash` of every team a verified dataset holds: the teams that are synthetic by source. */
export function datasetSyntheticHashes(datasets: readonly VerifiedDataset[]): SyntheticTeamHashes {
  return syntheticTeamHashes(datasets.flatMap((d) => [...COMMITTED_SPLITS.flatMap((split) => d.splits[split]), ...(d.training ?? [])].map((r) => r.team_hash)));
}

export class TrainingDrawError extends Error {
  override name = "TrainingDrawError";
}

const refuse = (message: string): never => {
  throw new TrainingDrawError(message);
};

const pools = new WeakSet<object>();

/** Teams a training run may play: only `trainingPool` makes one. */
export interface TrainingPool {
  readonly records: readonly TeamRecord[];
  readonly [POOL]: true;
}
declare const POOL: unique symbol;

/**
 * The training pool (ADR-010 E2) of verified datasets: their dev records and,
 * for a dataset directory `generate` wrote, their training records; never
 * validation. Nothing is taken on a caller's word: a dataset must be
 * the result of `readDataset`; every canonical hash is recomputed from the
 * paste through the engine (a record's own hash field is not trusted); and a
 * dev team whose recomputed hash equals a validation team's, in any of the
 * given datasets, is refused. The refusal is absolute.
 */
export function trainingPool(datasets: readonly VerifiedDataset[], engine: Engine): TrainingPool {
  if (datasets.length === 0) refuse("the training pool has no dataset");
  const canonical = (record: TeamRecord): string => canonicalTeamHash(importPaste(engine, record.paste));
  for (const dataset of datasets) if (!verifiedDatasets.has(dataset)) refuse("a dataset that readDataset did not return is not trusted");
  const blocked = new Set(datasets.flatMap((d) => d.splits.validation.map(canonical)));
  const records: TeamRecord[] = [];
  const seen = new Set<string>();
  for (const dataset of datasets) {
    const taken = [...dataset.splits.dev.map((record) => ["dev", record] as const), ...(dataset.training ?? []).map((record) => ["training", record] as const)];
    for (const [split, record] of taken) {
      if (record.origin.split !== split) refuse(`${record.canonical_team_hash} is not a ${split} record`);
      const hash = canonical(record);
      if (blocked.has(hash)) refuse(`${hash} duplicates a validation team`);
      if (seen.has(hash)) refuse(`${hash} is in the training pool twice`);
      seen.add(hash);
      records.push(record);
    }
  }
  if (records.length === 0) refuse("the training pool is empty");
  const pool = { records } as unknown as TrainingPool;
  pools.add(pool);
  return pool;
}

/** The team at `index` of a pool `trainingPool` made (the caller's seeded generator picks the index). */
export function drawTrainingTeam(pool: TrainingPool, index: number): TeamRecord {
  if (!pools.has(pool)) refuse("a pool that trainingPool did not make is not trusted");
  if (!Number.isSafeInteger(index) || index < 0 || index >= pool.records.length) return refuse(`draw ${index} is outside the pool of ${pool.records.length} teams`);
  return pool.records[index] as TeamRecord;
}
