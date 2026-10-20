// Converting a dataset directory (JSON Lines records and manifest.json, as the
// generator or the committed fixtures hold them) into shards and a snapshot
// manifest, and reading shards back to the original records.
//
// Memory is bounded by one shard, not by the split: pass one streams the
// records, checks their canonical JSON form, assigns each to its shard by the
// HMAC rule and appends it to that shard's scratch file; pass two sorts one
// shard at a time (stratum, canonical hash), writes its rows for the writer,
// has DuckDB write the Parquet file, names the file by its SHA-256, and reads
// it back through `readShard`. A shard that does not read back is an error;
// nothing the converter writes has not been read once.

import { createHash } from "node:crypto";
import { closeSync, createReadStream, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, renameSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readTeamDatasetManifest, readTeamSnapshotManifest, recordHash, shardLogicalHash, type TeamDatasetManifest, type TeamLabelSet, type TeamRecord, type TeamSnapshotManifest } from "@areneon/showdown-bridge";
import { type Column, columnsFor, recordJson, refuse, rowOf, ShardError } from "./columns.ts";
import { pretty } from "../dataset-io.ts";
import { DuckdbWriter, type WriteOptions } from "./duckdb.ts";
import { cellRanges, shardAssigner, shardCount, sortShard } from "./layout.ts";
import { readShard, type ShardEntry } from "./reader.ts";
import { COMMITTED_SPLITS, type HashIndex, HASHES_FILE } from "../dataset-reader.ts";
import { buildSnapshotManifest, checkFixtureAgreement, datasetPlan, eachSnapshotShard, shardsInDirectory } from "./snapshot.ts";

type Split = ShardEntry["split"];

/** The records file of a split in a dataset directory: the generator's, else a committed fixture's. */
export function datasetRecordsFile(dir: string, split: string): string {
  for (const name of [`records.${split}.jsonl`, `teams-${split}.jsonl`]) if (existsSync(join(dir, name))) return join(dir, name);
  return refuse(`${dir} has no records file for the ${split} split`);
}

/** The lines of a text file one at a time (a split can be larger than a JavaScript string). */
export async function* fileLines(path: string): AsyncGenerator<string> {
  let rest = "";
  for await (const chunk of createReadStream(path, { encoding: "utf8", highWaterMark: 1 << 22 })) {
    const parts = (rest + (chunk as string)).split("\n");
    rest = parts.pop() as string;
    yield* parts;
  }
  if (rest !== "") yield rest;
}

export interface ConvertOptions {
  /** The dataset directory: `manifest.json` and one records file per split. */
  datasetDir: string;
  /** Where the snapshot goes: `snapshot.json` and `pub/v1/shards/<sha256>`. */
  outDir: string;
  labelSet: TeamLabelSet;
  shardKey: string;
  snapshotId: string;
  splits: readonly Split[];
  writer: DuckdbWriter;
  parentSnapshotHash?: string;
  write?: WriteOptions;
  /** Rows per shard; 2^18 unless a test wants several shards of a small split. */
  shardRows?: number;
}

export interface ConvertResult {
  snapshot: TeamSnapshotManifest;
  /** Where `snapshot.json` was written. */
  snapshotFile: string;
}

const append = (fds: Map<number, number>, dir: string, shard: number, text: string): void => {
  let fd = fds.get(shard);
  if (fd === undefined) {
    fd = openSync(join(dir, `shard-${shard}.rows`), "a");
    fds.set(shard, fd);
  }
  writeSync(fd, text);
};

interface Staged {
  stratum: number;
  canonical_team_hash: string;
  record_hash: string;
  row: string;
}

/** Pass one for a split: stream, check, assign, stage. Returns the shard count. */
async function stage(file: string, split: Split, manifest: TeamDatasetManifest, options: ConvertOptions, scratch: string): Promise<number> {
  const plan = manifest.splits.find((s) => s.split === split) ?? refuse(`the manifest has no ${split} split`);
  const shards = shardCount(plan.teams, options.shardRows);
  const assign = shardAssigner(options.shardKey, Math.max(shards, 1));
  const strata = datasetPlan(manifest, options.labelSet);
  const fds = new Map<number, number>();
  let rows = 0;
  try {
    for await (const line of fileLines(file)) {
      const record = JSON.parse(line) as TeamRecord;
      let canonical: string;
      try {
        canonical = recordJson(record);
      } catch (error) {
        if (error instanceof ShardError) throw error;
        return refuse(`${file} line ${rows + 1} is not a TeamRecord`);
      }
      if (canonical !== line) refuse(`${file} line ${rows + 1} is not in canonical (JCS) form`);
      if (record.origin.split !== split || record.origin.dataset_id !== manifest.dataset_id) refuse(`${file} line ${rows + 1}: its origin is ${record.origin.dataset_id}/${record.origin.split}`);
      const stratum = strata.stratumOf(record.labels);
      const row = JSON.stringify(rowOf(record, stratum));
      const hash = createHash("sha256").update(line, "utf8").digest("hex");
      append(fds, scratch, assign(record.canonical_team_hash), `${stratum}\t${record.canonical_team_hash}\t${hash}\t${row}\n`);
      rows += 1;
    }
  } finally {
    for (const fd of fds.values()) closeSync(fd);
  }
  if (rows !== plan.teams) refuse(`${file} holds ${rows} records, the manifest says ${plan.teams}`);
  return shards;
}

/** The rows of a sorted shard as JSON Lines, written in chunks (a shard's text is near the largest string a JavaScript engine can hold). */
function writeRows(file: string, rows: readonly Staged[]): void {
  const fd = openSync(file, "w");
  try {
    for (let from = 0; from < rows.length; from += 4096) writeSync(fd, rows.slice(from, from + 4096).map((r) => `${r.row}\n`).join(""));
  } finally {
    closeSync(fd);
  }
}

async function* staged(file: string): AsyncGenerator<Staged> {
  for await (const line of fileLines(file)) {
    const [stratum, canonical, hash, ...row] = line.split("\t") as [string, string, string, ...string[]];
    yield { stratum: Number(stratum), canonical_team_hash: canonical, record_hash: hash, row: row.join("\t") };
  }
}

/**
 * Convert the splits of a dataset directory into a snapshot. The manifest is
 * read through the bridge's reader; the label set must be the one it names.
 */
export async function convertDataset(options: ConvertOptions): Promise<ConvertResult> {
  const manifest = readTeamDatasetManifest(JSON.parse(readFileSync(join(options.datasetDir, "manifest.json"), "utf8")), { labelSet: options.labelSet });
  const columns: readonly Column[] = columnsFor(options.labelSet);
  const plan = datasetPlan(manifest, options.labelSet);
  const objects = join(options.outDir, "pub", "v1", "shards");
  mkdirSync(objects, { recursive: true });
  const scratch = mkdtempSync(join(tmpdir(), "team-shards-"));
  const entries: ShardEntry[] = [];
  try {
    for (const split of options.splits) {
      const shards = await stage(datasetRecordsFile(options.datasetDir, split), split, manifest, options, scratch);
      for (let k = 0; k < shards; k++) {
        const staging = join(scratch, `shard-${k}.rows`);
        if (!existsSync(staging)) refuse(`${split}: shard ${k} of ${shards} received no row; use another shard key`);
        const rows: Staged[] = [];
        for await (const row of staged(staging)) rows.push(row);
        sortShard(rows);
        const rowsFile = join(scratch, `shard-${k}.jsonl`);
        writeRows(rowsFile, rows);
        const temp = join(scratch, `shard-${k}.parquet`);
        options.writer.writeShard(rowsFile, temp, columns, options.write);
        const bytes = readFileSync(temp);
        const sha = createHash("sha256").update(bytes).digest("hex");
        const logical = shardLogicalHash(rows.map((r) => r.record_hash));
        const entry: ShardEntry = { object_key: `pub/v1/shards/${sha}`, file_sha256: sha, bytes: bytes.byteLength, logical_hash: logical, rows: rows.length, split, cell_ranges: cellRanges(rows.map((r) => r.stratum)) as ShardEntry["cell_ranges"] };
        await readShard(bytes, entry, plan, shardAssigner(options.shardKey, shards));
        renameSync(temp, join(objects, sha));
        rmSync(rowsFile);
        rmSync(staging);
        entries.push(entry);
      }
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
  const snapshot = buildSnapshotManifest({
    snapshot_id: options.snapshotId,
    ...(options.parentSnapshotHash === undefined ? {} : { parent_snapshot_hash: options.parentSnapshotHash }),
    dataset_hashes: [recordHash(manifest)],
    manifest: manifest.generator.manifest,
    vocabulary_hash: manifest.generator.vocabulary_hash,
    shard_key: options.shardKey,
    shards: entries,
    writer: options.writer.identity,
  });
  const snapshotFile = join(options.outDir, "snapshot.json");
  writeFileSync(snapshotFile, pretty(snapshot));
  return { snapshot, snapshotFile };
}

export interface VerifyOptions {
  snapshotFile: string;
  /** The directory that holds `pub/v1/shards/<sha256>`. */
  shardsDir: string;
  datasetDir: string;
  labelSet: TeamLabelSet;
  shardRows?: number;
}

/**
 * Verify-from-shards: read every shard of the snapshot (all the rules of
 * `readShard` and the layout), and check that the records come back to the
 * dataset's records file byte for byte (the JCS line of record `n` has the
 * SHA-256 of the shard record whose `origin.index` is `n`), and, for a
 * committed split, that the record hashes are those of `hashes.json`.
 * Returns the rows read per split.
 */
export async function verifySnapshot(options: VerifyOptions): Promise<Record<string, number>> {
  const snapshot = readTeamSnapshotManifest(JSON.parse(readFileSync(options.snapshotFile, "utf8")));
  const manifest = readTeamDatasetManifest(JSON.parse(readFileSync(join(options.datasetDir, "manifest.json"), "utf8")), { labelSet: options.labelSet });
  if (!snapshot.dataset_hashes.includes(recordHash(manifest))) refuse("the snapshot does not list this dataset's hash");
  const byIndex = new Map<Split, string[]>();
  for await (const { entry, read } of eachSnapshotShard(snapshot, shardsInDirectory(options.shardsDir), datasetPlan(manifest, options.labelSet), options.shardRows)) {
    const slots = byIndex.get(entry.split) ?? [];
    byIndex.set(entry.split, slots);
    read.records.forEach((record, k) => {
      const at = record.origin.index;
      if (slots[at] !== undefined) refuse(`${entry.split}: origin index ${at} is in two records`);
      slots[at] = read.recordHashes[k] as string;
    });
  }
  const rows: Record<string, number> = {};
  for (const [split, slots] of byIndex) {
    if (slots.includes(undefined as never)) refuse(`${split}: the origin indices of the shards' records have gaps (${slots.length} slots)`);
    let n = 0;
    for await (const line of fileLines(datasetRecordsFile(options.datasetDir, split))) {
      if (slots[n] !== createHash("sha256").update(line, "utf8").digest("hex")) refuse(`${split}: record ${n} of the shards is not the dataset's record ${n}`);
      n += 1;
    }
    if (n !== slots.length) refuse(`${split}: the shards hold ${slots.length} records, the dataset file ${n}`);
    rows[split] = n;
    const hashesFile = join(options.datasetDir, HASHES_FILE);
    if ((COMMITTED_SPLITS as readonly string[]).includes(split) && existsSync(hashesFile)) {
      checkFixtureAgreement(split as "dev" | "validation", slots, JSON.parse(readFileSync(hashesFile, "utf8")) as HashIndex);
    }
  }
  return rows;
}
