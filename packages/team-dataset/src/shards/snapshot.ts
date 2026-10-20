// The snapshot manifest of a set of shards (ADR-013 section 4): built from the
// shard entries, checked by the same reader (`readTeamSnapshotManifest`) a
// consumer uses, and the checks that need the shard files too.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  logicalSnapshotHash,
  readTeamSnapshotManifest,
  type TeamDatasetManifest,
  type TeamLabelSet,
  type TeamSnapshotManifest,
} from "@areneon/showdown-bridge";
import { type LabelEntry, PARQUET_SCHEMA_VERSION, refuse, type RowPlan } from "./columns.ts";
import { SHARD_ROWS, shardAssigner, shardCount } from "./layout.ts";
import { readShard, type ShardEntry, type ShardRead } from "./reader.ts";
import { cellOf } from "../labeller.ts";
import type { HashIndex } from "../dataset-reader.ts";

const SPLIT_ORDER = { dev: 0, validation: 1, training: 2 } as const;
const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** The stratum rule of one dataset: a base cell is its own stratum unless a merged group owns it. */
export function datasetPlan(manifest: TeamDatasetManifest, labelSet: TeamLabelSet): RowPlan {
  const owner = new Map<number, number>();
  for (const group of manifest.merged_cells ?? []) for (const cell of group.cells) owner.set(cell, group.stratum);
  return {
    labelSet,
    stratumOf: (labels: readonly LabelEntry[]) => {
      const cell = cellOf(labelSet, labels);
      return owner.get(cell) ?? cell;
    },
  };
}

export interface SnapshotInput {
  snapshot_id: string;
  parent_snapshot_hash?: string;
  dataset_hashes: string[];
  manifest: TeamSnapshotManifest["manifest"];
  vocabulary_hash: string;
  shard_key: string;
  shards: ShardEntry[];
  writer: TeamSnapshotManifest["writer"];
}

/** Build and check a snapshot manifest: shards in fixed order, cells summed, `logical_snapshot_hash` computed. */
export function buildSnapshotManifest(input: SnapshotInput): TeamSnapshotManifest {
  const shards = [...input.shards].sort((a, b) => SPLIT_ORDER[a.split] - SPLIT_ORDER[b.split] || codeUnits(a.logical_hash, b.logical_hash));
  const totals = new Map<string, { split: ShardEntry["split"]; stratum: number; rows: number }>();
  for (const shard of shards) {
    for (const range of shard.cell_ranges) {
      const key = `${shard.split}:${range.stratum}`;
      const cell = totals.get(key) ?? { split: shard.split, stratum: range.stratum, rows: 0 };
      cell.rows += range.rows;
      totals.set(key, cell);
    }
  }
  const cells = [...totals.values()].sort((a, b) => SPLIT_ORDER[a.split] - SPLIT_ORDER[b.split] || a.stratum - b.stratum);
  const draft: TeamSnapshotManifest = {
    schema_version: 1,
    data_class: "synthetic",
    snapshot_id: input.snapshot_id,
    ...(input.parent_snapshot_hash === undefined ? {} : { parent_snapshot_hash: input.parent_snapshot_hash }),
    dataset_hashes: [...input.dataset_hashes].sort(codeUnits) as TeamSnapshotManifest["dataset_hashes"],
    manifest: input.manifest,
    vocabulary_hash: input.vocabulary_hash,
    logical_snapshot_hash: "0".repeat(64),
    shard_key: input.shard_key,
    shards: shards as TeamSnapshotManifest["shards"],
    cells: cells as TeamSnapshotManifest["cells"],
    writer: input.writer,
    parquet_schema_version: PARQUET_SCHEMA_VERSION,
  };
  draft.logical_snapshot_hash = logicalSnapshotHash(draft);
  return readTeamSnapshotManifest(draft);
}

/** A shard file as a store would serve it: by its object key. */
export type ShardSource = (entry: ShardEntry) => Uint8Array;

/** The files of a directory laid out as the object keys say (`<dir>/pub/v1/shards/<sha256>`). */
export const shardsInDirectory =
  (dir: string): ShardSource =>
  (entry) =>
    readFileSync(join(dir, entry.object_key));

export interface SnapshotRead {
  /** The records of each shard, in the manifest's shard order (split, then logical hash). */
  shards: ShardRead[];
}

/**
 * Read the shards of a snapshot one at a time, each checked against the
 * manifest: the file's SHA-256 and size, the decode and rebuild rules of
 * `readShard`, and the layout (each split's shards are one-to-one with the
 * indices 0..S-1 of the HMAC rule, S = ceil(rows / `shardRows`), 2^18 unless a
 * test says otherwise). Only one decoded shard is alive at a time if the
 * caller does not keep them, which is how a worker reads a snapshot.
 */
export async function* eachSnapshotShard(
  snapshot: TeamSnapshotManifest,
  source: ShardSource,
  plan: RowPlan,
  shardRows: number = SHARD_ROWS,
): AsyncGenerator<{ entry: ShardEntry; read: ShardRead }> {
  for (const split of ["dev", "validation", "training"] as const) {
    const entries = snapshot.shards.filter((s) => s.split === split);
    if (entries.length === 0) continue;
    const rows = entries.reduce((n, e) => n + e.rows, 0);
    if (shardCount(rows, shardRows) !== entries.length) refuse(`${split}: ${rows} rows need ${shardCount(rows, shardRows)} shards, the manifest lists ${entries.length}`);
    const assign = shardAssigner(snapshot.shard_key, entries.length);
    const indices = new Set<number>();
    for (const entry of entries) {
      const bytes = source(entry);
      if (bytes.byteLength !== entry.bytes) refuse(`shard ${entry.logical_hash}: the file has ${bytes.byteLength} bytes, the manifest says ${entry.bytes}`);
      if (createHash("sha256").update(bytes).digest("hex") !== entry.file_sha256) refuse(`shard ${entry.logical_hash}: the file's SHA-256 is not ${entry.file_sha256}`);
      const read = await readShard(bytes, entry, plan, assign);
      if (indices.has(read.index as number)) refuse(`shard ${entry.logical_hash}: shard index ${read.index} is held by two shards of ${split}`);
      indices.add(read.index as number);
      yield { entry, read };
    }
  }
}

/** Every shard of a snapshot, read and kept (small snapshots and tests; a worker uses `eachSnapshotShard`). */
export async function readSnapshotShards(snapshot: TeamSnapshotManifest, source: ShardSource, plan: RowPlan, shardRows: number = SHARD_ROWS): Promise<SnapshotRead> {
  const shards: ShardRead[] = [];
  for await (const { read } of eachSnapshotShard(snapshot, source, plan, shardRows)) shards.push(read);
  return { shards };
}

/**
 * Fixture and shard agreement (ADR-013 section 4): the committed JSONL file of
 * a split and the snapshot's shards of that split must hold the same sorted
 * list of `record_hash`es. `hashes` is the (already verified) index of the
 * committed dataset; `shardHashes` are the `record_hash`es of every record of
 * the split's shards, in any order.
 */
export function checkFixtureAgreement(split: "dev" | "validation", shardHashes: readonly string[], hashes: HashIndex): void {
  const fromShards = [...shardHashes].sort(codeUnits);
  const fixture = hashes.splits[split].record_hashes;
  if (fromShards.length !== fixture.length) refuse(`${split}: the shards hold ${fromShards.length} records, the committed fixture ${fixture.length}`);
  const at = fromShards.findIndex((h, k) => h !== fixture[k]);
  if (at >= 0) refuse(`${split}: record hash ${fromShards[at]} differs from the committed fixture's ${fixture[at]} at position ${at}`);
}
