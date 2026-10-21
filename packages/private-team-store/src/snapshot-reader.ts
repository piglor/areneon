// readPrivateSnapshot (ADR-015 sections 2 and 3): the one check of a private
// snapshot, run by the import after writing, after downloading, and by every
// reader of the stored snapshot. Nothing in the manifest or a shard file is
// trusted. In order:
//   - the manifest through `readPrivateSnapshotManifest` (shape, shard order
//     and indices, tiling, sums, counts, recomputed logical hash), a parent
//     that is not the snapshot's own hash, the hash it was fetched under (always
//     given), the version manifest and vocabulary hash the caller expects (when
//     given), the label set and the Parquet layout version;
//   - the shard layout: `shardCount(rows)` shards, no two with one file digest;
//   - each shard file: its size and SHA-256, then the engine-free decode and
//     rebuild of `readPrivateShard` (columns, order, placement, logical hash,
//     cell ranges with active rows);
//   - each record again through `readPrivateTeamRecord` (the engine round trip
//     of its paste, both team hashes recomputed), its hold-out flag recomputed
//     from the manifest's key and refused on a mismatch;
//   - the record set (`checkPrivateRecordSet`).
// The counts need no check of their own: the manifest reader ties `records` and
// `cells` to the shards' cell ranges, and each shard read ties its ranges (with
// their active rows) to the rows it decoded.
// It never calls the object store: the shard files come through `shard`. Every
// refusal is a `SnapshotIoError` whose message is fixed; what a lower layer
// said is dropped, because it may quote a paste.

import { createHash } from "node:crypto";
import {
  checkPrivateRecordSet,
  type Engine,
  canonicalJson,
  holdoutFlag,
  type Json,
  type PrivateSnapshotManifest,
  type PrivateTeamRecord,
  privateSnapshotHash,
  readPrivateSnapshotManifest,
  readPrivateTeamRecord,
  recordHash,
  type TeamLabelSet,
} from "@areneon/showdown-bridge";
import { PARQUET_SCHEMA_VERSION, readPrivateShard, SHARD_ROWS, shardAssigner, shardCount } from "@areneon/team-dataset";
import { SnapshotIoError } from "./snapshot-io-error.ts";
import { labelPlan } from "./snapshot-plan.ts";

export interface ReadSnapshotInput {
  readonly engine: Engine;
  /** The parsed `manifest.json`. */
  readonly manifest: unknown;
  /** The shard file with this manifest index (`snapshots/<hash>/shards/<index>.parquet`). */
  readonly shard: (index: number) => Uint8Array | Promise<Uint8Array>;
  readonly labelSet: TeamLabelSet;
  /**
   * The snapshot hash the manifest was stored under (its bucket key, or the
   * hash a run pinned); the manifest must hash to it. Required: a reader that
   * takes whatever manifest it is handed has no anchor, and there is no way to
   * switch the check off.
   */
  readonly expectedSnapshotHash: string;
  /**
   * The version manifest and vocabulary hash the caller derives from the
   * pinned engine and rules; when given, the snapshot must carry exactly
   * these. (Left out only by a reader that inspects a snapshot made under
   * another pin.)
   */
  readonly expected?: { readonly manifest: PrivateSnapshotManifest["manifest"]; readonly vocabularyHash: string };
  /** Rows per shard the layout used; the default is 2^18. Tests make it small. */
  readonly shardRows?: number;
  /** Wall-clock limit of one shard's decode, in ms; the default is 300000. Tests make it short. */
  readonly decodeTimeoutMs?: number;
}

export interface SnapshotRead {
  readonly manifest: PrivateSnapshotManifest;
  readonly snapshotHash: string;
  /** Every record of every shard, sorted by team_id. */
  readonly records: PrivateTeamRecord[];
}

const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

export async function readPrivateSnapshot(input: ReadSnapshotInput): Promise<SnapshotRead> {
  let manifest: PrivateSnapshotManifest;
  try {
    manifest = readPrivateSnapshotManifest(input.manifest);
  } catch {
    throw new SnapshotIoError("manifest_invalid");
  }
  const snapshotHash = privateSnapshotHash(manifest);
  if (manifest.parent_snapshot_hash === snapshotHash) throw new SnapshotIoError("parent_is_self");
  if (input.expectedSnapshotHash !== snapshotHash) throw new SnapshotIoError("snapshot_hash_mismatch");
  if (manifest.label_set_hash !== recordHash(input.labelSet)) throw new SnapshotIoError("label_set_mismatch");
  if (input.expected !== undefined && (canonicalJson(manifest.manifest as unknown as Json) !== canonicalJson(input.expected.manifest as unknown as Json) || manifest.vocabulary_hash !== input.expected.vocabularyHash)) {
    throw new SnapshotIoError("version_manifest_mismatch");
  }
  if (manifest.parquet_schema_version !== PARQUET_SCHEMA_VERSION) throw new SnapshotIoError("version_mismatch");

  if (new Set(manifest.shards.map((s) => s.file_sha256)).size !== manifest.shards.length) throw new SnapshotIoError("shard_file_repeated");
  const totalRows = manifest.shards.reduce((n, s) => n + s.rows, 0);
  const shardRows = input.shardRows ?? SHARD_ROWS;
  let assign: (canonicalTeamHash: string) => number;
  try {
    if (shardCount(totalRows, shardRows) !== manifest.shards.length) throw new SnapshotIoError("shard_layout_invalid");
    assign = shardAssigner(manifest.shard_key, manifest.shards.length);
  } catch {
    throw new SnapshotIoError("shard_layout_invalid");
  }

  const plan = labelPlan(input.labelSet);
  const records: PrivateTeamRecord[] = [];
  const placed = new Set<number>();
  for (const entry of manifest.shards) {
    let bytes: Uint8Array;
    try {
      bytes = await input.shard(entry.index);
    } catch {
      throw new SnapshotIoError("shard_missing", { shard: entry.index });
    }
    if (bytes.byteLength !== entry.bytes) throw new SnapshotIoError("shard_size_mismatch", { shard: entry.index });
    if (createHash("sha256").update(bytes).digest("hex") !== entry.file_sha256) throw new SnapshotIoError("shard_digest_mismatch", { shard: entry.index });
    let read: Awaited<ReturnType<typeof readPrivateShard>>;
    try {
      read = await readPrivateShard(bytes, entry, plan, assign, input.decodeTimeoutMs);
    } catch {
      throw new SnapshotIoError("shard_invalid", { shard: entry.index });
    }
    if (placed.has(read.index as number)) throw new SnapshotIoError("shard_layout_invalid", { shard: entry.index });
    placed.add(read.index as number);
    read.records.forEach((record, row) => {
      try {
        readPrivateTeamRecord(input.engine, record);
      } catch {
        throw new SnapshotIoError("record_invalid", { shard: entry.index, row });
      }
      if (record.holdout !== holdoutFlag(manifest.holdout.key, record.team_id, manifest.holdout.ppm)) {
        throw new SnapshotIoError("holdout_mismatch", { shard: entry.index, row });
      }
      records.push(record);
    });
  }
  try {
    checkPrivateRecordSet(records);
  } catch {
    throw new SnapshotIoError("record_invalid");
  }
  records.sort((a, b) => codeUnits(a.team_id, b.team_id));
  return { manifest, snapshotHash, records };
}
