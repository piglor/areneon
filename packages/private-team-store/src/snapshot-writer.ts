// The private snapshot writer (ADR-015 sections 2 and 3): the next record
// list, the hold-out key and the parent's hash in; the Parquet shards, the
// PrivateSnapshotManifest and the snapshot hash out. The shards are written by
// the pinned DuckDB command-line tool (a child process, never a library) on
// the private path of `DuckdbWriter`: an explicit minimal environment, a
// scratch directory inside the caller's working directory as HOME, TMPDIR and
// cwd, no history file, and a fixed message instead of the tool's stderr.
//
// Intermediates (the rows file and the shard file, both with team text in the
// clear) exist only inside a fresh 0700 directory under the working directory
// the caller names, which `assertWorkDir` has accepted (`private-data/` or
// tmpfs), and are removed afterwards, on failure too. Nothing is returned that
// could not be uploaded: the shard bytes, the manifest and its hash.
//
// Every shard is read back with the engine-free private shard reader before
// the manifest is built, so the manifest describes what the reader will see.
// The full verification (engine round trip of each paste, hold-out flags,
// digests) is `readPrivateSnapshot`, which the import runs on this output.

import { createHash } from "node:crypto";
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeSync } from "node:fs";
import { join } from "node:path";
import {
  checkPrivateRecordSet,
  HOLDOUT_PPM,
  HOLDOUT_RULE,
  holdoutFlag,
  type PrivateSnapshotManifest,
  type PrivateTeamRecord,
  privateLogicalSnapshotHash,
  privateSnapshotHash,
  readPrivateSnapshotManifest,
  recordHash,
  shardLogicalHash,
  type TeamLabelSet,
} from "@areneon/showdown-bridge";
import {
  type DuckdbPin,
  type DuckdbWriter,
  layoutPrivate,
  PARQUET_SCHEMA_VERSION,
  privateCellRanges,
  privateColumnsFor,
  privateRowOf,
  readPrivateShard,
  removeTree,
  shardAssigner,
  type WriteOptions,
} from "@areneon/team-dataset";
import { SnapshotIoError } from "./snapshot-io-error.ts";
import { labelPlan } from "./snapshot-plan.ts";
import { assertWorkDir } from "./work-dir.ts";

export interface WriteSnapshotInput {
  /** The whole next record list (one record per team_id), flags set. */
  readonly records: readonly PrivateTeamRecord[];
  /** The snapshot's hold-out key: generated at the first import, the parent's after. */
  readonly holdoutKey: string;
  /** The parent's snapshot hash, or null for the first snapshot. */
  readonly parentSnapshotHash: string | null;
  readonly snapshotId: string;
  readonly labelSet: TeamLabelSet;
  /** The version manifest the teams were validated under (`resolveFormat`). */
  readonly manifest: PrivateSnapshotManifest["manifest"];
  readonly vocabularyHash: string;
  /** The public shard key that fixes the layout, 64 lower-case hex characters. */
  readonly shardKey: string;
  readonly writer: DuckdbWriter;
  /** The pin the writer's binary must be (version and SHA-256). */
  readonly pin: DuckdbPin;
  /** A `private-data/` or tmpfs directory, mode 0700 (see `assertWorkDir`). */
  readonly workDir: string;
  /** Rows per shard; the default is 2^18. Tests make it small. */
  readonly shardRows?: number;
  /** Row group size and compression; the private path is added by this function. */
  readonly write?: Omit<WriteOptions, "isolated">;
}

export interface WrittenSnapshot {
  readonly manifest: PrivateSnapshotManifest;
  /** SHA-256 of the manifest's JCS bytes: the name `snapshots/<hash>/` in the bucket. */
  readonly snapshotHash: string;
  /** The shard files by manifest index (`snapshots/<hash>/shards/<index>.parquet`). */
  readonly shards: readonly Uint8Array[];
}

const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** Run a step; whatever it throws becomes the fixed refusal `code` (the text may hold a paste). */
function step<T>(code: ConstructorParameters<typeof SnapshotIoError>[0], run: () => T): T {
  try {
    return run();
  } catch {
    throw new SnapshotIoError(code);
  }
}

function writeLines(file: string, lines: readonly string[]): void {
  const fd = openSync(file, "w", 0o600);
  try {
    for (let from = 0; from < lines.length; from += 4096) writeSync(fd, lines.slice(from, from + 4096).map((l) => `${l}\n`).join(""));
  } finally {
    closeSync(fd);
  }
}

/** Write a snapshot. Throws `SnapshotIoError` and nothing else; leaves nothing behind in `workDir`. */
export async function writePrivateSnapshot(input: WriteSnapshotInput): Promise<WrittenSnapshot> {
  const workDir = assertWorkDir(input.workDir);
  step("writer_not_pinned", () => input.writer.requirePinned(input.pin));
  if (input.records.length === 0) throw new SnapshotIoError("no_records");
  step("records_invalid", () => checkPrivateRecordSet(input.records));
  const ppm = HOLDOUT_PPM;
  const flags = step("holdout_key_invalid", () => input.records.map((r) => holdoutFlag(input.holdoutKey, r.team_id, ppm)));
  if (input.records.some((r, k) => r.holdout !== flags[k])) throw new SnapshotIoError("holdout_mismatch");

  const plan = labelPlan(input.labelSet);
  const columns = privateColumnsFor(input.labelSet);
  const keyed = step("labels_invalid", () =>
    input.records.map((record) => ({ record, stratum: plan.stratumOf(record.labels), canonical_team_hash: record.canonical_team_hash, team_id: record.team_id })),
  );
  const shards = step("layout_failed", () => layoutPrivate(keyed, input.shardKey, input.shardRows));
  const assign = shardAssigner(input.shardKey, shards.length);

  const scratch = step("writer_failed", () => mkdtempSync(join(workDir, "snapshot-")));
  const built: { file: Uint8Array; entry: PrivateSnapshotManifest["shards"][number] }[] = [];
  try {
    for (const [k, shard] of shards.entries()) {
      const rowsFile = join(scratch, `shard-${k}.jsonl`);
      const out = join(scratch, `shard-${k}.parquet`);
      step("writer_failed", () => writeLines(rowsFile, shard.map((s) => JSON.stringify(privateRowOf(s.record, s.stratum)))));
      step("writer_failed", () => input.writer.writeShard(rowsFile, out, columns, { ...input.write, isolated: { dir: scratch } }));
      const bytes = step("writer_failed", () => new Uint8Array(readFileSync(out)));
      step("writer_failed", () => rmSync(rowsFile));
      step("writer_failed", () => rmSync(out));
      const recordHashes = shard.map((s) => recordHash(s.record));
      const logical = shardLogicalHash(recordHashes);
      const entry = {
        index: 0,
        file_sha256: createHash("sha256").update(bytes).digest("hex"),
        bytes: bytes.byteLength,
        logical_hash: logical,
        rows: shard.length,
        split: "owner" as const,
        cell_ranges: privateCellRanges(shard.map((s) => ({ stratum: s.stratum, status: s.record.status }))),
      };
      // The reader's own view of the file: any difference from what was meant to be written is a failure.
      try {
        await readPrivateShard(bytes, entry, plan, assign);
      } catch {
        throw new SnapshotIoError("shard_invalid", { shard: k });
      }
      built.push({ file: bytes, entry });
    }
  } catch (error) {
    // Removal never replaces the error that stopped the write, and says nothing about the contents.
    removeTree(scratch);
    throw error;
  }
  if (!removeTree(scratch)) throw new SnapshotIoError("cleanup_failed");

  built.sort((a, b) => codeUnits(a.entry.logical_hash, b.entry.logical_hash));
  const entries = built.map((b, index) => ({ ...b.entry, index }));
  const active = new Map<number, number>();
  for (const entry of entries) for (const range of entry.cell_ranges) if (range.active_rows > 0) active.set(range.stratum, (active.get(range.stratum) ?? 0) + range.active_rows);
  const cells = [...active.entries()].sort((a, b) => a[0] - b[0]).map(([stratum, rows]) => ({ split: "owner" as const, stratum, rows }));
  const activeCount = input.records.filter((r) => r.status === "active").length;
  const draft = {
    schema_version: 1 as const,
    data_class: "private" as const,
    snapshot_id: input.snapshotId,
    ...(input.parentSnapshotHash === null ? { first: true as const } : { parent_snapshot_hash: input.parentSnapshotHash }),
    manifest: input.manifest,
    vocabulary_hash: input.vocabularyHash,
    label_set_hash: recordHash(input.labelSet),
    logical_snapshot_hash: "0".repeat(64),
    shard_key: input.shardKey,
    shards: entries,
    cells,
    records: { active: activeCount, retired: input.records.length - activeCount },
    holdout: { rule: HOLDOUT_RULE, key: input.holdoutKey, ppm },
    writer: input.writer.identity,
    parquet_schema_version: PARQUET_SCHEMA_VERSION,
  };
  const manifest = step("manifest_invalid", () => {
    const sealed = { ...draft, logical_snapshot_hash: privateLogicalSnapshotHash(draft as unknown as PrivateSnapshotManifest) };
    return readPrivateSnapshotManifest(sealed);
  });
  return { manifest, snapshotHash: privateSnapshotHash(manifest), shards: built.map((b) => b.file) };
}
