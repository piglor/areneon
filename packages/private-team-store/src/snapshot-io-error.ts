// The typed refusal of the private snapshot writer and reader (ADR-015 section
// 3, "No leaks"). A refusal is a code plus, where it helps, the shard index and
// the row number inside it: never text taken from a paste, a tag, a label, a
// file, DuckDB or the engine. The messages are fixed strings, so an error can
// be printed or logged without leaking a team. Whatever a lower layer threw is
// dropped (not attached as `cause`: a cause is printed by the runtime).

export const SNAPSHOT_IO_REFUSALS = [
  "work_dir_unsafe",
  "no_records",
  "records_invalid",
  "holdout_key_invalid",
  "holdout_mismatch",
  "labels_invalid",
  "writer_not_pinned",
  "layout_failed",
  "writer_failed",
  "manifest_invalid",
  "parent_is_self",
  "snapshot_hash_mismatch",
  "label_set_mismatch",
  "version_mismatch",
  "shard_missing",
  "shard_size_mismatch",
  "shard_digest_mismatch",
  "shard_file_repeated",
  "shard_layout_invalid",
  "shard_invalid",
  "record_invalid",
] as const;
export type SnapshotIoRefusal = (typeof SNAPSHOT_IO_REFUSALS)[number];

const MESSAGES: Readonly<Record<SnapshotIoRefusal, string>> = {
  work_dir_unsafe: "the working directory is not an absolute, real directory owned by this user with mode 0700 under a private-data path or on tmpfs",
  no_records: "a snapshot needs at least one record",
  records_invalid: "the records are not a valid private record set",
  holdout_key_invalid: "the hold-out key is not 64 lower-case hex characters",
  holdout_mismatch: "a record's hold-out flag differs from the hold-out rule's value for its team_id",
  labels_invalid: "a record's labels are not the label set's",
  writer_not_pinned: "the DuckDB binary is not the pinned one",
  layout_failed: "the records cannot be laid out in shards",
  writer_failed: "the shard writer failed; its output is withheld because it can quote a paste",
  manifest_invalid: "the manifest is not a valid private snapshot manifest",
  parent_is_self: "parent_snapshot_hash is the snapshot's own hash",
  snapshot_hash_mismatch: "the manifest does not hash to the snapshot hash it was fetched under",
  label_set_mismatch: "the manifest's label_set_hash is not the label set given",
  version_mismatch: "the manifest's parquet_schema_version is not the one this reader reads",
  shard_missing: "a shard file could not be fetched",
  shard_size_mismatch: "a shard file's size is not the manifest's",
  shard_digest_mismatch: "a shard file's SHA-256 is not the manifest's",
  shard_file_repeated: "two shards of the manifest have the same file_sha256",
  shard_layout_invalid: "the shards do not match the layout the shard key and row count give",
  shard_invalid: "a shard file is not a valid private shard of this manifest",
  record_invalid: "a record is refused by the private record reader or the pinned engine",
};

export interface SnapshotIoPosition {
  /** The manifest index of the shard, when a shard is at fault. */
  readonly shard: number | null;
  /** The 0-based row inside that shard, when a row is at fault. */
  readonly row: number | null;
}

export class SnapshotIoError extends Error {
  override name = "SnapshotIoError";
  readonly code: SnapshotIoRefusal;
  readonly position: SnapshotIoPosition;

  constructor(code: SnapshotIoRefusal, position: Partial<SnapshotIoPosition> = {}) {
    const at = { shard: position.shard ?? null, row: position.row ?? null };
    const where = at.shard === null ? "" : at.row === null ? ` (shard ${at.shard})` : ` (shard ${at.shard}, row ${at.row})`;
    super(`private snapshot refused${where}: ${MESSAGES[code]}`);
    this.code = code;
    this.position = at;
  }
}
