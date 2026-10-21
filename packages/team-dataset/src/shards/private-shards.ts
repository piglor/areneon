// The private column set and layout of a team shard (ADR-015 section 2): the
// d1 layout (ADR-013 section 4) with the PrivateTeamRecord's columns in place
// of the TeamRecord's, `data_class` private, one split (`owner`).
//
// What is reused from the public layout: the shard count (`shardCount`), the
// HMAC placement by canonical team hash (`shardAssigner`), the stratum cell
// ranges (`cellRanges`), the `labels` list column and the derived `stratum`
// and `label:<id>` columns, the int64 rule, and the record identities
// (`recordHash`, `shardLogicalHash`). What differs:
//   - three more column kinds: `supersedes_record_hash` may be null (revision
//     1), `tags` is a list of strings and `holdout` is a boolean;
//   - a retired record may share its team's canonical hash with an active one
//     (a revision chain is one record per `team_id`), so the row order is
//     stratum, canonical hash, then `team_id`, and the same canonical hash in
//     two rows is not an error here (the record-set rules are the snapshot
//     reader's);
//   - each cell range counts its active rows.
//
// A private shard holds team content, so no message in this module repeats a
// value read from a row or a file: a refusal names a column of the layout or a
// row position, never a paste, tag, label, id or file text.

import { ajvFor } from "@areneon/schema/ajv";
import privateTeamRecordSchema from "@areneon/schema/contracts/v1/private-team-record.schema.json" with { type: "json" };
import type { PrivateSnapshotManifest, PrivateTeamRecord, TeamLabelSet } from "@areneon/showdown-bridge";
import { type Column, labelColumn, type LabelEntry, refuse, type RowPlan, ShardError } from "./columns.ts";
import { cellRanges, shardAssigner, shardCount } from "./layout.ts";

/** The columns of a private shard for a label set, in file order. */
export function privateColumnsFor(labelSet: TeamLabelSet): Column[] {
  return [
    { name: "schema_version", type: "BIGINT" },
    { name: "data_class", type: "VARCHAR" },
    { name: "team_id", type: "VARCHAR" },
    { name: "revision", type: "BIGINT" },
    { name: "supersedes_record_hash", type: "NULLABLE_VARCHAR" },
    { name: "team_hash", type: "VARCHAR" },
    { name: "canonical_team_hash", type: "VARCHAR" },
    { name: "paste", type: "VARCHAR" },
    { name: "status", type: "VARCHAR" },
    { name: "tags", type: "VARCHAR_LIST" },
    { name: "holdout", type: "BOOLEAN" },
    { name: "labels", type: "LABELS" },
    { name: "stratum", type: "BIGINT" },
    ...labelSet.labels.map((l) => ({ name: labelColumn(l.label_id), type: "VARCHAR" as const })),
  ];
}

export type PrivateRowValue = string | number | boolean | null | LabelEntry[] | string[];
/** One private row as a plain object keyed by column name, ready to be written. */
export type PrivateRow = Record<string, PrivateRowValue>;

/** The row of a private record: its fields, its stratum and its label columns. */
export function privateRowOf(record: PrivateTeamRecord, stratum: number): PrivateRow {
  const row: PrivateRow = {
    schema_version: record.schema_version,
    data_class: record.data_class,
    team_id: record.team_id,
    revision: record.revision,
    supersedes_record_hash: record.supersedes_record_hash,
    team_hash: record.team_hash,
    canonical_team_hash: record.canonical_team_hash,
    paste: record.paste,
    status: record.status,
    tags: [...record.tags],
    holdout: record.holdout,
    labels: record.labels.map((l) => ({ label_id: l.label_id, value: l.value })),
    stratum,
  };
  for (const l of record.labels) row[labelColumn(l.label_id)] = l.value;
  return row;
}

const validateRecord = ajvFor("contract").compile(privateTeamRecordSchema);
const SAFE = BigInt(Number.MAX_SAFE_INTEGER);

function int(value: unknown, column: string): number {
  if (typeof value === "bigint") {
    if (value > SAFE || value < -SAFE) return refuse(`${column}: outside +-(2^53-1)`);
    return Number(value);
  }
  if (typeof value === "number" && Number.isSafeInteger(value)) return value;
  return refuse(`${column}: not an int64 within +-(2^53-1)`);
}

function text(value: unknown, column: string): string {
  if (typeof value !== "string") return refuse(`${column}: not a string`);
  return value;
}

/** Rebuild one private record from a row as the Parquet reader returns it (`row` is checked, never trusted). */
export function privateRecordOf(row: Record<string, unknown>, plan: RowPlan, where: string): { record: PrivateTeamRecord; stratum: number } {
  try {
    return rebuild(row, plan);
  } catch (error) {
    // The message of a refusal here is built from column names only; anything else (a library error) is dropped.
    throw new ShardError(`${where}: ${error instanceof ShardError ? error.message : "the row cannot be read"}`);
  }
}

function rebuild(row: Record<string, unknown>, plan: RowPlan): { record: PrivateTeamRecord; stratum: number } {
  const labelsRaw = row.labels;
  if (!Array.isArray(labelsRaw)) return refuse("labels: not a list");
  const set = plan.labelSet.labels;
  if (labelsRaw.length !== set.length) return refuse("labels: not one entry per label of the label set");
  const labels = labelsRaw.map((entry, k): LabelEntry => {
    if (typeof entry !== "object" || entry === null) return refuse(`labels[${k}]: not a struct`);
    if (Object.keys(entry).sort().join(",") !== "label_id,value") return refuse(`labels[${k}]: not exactly label_id and value`);
    const { label_id, value } = entry as Record<string, unknown>;
    return { label_id: text(label_id, `labels[${k}].label_id`), value: text(value, `labels[${k}].value`) };
  });
  labels.forEach((l, k) => {
    const label = set[k] as TeamLabelSet["labels"][number];
    if (l.label_id !== label.label_id) refuse(`labels[${k}]: not the label set's label`);
    if (l.value !== label.default && !label.values.some((v) => v.value === l.value)) refuse(`labels[${k}]: not one of the label's values`);
    if (row[labelColumn(l.label_id)] !== l.value) refuse(`labels[${k}]: its derived column differs`);
  });
  const tagsRaw = row.tags;
  if (!Array.isArray(tagsRaw)) return refuse("tags: not a list");
  const supersedes = row.supersedes_record_hash;
  if (supersedes !== null && typeof supersedes !== "string") return refuse("supersedes_record_hash: neither a string nor null");
  if (typeof row.holdout !== "boolean") return refuse("holdout: not a boolean");
  const record = {
    schema_version: int(row.schema_version, "schema_version"),
    data_class: text(row.data_class, "data_class"),
    team_id: text(row.team_id, "team_id"),
    revision: int(row.revision, "revision"),
    supersedes_record_hash: supersedes,
    team_hash: text(row.team_hash, "team_hash"),
    canonical_team_hash: text(row.canonical_team_hash, "canonical_team_hash"),
    paste: text(row.paste, "paste"),
    status: text(row.status, "status"),
    tags: tagsRaw.map((t, k) => text(t, `tags[${k}]`)),
    holdout: row.holdout,
    labels,
  };
  if (!validateRecord(record)) {
    // Path and keyword only: an error's params can hold a key name from the file.
    const errors = (validateRecord.errors ?? []) as { instancePath: string; keyword: string }[];
    return refuse(`not a PrivateTeamRecord: ${errors.map((e) => `${e.instancePath || "/"} ${e.keyword}`).join("; ")}`);
  }
  const stratum = int(row.stratum, "stratum");
  if (stratum !== plan.stratumOf(labels)) refuse("stratum: not the one the labels give");
  return { record: record as unknown as PrivateTeamRecord, stratum };
}

/** What sorting a private shard needs of a row. */
export interface PrivateKeyed {
  stratum: number;
  canonical_team_hash: string;
  team_id: string;
}

const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** The order of rows in a private shard: stratum, then canonical hash, then `team_id` (code units). */
export const byPrivateOrder = (a: PrivateKeyed, b: PrivateKeyed): number =>
  a.stratum - b.stratum || codeUnits(a.canonical_team_hash, b.canonical_team_hash) || codeUnits(a.team_id, b.team_id);

/** Each stratum's contiguous rows with the number of active ones, for rows already in shard order. */
export function privateCellRanges(rows: readonly { stratum: number; status: string }[]): PrivateSnapshotManifest["shards"][number]["cell_ranges"] {
  const ranges = cellRanges(rows.map((r) => r.stratum));
  return ranges.map((range) => ({
    ...range,
    active_rows: rows.slice(range.first_row, range.first_row + range.rows).filter((r) => r.status === "active").length,
  })) as PrivateSnapshotManifest["shards"][number]["cell_ranges"];
}

/**
 * Lay out the rows of a snapshot: `shardCount(rows)` shards by the HMAC of the
 * canonical hash, each sorted in private order. A shard that receives no row
 * refuses (the manifest cannot hold an empty shard).
 */
export function layoutPrivate<T extends PrivateKeyed>(rows: readonly T[], shardKey: string, shardRows?: number): T[][] {
  const count = shardCount(rows.length, shardRows);
  const assign = shardAssigner(shardKey, Math.max(count, 1));
  const shards: T[][] = Array.from({ length: count }, () => []);
  for (const row of rows) (shards[assign(row.canonical_team_hash)] as T[]).push(row);
  shards.forEach((shard, k) => {
    if (shard.length === 0) refuse(`shard ${k} of ${count} received no row; use another shard key`);
    shard.sort(byPrivateOrder);
  });
  return shards;
}
