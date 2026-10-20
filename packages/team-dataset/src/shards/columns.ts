// The Parquet column layout of a team shard (ADR-013 section 4, parquet schema
// version 1) and the rule that rebuilds a TeamRecord from one row.
//
// One column per scalar field of the record, the `labels` list of
// {label_id, value} structs, and the derived columns: `stratum` and one
// `label:<label_id>` column per label (for pushdown in a SQL engine). The
// derived columns are never trusted: the reader recomputes them from the
// labels and refuses a row where they differ.
//
// Rebuild rules (a refusal is a `ShardError`):
//   - a required column is never null (every column of the record is required,
//     so no column here is optional); null, a missing column or an extra
//     column refuses the shard;
//   - integers are int64 and must lie within +-(2^53 - 1), so JSON and JCS
//     stay exact; a BigInt or number outside it refuses the shard;
//   - `labels` keeps the label set's order, which is significant;
//   - strings are UTF-8 text as given.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import teamRecordSchema from "@areneon/schema/contracts/v1/team-record.schema.json" with { type: "json" };
import type { TeamLabelSet, TeamRecord } from "@areneon/showdown-bridge";

/** The version of this column layout; it is in the manifest and in each file's key-value metadata. */
export const PARQUET_SCHEMA_VERSION = 1;
/** The key-value metadata key that carries `PARQUET_SCHEMA_VERSION` inside each shard file. */
export const SCHEMA_VERSION_KEY = "areneon_parquet_schema_version";

export class ShardError extends Error {
  override name = "ShardError";
}

export const refuse = (message: string): never => {
  throw new ShardError(message);
};

export type ColumnType = "VARCHAR" | "BIGINT" | "LABELS";
export interface Column {
  name: string;
  type: ColumnType;
}

/** The name of a label's derived column. */
export const labelColumn = (labelId: string): string => `label:${labelId}`;

/** The columns of a shard for a label set, in file order. */
export function columnsFor(labelSet: TeamLabelSet): Column[] {
  return [
    { name: "schema_version", type: "BIGINT" },
    { name: "data_class", type: "VARCHAR" },
    { name: "team_hash", type: "VARCHAR" },
    { name: "canonical_team_hash", type: "VARCHAR" },
    { name: "paste", type: "VARCHAR" },
    { name: "dataset_id", type: "VARCHAR" },
    { name: "split", type: "VARCHAR" },
    { name: "origin_index", type: "BIGINT" },
    { name: "origin_draw_counter", type: "BIGINT" },
    { name: "label_set_hash", type: "VARCHAR" },
    { name: "labels", type: "LABELS" },
    { name: "stratum", type: "BIGINT" },
    ...labelSet.labels.map((l) => ({ name: labelColumn(l.label_id), type: "VARCHAR" as const })),
  ];
}

export type LabelEntry = { label_id: string; value: string };
export type RowValue = string | number | LabelEntry[];
/** One row as a plain object keyed by column name, ready to be written. */
export type Row = Record<string, RowValue>;

/** The row of a record: its fields, its stratum and its label columns. */
export function rowOf(record: TeamRecord, stratum: number): Row {
  const row: Row = {
    schema_version: record.schema_version,
    data_class: record.data_class,
    team_hash: record.team_hash,
    canonical_team_hash: record.canonical_team_hash,
    paste: record.paste,
    dataset_id: record.origin.dataset_id,
    split: record.origin.split,
    origin_index: record.origin.index,
    origin_draw_counter: record.origin.draw_counter,
    label_set_hash: record.label_set_hash,
    labels: record.labels.map((l) => ({ label_id: l.label_id, value: l.value })),
    stratum,
  };
  for (const l of record.labels) row[labelColumn(l.label_id)] = l.value;
  return row;
}

/** What a row is rebuilt against. */
export interface RowPlan {
  labelSet: TeamLabelSet;
  /** The stratum of a record: the merged cell its labels fall in. Derived, so recomputed here. */
  stratumOf(labels: readonly LabelEntry[]): number;
}

const validateRecord = ajvFor("contract").compile(teamRecordSchema);
const SAFE = BigInt(Number.MAX_SAFE_INTEGER);

function int(value: unknown, column: string): number {
  if (typeof value === "bigint") {
    if (value > SAFE || value < -SAFE) return refuse(`${column}: ${value} is outside +-(2^53-1)`);
    return Number(value);
  }
  if (typeof value === "number" && Number.isSafeInteger(value)) return value;
  return refuse(`${column}: expected an int64 within +-(2^53-1), got ${value === null ? "null" : typeof value}`);
}

function text(value: unknown, column: string): string {
  if (typeof value !== "string") return refuse(`${column}: expected a string, got ${value === null ? "null" : typeof value}`);
  return value;
}

/** Rebuild one record from a row as the Parquet reader returns it (`row` is checked, never trusted). */
export function recordOf(row: Record<string, unknown>, plan: RowPlan, where: string): { record: TeamRecord; stratum: number } {
  try {
    return rebuild(row, plan);
  } catch (error) {
    throw new ShardError(`${where}: ${(error as Error).message}`);
  }
}

function rebuild(row: Record<string, unknown>, plan: RowPlan): { record: TeamRecord; stratum: number } {
  const labelsRaw = row.labels;
  if (!Array.isArray(labelsRaw)) return refuse(`labels: expected a list, got ${labelsRaw === null ? "null" : typeof labelsRaw}`);
  const set = plan.labelSet.labels;
  if (labelsRaw.length !== set.length) return refuse(`labels holds ${labelsRaw.length} entries, the label set has ${set.length}`);
  const labels = labelsRaw.map((entry, k): LabelEntry => {
    if (typeof entry !== "object" || entry === null) return refuse(`labels[${k}] is not a struct`);
    const fields = Object.keys(entry).sort().join(",");
    if (fields !== "label_id,value") return refuse(`labels[${k}] has fields ${fields}, expected label_id,value`);
    const { label_id, value } = entry as Record<string, unknown>;
    return { label_id: text(label_id, `labels[${k}].label_id`), value: text(value, `labels[${k}].value`) };
  });
  labels.forEach((l, k) => {
    const label = set[k] as TeamLabelSet["labels"][number];
    if (l.label_id !== label.label_id) refuse(`labels[${k}] is ${l.label_id}, the label set's is ${label.label_id}`);
    if (l.value !== label.default && !label.values.some((v) => v.value === l.value)) refuse(`label ${l.label_id}: ${l.value} is not one of its values`);
    if (row[labelColumn(l.label_id)] !== l.value) refuse(`column ${labelColumn(l.label_id)} is not the label's value ${l.value}`);
  });
  const record: TeamRecord = {
    schema_version: int(row.schema_version, "schema_version") as 1,
    data_class: text(row.data_class, "data_class") as "synthetic",
    team_hash: text(row.team_hash, "team_hash"),
    canonical_team_hash: text(row.canonical_team_hash, "canonical_team_hash"),
    paste: text(row.paste, "paste"),
    origin: {
      dataset_id: text(row.dataset_id, "dataset_id"),
      split: text(row.split, "split") as TeamRecord["origin"]["split"],
      index: int(row.origin_index, "origin_index"),
      draw_counter: int(row.origin_draw_counter, "origin_draw_counter"),
    },
    label_set_hash: text(row.label_set_hash, "label_set_hash"),
    labels,
  };
  if (!validateRecord(record)) return refuse(`not a TeamRecord: ${JSON.stringify(validateRecord.errors)}`);
  const stratum = int(row.stratum, "stratum");
  const derived = plan.stratumOf(labels);
  if (stratum !== derived) refuse(`stratum is ${stratum}, the labels give ${derived}`);
  return { record, stratum };
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

/**
 * The canonical (JCS) text of a TeamRecord, built without the generic object
 * walk: the keys are the schema's, in code-unit order; every string but
 * `paste` is pattern-checked ASCII, and `paste` is checked for a lone surrogate
 * as JCS requires. Only meaningful for a record that passed the TeamRecord
 * schema (a stray or missing field gives text that differs from the canonical
 * text of the real record). A test pins it to the bridge's `jcs`.
 */
export function recordJson(record: TeamRecord): string {
  if (LONE_SURROGATE.test(record.paste)) return refuse("jcs: lone surrogate in a paste");
  const o = record.origin;
  const labels = record.labels.map((l) => `{"label_id":${JSON.stringify(l.label_id)},"value":${JSON.stringify(l.value)}}`).join(",");
  return (
    `{"canonical_team_hash":${JSON.stringify(record.canonical_team_hash)},"data_class":${JSON.stringify(record.data_class)},"label_set_hash":${JSON.stringify(record.label_set_hash)},"labels":[${labels}],` +
    `"origin":{"dataset_id":${JSON.stringify(o.dataset_id)},"draw_counter":${o.draw_counter},"index":${o.index},"split":${JSON.stringify(o.split)}},` +
    `"paste":${JSON.stringify(record.paste)},"schema_version":${record.schema_version},"team_hash":${JSON.stringify(record.team_hash)}}`
  );
}

/** `record_hash`: SHA-256 of the record's JCS bytes, the same as the bridge's `recordHash`. */
export const shardRecordHash = (record: TeamRecord): string => createHash("sha256").update(recordJson(record), "utf8").digest("hex");
