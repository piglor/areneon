// Reading one Parquet shard back into TeamRecords (ADR-013 section 4, "Rebuilding
// a record from a Parquet row"). The decoder is hyparquet (MIT) with
// hyparquet-compressors (MIT), pure JavaScript, no native code.
//
// Nothing in the file is trusted. The reader checks the footer (the column
// set, the schema version in the key-value metadata, the row count), rebuilds
// every record under the rules in `columns.ts`, computes each `record_hash`,
// and then checks what the manifest says about the shard: the rows are in
// shard order (stratum, then canonical hash, strictly ascending), they belong
// to the shard's split, the `logical_hash` and the `cell_ranges` are the
// recomputed ones, and (with an assigner) every row belongs in one shard by
// the HMAC rule. It does not call the engine: whether each team is legal is
// the dataset reader's and the generator's work; a caller that wants the
// paste's round trip passes the records to `readTeamRecord`.

import { parquetMetadataAsync, parquetReadObjects, parquetSchema, type AsyncBuffer } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { shardLogicalHash, type TeamRecord, type TeamSnapshotManifest } from "@areneon/showdown-bridge";
import { columnsFor, PARQUET_SCHEMA_VERSION, recordOf, refuse, type RowPlan, SCHEMA_VERSION_KEY, shardRecordHash } from "./columns.ts";
import { byStratumThenHash, cellRanges } from "./layout.ts";

export type ShardEntry = TeamSnapshotManifest["shards"][number];
/** What the manifest says a shard holds. */
export type ShardExpectation = Pick<ShardEntry, "rows" | "split" | "logical_hash" | "cell_ranges">;

export interface ShardRead {
  /** The records in file order. */
  records: TeamRecord[];
  /** `record_hash` of each record, in file order. */
  recordHashes: string[];
  /** The stratum of each record, in file order. */
  strata: number[];
  /** The one shard index every row maps to, when an assigner was given. */
  index?: number;
}

const toBuffer = (file: ArrayBuffer | Uint8Array | AsyncBuffer): AsyncBuffer => {
  if (file instanceof Uint8Array) return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer;
  return file;
};

/** The top-level column names of the file, from the footer. */
function checkFooter(metadata: Awaited<ReturnType<typeof parquetMetadataAsync>>, plan: RowPlan, expectedRows: number): void {
  const version = metadata.key_value_metadata?.find((kv) => kv.key === SCHEMA_VERSION_KEY)?.value;
  if (version !== String(PARQUET_SCHEMA_VERSION)) refuse(`the file's ${SCHEMA_VERSION_KEY} is ${version === undefined ? "missing" : version}, this reader reads ${PARQUET_SCHEMA_VERSION}`);
  const top = parquetSchema(metadata).children;
  const present = top.map((c) => c.element.name);
  const columns = columnsFor(plan.labelSet);
  const wanted = columns.map((c) => c.name);
  const missing = wanted.filter((name) => !present.includes(name));
  const extra = present.filter((name) => !wanted.includes(name));
  if (missing.length > 0) refuse(`the file lacks column ${missing[0]}`);
  if (extra.length > 0 || present.length !== wanted.length) refuse(`the file has an unexpected column ${extra[0] ?? "(repeated)"}`);
  for (const column of columns) {
    const node = top[present.indexOf(column.name)] as (typeof top)[number];
    if (column.type === "LABELS") {
      const element = node.children[0]?.children[0];
      const fields = element?.children.map((f) => `${f.element.name}:${f.element.type}`).join(",");
      if (node.element.converted_type !== "LIST" || node.children.length !== 1 || element?.children.length !== 2 || fields !== "label_id:BYTE_ARRAY,value:BYTE_ARRAY") {
        refuse(`column ${column.name} is not a list of {label_id, value} byte-array structs`);
      }
    } else if (node.children.length > 0 || node.element.type !== (column.type === "BIGINT" ? "INT64" : "BYTE_ARRAY")) {
      refuse(`column ${column.name} has physical type ${node.element.type ?? "group"}, the layout says ${column.type === "BIGINT" ? "INT64" : "BYTE_ARRAY"}`);
    }
  }
  if (Number(metadata.num_rows) !== expectedRows) refuse(`the file holds ${metadata.num_rows} rows, the manifest says ${expectedRows}`);
}

/**
 * Decode, rebuild and verify one shard. Every refusal is a `ShardError`.
 * `assign`, when given, maps a canonical hash to its shard index; every row
 * must give the same index, which is returned.
 */
export async function readShard(
  file: ArrayBuffer | Uint8Array | AsyncBuffer,
  expected: ShardExpectation,
  plan: RowPlan,
  assign?: (canonicalTeamHash: string) => number,
): Promise<ShardRead> {
  const buffer = toBuffer(file);
  let metadata: Awaited<ReturnType<typeof parquetMetadataAsync>>;
  try {
    metadata = await parquetMetadataAsync(buffer);
  } catch (error) {
    return refuse(`not a readable Parquet file: ${(error as Error).message}`);
  }
  checkFooter(metadata, plan, expected.rows);
  const records: TeamRecord[] = [];
  const recordHashes: string[] = [];
  const strata: number[] = [];
  let index: number | undefined;
  let start = 0;
  // One row group at a time, so the decoded rows of the whole shard are never alive together.
  for (const group of metadata.row_groups) {
    const end = start + Number(group.num_rows);
    let rows: Record<string, unknown>[];
    try {
      rows = await parquetReadObjects({ file: buffer, metadata, compressors, rowStart: start, rowEnd: end });
    } catch (error) {
      return refuse(`not a readable Parquet file: ${(error as Error).message}`);
    }
    rows.forEach((row, k) => {
      const n = start + k;
      const where = `row ${n}`;
      const { record, stratum } = recordOf(row, plan, where);
      if (record.origin.split !== expected.split) refuse(`${where}: its split is ${record.origin.split}, the shard's is ${expected.split}`);
      const previous = records[n - 1];
      if (previous !== undefined) {
        const key = { stratum, canonical_team_hash: record.canonical_team_hash };
        if (byStratumThenHash({ stratum: strata[n - 1] as number, canonical_team_hash: previous.canonical_team_hash }, key) >= 0) {
          refuse(`${where}: rows are not in strictly ascending (stratum, canonical hash) order`);
        }
      }
      if (assign !== undefined) {
        const at = assign(record.canonical_team_hash);
        if (index !== undefined && at !== index) refuse(`${where}: it belongs to shard ${at}, the shard's other rows to ${index}`);
        index = at;
      }
      records.push(record);
      recordHashes.push(shardRecordHash(record));
      strata.push(stratum);
    });
    start = end;
  }
  if (records.length !== expected.rows) refuse(`the row groups hold ${records.length} rows, the manifest says ${expected.rows}`);
  if (shardLogicalHash(recordHashes) !== expected.logical_hash) refuse("the shard's logical hash is not the manifest's");
  const ranges = cellRanges(strata);
  const same = ranges.length === expected.cell_ranges.length && ranges.every((r, k) => {
    const given = expected.cell_ranges[k] as ShardExpectation["cell_ranges"][number];
    return r.stratum === given.stratum && r.first_row === given.first_row && r.rows === given.rows;
  });
  if (!same) refuse("the shard's cell ranges are not the manifest's");
  return { records, recordHashes, strata, ...(index === undefined ? {} : { index }) };
}
