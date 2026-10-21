// Reading one private Parquet shard back into PrivateTeamRecords (ADR-015
// sections 2 and 3). The decoder is hyparquet, as for the public shards; the
// checks are those of `reader.ts` with the private columns: the footer, the
// rebuild rules of `private-shards.ts`, the row order, the HMAC placement, the
// recomputed `logical_hash` and the cell ranges with their active rows.
//
// It does not call the engine and holds no key: whether each team is legal and
// each hold-out flag is right is the snapshot reader's work. Nothing in a
// message comes from the file (no column name, value or library text).

import { parquetMetadataAsync, parquetReadObjects, parquetSchema, type AsyncBuffer } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { recordHash, shardLogicalHash, type PrivateTeamRecord, type PrivateSnapshotManifest } from "@areneon/showdown-bridge";
import { PARQUET_SCHEMA_VERSION, refuse, type RowPlan, SCHEMA_VERSION_KEY } from "./columns.ts";
import { byPrivateOrder, privateCellRanges, privateColumnsFor, privateRecordOf } from "./private-shards.ts";

/** What the manifest says a private shard holds. */
export type PrivateShardExpectation = Pick<PrivateSnapshotManifest["shards"][number], "rows" | "logical_hash" | "cell_ranges">;

export interface PrivateShardRead {
  /** The records in file order. */
  records: PrivateTeamRecord[];
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

type Footer = Awaited<ReturnType<typeof parquetMetadataAsync>>;
type SchemaNode = ReturnType<typeof parquetSchema>["children"][number];

const PHYSICAL = { VARCHAR: "BYTE_ARRAY", NULLABLE_VARCHAR: "BYTE_ARRAY", BIGINT: "INT64", BOOLEAN: "BOOLEAN" } as const;

function checkFooter(metadata: Footer, plan: RowPlan, expectedRows: number): void {
  const version = metadata.key_value_metadata?.find((kv) => kv.key === SCHEMA_VERSION_KEY)?.value;
  if (version !== String(PARQUET_SCHEMA_VERSION)) refuse("the file's schema version is missing or not the one this reader reads");
  const top = parquetSchema(metadata).children;
  const columns = privateColumnsFor(plan.labelSet);
  const present = top.map((c) => c.element.name);
  // Position by position, so a missing, extra, repeated or reordered column refuses without naming any.
  if (present.length !== columns.length || columns.some((c, k) => present[k] !== c.name)) refuse("the file's columns are not the private layout's");
  columns.forEach((column, k) => {
    const node = top[k] as SchemaNode;
    const ok = (() => {
      if (column.type === "LABELS") {
        const element = node.children[0]?.children[0];
        const fields = element?.children.map((f) => `${f.element.name}:${f.element.type}`).join(",");
        return node.element.converted_type === "LIST" && node.children.length === 1 && element?.children.length === 2 && fields === "label_id:BYTE_ARRAY,value:BYTE_ARRAY";
      }
      if (column.type === "VARCHAR_LIST") {
        const element = node.children[0]?.children[0];
        return node.element.converted_type === "LIST" && node.children.length === 1 && element?.children.length === 0 && element.element.type === "BYTE_ARRAY";
      }
      return node.children.length === 0 && node.element.type === PHYSICAL[column.type];
    })();
    if (!ok) refuse(`column ${column.name} does not have the layout's type`);
  });
  if (Number(metadata.num_rows) !== expectedRows) refuse("the file's row count is not the manifest's");
}

/**
 * Decode, rebuild and verify one private shard. Every refusal is a
 * `ShardError` with a fixed message. `assign`, when given, maps a canonical
 * hash to its shard index; every row must give the same index, which is
 * returned.
 */
export async function readPrivateShard(
  file: ArrayBuffer | Uint8Array | AsyncBuffer,
  expected: PrivateShardExpectation,
  plan: RowPlan,
  assign?: (canonicalTeamHash: string) => number,
): Promise<PrivateShardRead> {
  const buffer = toBuffer(file);
  let metadata: Footer;
  try {
    metadata = await parquetMetadataAsync(buffer);
  } catch {
    return refuse("not a readable Parquet file");
  }
  checkFooter(metadata, plan, expected.rows);
  const records: PrivateTeamRecord[] = [];
  const recordHashes: string[] = [];
  const strata: number[] = [];
  let index: number | undefined;
  let start = 0;
  for (const group of metadata.row_groups) {
    const end = start + Number(group.num_rows);
    let rows: Record<string, unknown>[];
    try {
      rows = await parquetReadObjects({ file: buffer, metadata, compressors, rowStart: start, rowEnd: end });
    } catch {
      return refuse("not a readable Parquet file");
    }
    rows.forEach((row, k) => {
      const n = start + k;
      const where = `row ${n}`;
      const { record, stratum } = privateRecordOf(row, plan, where);
      const previous = records[n - 1];
      if (previous !== undefined) {
        const before = { stratum: strata[n - 1] as number, canonical_team_hash: previous.canonical_team_hash, team_id: previous.team_id };
        if (byPrivateOrder(before, { stratum, canonical_team_hash: record.canonical_team_hash, team_id: record.team_id }) >= 0) {
          refuse(`${where}: rows are not in strictly ascending (stratum, canonical hash, team_id) order`);
        }
      }
      if (assign !== undefined) {
        const at = assign(record.canonical_team_hash);
        if (index !== undefined && at !== index) refuse(`${where}: it belongs to another shard than the shard's other rows`);
        index = at;
      }
      records.push(record);
      recordHashes.push(recordHash(record));
      strata.push(stratum);
    });
    start = end;
  }
  if (records.length !== expected.rows) refuse("the row groups do not hold the manifest's row count");
  if (shardLogicalHash(recordHashes) !== expected.logical_hash) refuse("the shard's logical hash is not the manifest's");
  const ranges = privateCellRanges(records.map((r, k) => ({ stratum: strata[k] as number, status: r.status })));
  const same =
    ranges.length === expected.cell_ranges.length &&
    ranges.every((r, k) => {
      const given = expected.cell_ranges[k] as PrivateShardExpectation["cell_ranges"][number];
      return r.stratum === given.stratum && r.first_row === given.first_row && r.rows === given.rows && r.active_rows === given.active_rows;
    });
  if (!same) refuse("the shard's cell ranges are not the manifest's");
  return { records, recordHashes, strata, ...(index === undefined ? {} : { index }) };
}
