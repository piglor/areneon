// What the shard tests share: the committed fixture dataset read without the
// engine, a second Parquet writer (hyparquet-writer, pure JavaScript) that
// makes shards of any shape, including malformed ones, and a stand-in for the
// DuckDB command-line tool that writes Parquet through that writer so the
// conversion paths run where no DuckDB binary is installed.

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ByteWriter, parquetWrite } from "hyparquet-writer";
import {
  readTeamDatasetManifest,
  readTeamLabelSet,
  recordHash,
  shardLogicalHash,
  type TeamDatasetManifest,
  type TeamLabelSet,
  type TeamRecord,
} from "@areneon/showdown-bridge";
import { type Column, columnsFor, PARQUET_SCHEMA_VERSION, type Row, rowOf, type RowPlan, SCHEMA_VERSION_KEY } from "../../src/shards/columns.ts";
import type { Runner } from "../../src/shards/duckdb.ts";
import { byStratumThenHash, cellRanges } from "../../src/shards/layout.ts";
import type { ShardExpectation } from "../../src/shards/reader.ts";
import { datasetPlan } from "../../src/shards/snapshot.ts";
import { CONFIG_DIR, DATASET_DIR } from "./paths.ts";


/** The element at `k` (negative counts from the end), or a failure: tests index without `!`. */
export function at<T>(list: readonly T[] | RegExpExecArray | null | undefined, k: number): T {
  const value = list?.at(k) as T | undefined;
  if (value === undefined) throw new Error(`no element at ${k}`);
  return value;
}

export const labelSet: TeamLabelSet = readTeamLabelSet(JSON.parse(readFileSync(join(CONFIG_DIR, "labels.json"), "utf8")));
export const manifest: TeamDatasetManifest = readTeamDatasetManifest(JSON.parse(readFileSync(join(DATASET_DIR, "manifest.json"), "utf8")), { labelSet });
export const plan: RowPlan = datasetPlan(manifest, labelSet);
export const SHARD_KEY = "ab".repeat(32);

export const fixtureRecords = (split: "dev" | "validation"): TeamRecord[] =>
  readFileSync(join(DATASET_DIR, `teams-${split}.jsonl`), "utf8")
    .split("\n")
    .filter((l) => l !== "")
    .map((l) => JSON.parse(l) as TeamRecord);

export interface Staged {
  record: TeamRecord;
  stratum: number;
  canonical_team_hash: string;
  row: Row;
}

/** Records with their strata and rows, in shard order (stratum, then canonical hash). */
export function shardOrder(records: readonly TeamRecord[]): Staged[] {
  return records
    .map((record) => {
      const stratum = plan.stratumOf(record.labels);
      return { record, stratum, canonical_team_hash: record.canonical_team_hash, row: rowOf(record, stratum) };
    })
    .sort(byStratumThenHash);
}

/** What a manifest would say about a shard of these rows. */
export function expectationOf(staged: readonly Staged[], split: "dev" | "validation" | "training"): ShardExpectation {
  return {
    rows: staged.length,
    split,
    logical_hash: shardLogicalHash(staged.map((s) => recordHash(s.record))),
    cell_ranges: cellRanges(staged.map((s) => s.stratum)) as ShardExpectation["cell_ranges"],
  };
}

export interface ParquetOptions {
  /** Leave out the schema-version key-value metadata. */
  noVersion?: boolean;
  version?: string;
  rowGroupSize?: number;
  codec?: "SNAPPY" | "UNCOMPRESSED";
  /** Extra columns (name, values) added after the layout's. */
  extra?: { name: string; data: unknown[] }[];
  /** Write `origin_index` as DOUBLE. */
  indexAsDouble?: boolean;
  /** Give each labels struct a third field `extra`. */
  labelExtra?: boolean;
  /** Columns to leave out. */
  without?: string[];
}

/** A Parquet file of rows, written by hyparquet-writer; nulls and unusual values pass through. */
export function parquetOf(rows: readonly Record<string, unknown>[], columns: readonly Column[] = columnsFor(labelSet), options: ParquetOptions = {}): Uint8Array {
  const kept = columns.filter((c) => !(options.without ?? []).includes(c.name));
  const schema: Record<string, unknown>[] = [{ name: "root", num_children: kept.length + (options.extra?.length ?? 0) }];
  const columnData: { name: string; data: unknown[] }[] = [];
  for (const column of kept) {
    columnData.push({ name: column.name, data: rows.map((r) => (column.type === "BIGINT" && typeof r[column.name] === "number" && !(options.indexAsDouble === true && column.name === "origin_index") ? BigInt(r[column.name] as number) : r[column.name])) });
    if (column.type === "LABELS") {
      schema.push({ name: column.name, converted_type: "LIST", repetition_type: "OPTIONAL", num_children: 1 }, { name: "list", repetition_type: "REPEATED", num_children: 1 });
      schema.push({ name: "element", repetition_type: "OPTIONAL", num_children: options.labelExtra === true ? 3 : 2 });
      schema.push({ name: "label_id", type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" }, { name: "value", type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" });
      if (options.labelExtra === true) schema.push({ name: "extra", type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" });
    } else if (column.type === "VARCHAR_LIST") {
      schema.push({ name: column.name, converted_type: "LIST", repetition_type: "OPTIONAL", num_children: 1 }, { name: "list", repetition_type: "REPEATED", num_children: 1 });
      schema.push({ name: "element", type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" });
    } else if (column.type === "BOOLEAN") {
      schema.push({ name: column.name, type: "BOOLEAN", repetition_type: "OPTIONAL" });
    } else if (column.type === "BIGINT") {
      schema.push({ name: column.name, type: options.indexAsDouble === true && column.name === "origin_index" ? "DOUBLE" : "INT64", repetition_type: "OPTIONAL" });
    } else {
      schema.push({ name: column.name, type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" });
    }
  }
  for (const extra of options.extra ?? []) {
    columnData.push(extra);
    schema.push({ name: extra.name, type: "BYTE_ARRAY", converted_type: "UTF8", repetition_type: "OPTIONAL" });
  }
  const writer = new ByteWriter();
  parquetWrite({
    writer,
    columnData: columnData as never,
    schema: schema as never,
    codec: options.codec ?? "SNAPPY",
    rowGroupSize: options.rowGroupSize ?? 100,
    kvMetadata: options.noVersion === true ? [] : [{ key: SCHEMA_VERSION_KEY, value: options.version ?? String(PARQUET_SCHEMA_VERSION) }],
  });
  return new Uint8Array(writer.getBuffer());
}

/**
 * A stand-in for the DuckDB tool: answers `--version`, and on the SQL the
 * writer sends it reads the rows file named in it and writes the Parquet file
 * named in it through hyparquet-writer. `calls` records every invocation.
 */
export function standInRunner(version = "v1.5.6 (Variegata) 0123456789"): Runner & { calls: { args: readonly string[]; input: string; env: Record<string, string>; cwd: string | undefined }[] } {
  const calls: { args: readonly string[]; input: string; env: Record<string, string>; cwd: string | undefined }[] = [];
  const runner = ((_file, args, input, env, cwd) => {
    calls.push({ args, input, env, cwd });
    if (args[0] === "--version") return { status: 0, stdout: `${version}\n`, stderr: "" };
    const rowsFile = /read_json\('((?:[^']|'')*)'/.exec(input)?.[1]?.replaceAll("''", "'");
    const out = /TO '((?:[^']|'')*)'/.exec(input)?.[1]?.replaceAll("''", "'");
    if (rowsFile === undefined || out === undefined) return { status: 1, stdout: "", stderr: "stand-in: no read_json or TO in the SQL" };
    const rows = readFileSync(rowsFile, "utf8")
      .split("\n")
      .filter((l) => l !== "")
      .map((l) => JSON.parse(l) as Record<string, unknown>);
    const names = [...input.matchAll(/'((?:[^']|'')*)': '(VARCHAR|BIGINT|BOOLEAN|VARCHAR\[\]|STRUCT\(label_id VARCHAR, value VARCHAR\)\[\])'/g)];
    const typeOf = { VARCHAR: "VARCHAR", BIGINT: "BIGINT", BOOLEAN: "BOOLEAN", "VARCHAR[]": "VARCHAR_LIST" } as const;
    const columns: Column[] = names.map((m) => ({ name: (m[1] as string).replaceAll("''", "'"), type: typeOf[m[2] as keyof typeof typeOf] ?? "LABELS" }));
    const group = Number(/ROW_GROUP_SIZE (\d+)/.exec(input)?.[1] ?? 100);
    writeFileSync(out, parquetOf(rows, columns, { rowGroupSize: group, codec: /COMPRESSION uncompressed/.test(input) ? "UNCOMPRESSED" : "SNAPPY" }));
    return { status: 0, stdout: "", stderr: "" };
  }) as Runner & { calls: typeof calls };
  runner.calls = calls;
  return runner;
}
