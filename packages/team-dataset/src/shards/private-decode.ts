// The Parquet decode of a private shard, run in a worker thread (ADR-015
// section 3, "No leaks"). hyparquet is a pure-JavaScript decoder written for
// trusted files: on a damaged file it can loop forever (a page header whose
// size moves the reader backwards), allocate without bound, or leave a
// rejected promise nobody awaits, which ends the process with library text.
// None of that can be caught on the main thread, so the file is decoded in a
// worker with a wall-clock limit and a memory limit, and the worker is
// terminated afterwards. The main thread sees either the rows or one of the
// fixed refusals below; nothing the library said ever crosses.
//
// `decodePrivateParquet` is the decode itself (footer checks against the
// private layout, then the columns one at a time); `decodeIsolated` runs it
// in the worker. The worker has no file of its own: it is a `data:` module
// that imports this file, so there is no uncovered worker script, and it gets
// an empty environment.

import { Worker } from "node:worker_threads";
import { parquetMetadataAsync, parquetReadObjects, parquetSchema } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { type Column, PARQUET_SCHEMA_VERSION, refuse, SCHEMA_VERSION_KEY, ShardError } from "./columns.ts";

export interface DecodeSpec {
  /** The private layout's columns, in file order. */
  readonly columns: readonly Column[];
  /** The rows the manifest says the shard holds. */
  readonly rows: number;
}

/** What the worker answers: the rows, or the fixed message of a refusal. */
export type DecodeAnswer = { ok: true; rows: Record<string, unknown>[] } | { ok: false; message: string };

export const UNREADABLE = "not a readable Parquet file";
export const DEFAULT_DECODE_TIMEOUT_MS = 300_000;
/** Heap limit of the worker; a shard is at most 2^18 rows. */
const WORKER_HEAP_MB = 4096;

type Footer = Awaited<ReturnType<typeof parquetMetadataAsync>>;
type SchemaNode = ReturnType<typeof parquetSchema>["children"][number];

const PHYSICAL = { VARCHAR: "BYTE_ARRAY", NULLABLE_VARCHAR: "BYTE_ARRAY", BIGINT: "INT64", BOOLEAN: "BOOLEAN" } as const;

function checkFooter(metadata: Footer, columns: readonly Column[], expectedRows: number): void {
  const version = metadata.key_value_metadata?.find((kv) => kv.key === SCHEMA_VERSION_KEY)?.value;
  if (version !== String(PARQUET_SCHEMA_VERSION)) refuse("the file's schema version is missing or not the one this reader reads");
  const top = parquetSchema(metadata).children;
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
  // The row groups must add up to the manifest's rows before any is read: a damaged footer can claim any size.
  const groupRows = metadata.row_groups.map((g) => Number(g.num_rows));
  if (groupRows.some((n) => !Number.isSafeInteger(n) || n < 0) || groupRows.reduce((a, b) => a + b, 0) !== expectedRows) refuse("the row groups do not hold the manifest's row count");
}

/**
 * Check the footer against the layout and decode every row group, one top-level
 * column at a time. Throws `ShardError` with a fixed message for a layout
 * refusal and for anything the library throws. (Do not call this on the main
 * thread with a file you do not trust; see the file comment.)
 */
export async function decodePrivateParquet(bytes: Uint8Array, spec: DecodeSpec): Promise<Record<string, unknown>[]> {
  const file = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  let metadata: Footer;
  try {
    metadata = await parquetMetadataAsync(file);
  } catch {
    return refuse(UNREADABLE);
  }
  checkFooter(metadata, spec.columns, spec.rows);
  const rows: Record<string, unknown>[] = [];
  let start = 0;
  for (const group of metadata.row_groups) {
    const end = start + Number(group.num_rows);
    const part: Record<string, unknown>[] = Array.from({ length: end - start }, () => ({}));
    for (const column of spec.columns) {
      let cells: Record<string, unknown>[];
      try {
        cells = await parquetReadObjects({ file, metadata, compressors, rowStart: start, rowEnd: end, columns: [column.name] });
      } catch {
        return refuse(UNREADABLE);
      }
      cells.forEach((cell, k) => {
        (part[k] as Record<string, unknown>)[column.name] = cell[column.name];
      });
    }
    for (const row of part) rows.push(row);
    start = end;
  }
  return rows;
}

/** The answer of the decode for a worker: a refusal is its fixed message, any other throw is `UNREADABLE`. */
export async function answerDecode(bytes: Uint8Array, spec: DecodeSpec): Promise<DecodeAnswer> {
  try {
    return { ok: true, rows: await decodePrivateParquet(bytes, spec) };
  } catch (error) {
    return { ok: false, message: error instanceof ShardError ? error.message : UNREADABLE };
  }
}

const WORKER_SOURCE = (module: string): URL =>
  new URL(
    "data:text/javascript," +
      encodeURIComponent(
        `import { parentPort, workerData } from "node:worker_threads";` +
          `const { answerDecode } = await import(${JSON.stringify(module)});` +
          `parentPort.postMessage(await answerDecode(workerData.bytes, workerData.spec));`,
      ),
  );

/**
 * Decode in a worker thread with an empty environment, a heap limit and a
 * wall-clock limit. A refusal is a `ShardError` with a fixed message; a worker
 * that fails, runs out of memory or exceeds the limit is `UNREADABLE` (a worker
 * that ends silently is caught by the limit).
 */
export function decodeIsolated(bytes: Uint8Array, spec: DecodeSpec, timeoutMs: number = DEFAULT_DECODE_TIMEOUT_MS): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const copy = bytes.slice();
    const worker = new Worker(WORKER_SOURCE(new URL("./private-decode.ts", import.meta.url).href), {
      workerData: { bytes: copy, spec },
      transferList: [copy.buffer as ArrayBuffer],
      env: {},
      resourceLimits: { maxOldGenerationSizeMb: WORKER_HEAP_MB },
    });
    let settled = false;
    const finish = (done: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      void worker.terminate().catch(() => undefined);
      done();
    };
    const timer = setTimeout(() => finish(() => reject(new ShardError(UNREADABLE))), timeoutMs);
    worker.on("message", (answer: DecodeAnswer) => finish(() => (answer.ok ? resolve(answer.rows) : reject(new ShardError(answer.message)))));
    worker.on("error", () => finish(() => reject(new ShardError(UNREADABLE))));
  });
}
