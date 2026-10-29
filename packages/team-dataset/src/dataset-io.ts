// Writing and reading a generated dataset directory (JSON Lines and JSON, no
// Parquet: shards are a later slice). Every file is the canonical (JCS) form
// of its value, so the same inputs give the same bytes:
//   records.<split>.jsonl   one TeamRecord per line
//   manifest.json           the TeamDatasetManifest
//   coverage.json           the coverage report
//   pilot.json              the recorded pilot
//   rejections.jsonl        the rejection log (written during the run)
//   reference-hashes.txt    the sorted canonical hashes the dataset deduplicated against, one per line
//                           (only when `generate` was given `--reference`; the readers check them
//                           against the manifest's `reference_pool_hash`)

import { closeSync, fstatSync, openSync, readSync, writeFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import { jcs } from "@areneon/showdown-bridge";
import type { GenerateResult } from "./generate.ts";
import { SPLITS } from "./plan.ts";

/** Indented JSON with the keys in JCS order. */
export const pretty = (value: unknown): string => `${JSON.stringify(JSON.parse(jcs(value as never)), null, 2)}\n`;

export const recordsFile = (split: string): string => `records.${split}.jsonl`;

/** Bytes buffered before a write: the writer and the readers hold about this much of a file, never the file. */
export const CHUNK_BYTES = 1 << 20;

/** Write `lines` (each given without its newline) to `path` in chunks: the same bytes as joining them, without the joined string. */
export function writeLines(path: string, lines: Iterable<string>, chunkBytes = CHUNK_BYTES): void {
  const fd = openSync(path, "w");
  try {
    let pending: string[] = [];
    let size = 0;
    const flush = (): void => {
      if (pending.length > 0) writeSync(fd, pending.join(""));
      pending = [];
      size = 0;
    };
    for (const line of lines) {
      pending.push(line, "\n");
      size += line.length + 1;
      if (size >= chunkBytes) flush();
    }
    flush();
  } finally {
    closeSync(fd);
  }
}

/** Whether the file is empty or ends with a newline (one byte is read). */
export function endsWithNewline(path: string): boolean {
  const fd = openSync(path, "r");
  try {
    const { size } = fstatSync(fd);
    if (size === 0) return true;
    const last = Buffer.alloc(1);
    readSync(fd, last, 0, 1, size - 1);
    return last[0] === 0x0a;
  } finally {
    closeSync(fd);
  }
}

/**
 * Call `visit(line, n)` for every line of the file (n counts from 0; the
 * newline is not part of the line; a last line without one is still visited;
 * an empty file has no lines). The file is read in chunks and split on the
 * byte 0x0A, which never occurs inside a multi-byte UTF-8 sequence, so only
 * one chunk and one line are held at a time.
 */
export function eachLine(path: string, visit: (line: string, n: number) => void, chunkBytes = CHUNK_BYTES): void {
  const fd = openSync(path, "r");
  try {
    const chunk = Buffer.alloc(chunkBytes);
    let carry: Buffer = Buffer.alloc(0);
    let n = 0;
    for (;;) {
      const got = readSync(fd, chunk, 0, chunkBytes, null);
      if (got === 0) break;
      let from = 0;
      for (let at = chunk.indexOf(0x0a, 0); at !== -1 && at < got; at = chunk.indexOf(0x0a, from)) {
        const head = chunk.subarray(from, at);
        visit((carry.length === 0 ? head : Buffer.concat([carry, head])).toString("utf8"), n++);
        carry = Buffer.alloc(0);
        from = at + 1;
      }
      if (from < got) carry = Buffer.concat([carry, chunk.subarray(from, got)]);
    }
    if (carry.length > 0) visit(carry.toString("utf8"), n);
  } finally {
    closeSync(fd);
  }
}

/** Write the dataset into `dir` (which holds `rejections.jsonl` already). */
export function writeDataset(dir: string, result: GenerateResult): string[] {
  const files: string[] = [];
  for (const split of SPLITS) {
    const records = result.records[split];
    if (records === undefined) continue;
    writeLines(join(dir, recordsFile(split)), (function* () { for (const r of records) yield jcs(r as never); })());
    files.push(recordsFile(split));
  }
  writeFileSync(join(dir, "manifest.json"), pretty(result.manifest));
  writeFileSync(join(dir, "coverage.json"), pretty(result.coverage));
  writeFileSync(join(dir, "pilot.json"), pretty(result.pilot));
  return [...files, "manifest.json", "coverage.json", "pilot.json", "rejections.jsonl"];
}

export const REFERENCE_HASHES_FILE = "reference-hashes.txt";

/** Write the reference pool's sorted hashes beside the dataset (nothing for an empty pool: such a dataset has no file). */
export function writeReferenceHashes(dir: string, sorted: readonly string[]): string[] {
  if (sorted.length === 0) return [];
  writeLines(join(dir, REFERENCE_HASHES_FILE), sorted);
  return [REFERENCE_HASHES_FILE];
}

/** The `canonical_team_hash` of every record of JSON Lines files: the hashes a later dataset dedupes against. */
export function recordHashes(paths: readonly string[]): string[] {
  const hashes: string[] = [];
  for (const path of paths) {
    eachLine(path, (line, n) => {
      if (line === "") return;
      const hash = (JSON.parse(line) as { canonical_team_hash?: unknown }).canonical_team_hash;
      if (typeof hash !== "string") throw new Error(`${path} line ${n + 1} has no canonical_team_hash`);
      hashes.push(hash);
    });
  }
  return hashes;
}

/** The `paste` of every record of JSON Lines files (the teams an evaluation-only dataset keeps its distance from). */
export function recordPastes(paths: readonly string[]): string[] {
  const pastes: string[] = [];
  for (const path of paths) {
    eachLine(path, (line, n) => {
      if (line === "") return;
      const paste = (JSON.parse(line) as { paste?: unknown }).paste;
      if (typeof paste !== "string") throw new Error(`${path} line ${n + 1} has no paste`);
      pastes.push(paste);
    });
  }
  return pastes;
}
