// The shard writer: the DuckDB command-line tool, run as a child process
// (ADR-013 sections 6 and 7). It is a tooling dependency, never a shipped one:
// the binary is user-installed (or fetched in CI by the exact URL and SHA-256
// of `scripts/duckdb-pin.json`), it is in no package.json, lockfile, Dockerfile
// or image, and nothing in this repository imports a DuckDB library. A test
// asserts that.
//
// The rows reach the tool as a JSON Lines file this module's caller wrote,
// with explicit column types (no type inference), and leave as one Parquet
// file: zstd, one thread, a fixed row group size, the schema version in the
// file's key-value metadata. Different DuckDB versions (and settings) write
// different bytes for the same rows; the logical hashes are what stay equal.
// The writer's identity (`name`, `version`, `binary_sha256`) is read from the
// binary that ran, never from a configuration, and goes into the manifest.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { accessSync, constants, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import type { TeamSnapshotManifest } from "@areneon/showdown-bridge";
import { type Column, PARQUET_SCHEMA_VERSION, refuse, SCHEMA_VERSION_KEY } from "./columns.ts";
import { removeTree } from "./scratch.ts";

export type WriterIdentity = TeamSnapshotManifest["writer"];

/** One platform's pinned download. */
export interface PinnedAsset {
  url: string;
  /** SHA-256 of the downloaded zip. */
  zip_sha256: string;
  /** SHA-256 of the extracted `duckdb` binary. */
  binary_sha256: string;
}
export interface DuckdbPin {
  schema_version: 1;
  name: "duckdb";
  version: string;
  assets: Record<string, PinnedAsset>;
}

export const PIN_FILE = new URL("../../scripts/duckdb-pin.json", import.meta.url);

/** The pin committed with the package, read and shape-checked. */
export function readPin(file: URL | string = PIN_FILE): DuckdbPin {
  const pin = JSON.parse(readFileSync(file, "utf8")) as DuckdbPin;
  const hex = /^[0-9a-f]{64}$/;
  const ok =
    pin.schema_version === 1 &&
    pin.name === "duckdb" &&
    /^[0-9]+\.[0-9]+\.[0-9]+$/.test(pin.version) &&
    Object.values(pin.assets).every((a) => hex.test(a.zip_sha256) && hex.test(a.binary_sha256) && a.url === `https://github.com/duckdb/duckdb/releases/download/v${pin.version}/${a.url.split("/").pop()}`);
  if (!ok) return refuse("duckdb-pin.json is not a pin of the shape the writer reads");
  return pin;
}

/** The pin's key for a platform (`linux-amd64`, `linux-arm64`), if it has one. */
export function platformKey(platform: string = process.platform, arch: string = process.arch): string {
  return `${platform}-${arch === "x64" ? "amd64" : arch}`;
}

export interface CommandResult {
  status: number | null;
  stdout: string;
  stderr: string;
}
/** Run a program with text on stdin; the writer's only contact with the outside. */
export type Runner = (file: string, args: readonly string[], input: string, env: Record<string, string>, cwd?: string, timeoutMs?: number) => CommandResult;

const run: Runner = (file, args, input, env, cwd, timeoutMs) => {
  const result = spawnSync(file, [...args], {
    input,
    env,
    encoding: "utf8",
    maxBuffer: 1 << 24,
    ...(cwd === undefined ? {} : { cwd }),
    ...(timeoutMs === undefined ? {} : { timeout: timeoutMs, killSignal: "SIGKILL" as const }),
  });
  return { status: result.status, stdout: result.stdout ?? "", stderr: result.error === undefined ? (result.stderr ?? "") : result.error.message };
};

/** The binary named by `DUCKDB_BIN`, else the first `duckdb` on the PATH, else undefined. */
export function locateDuckdb(env: Record<string, string | undefined> = process.env): string | undefined {
  if (env.DUCKDB_BIN !== undefined && env.DUCKDB_BIN !== "") return env.DUCKDB_BIN;
  for (const dir of (env.PATH ?? "").split(delimiter)) {
    if (dir === "") continue;
    const candidate = join(dir, "duckdb");
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // not here
    }
  }
  return undefined;
}

/** Parquet writing options. The defaults are what the snapshot is published with. */
export interface WriteOptions {
  /** Rows per row group. */
  rowGroupSize?: number;
  compression?: "zstd" | "snappy" | "uncompressed";
  compressionLevel?: number;
  /**
   * The private path (ADR-015 section 3). The tool runs with a scratch HOME,
   * TMPDIR and working directory made inside `dir` (so its spill directory and
   * any stray file land there, never in the caller's directory or in `/tmp`), a
   * history file of `/dev/null`, and nothing else in its environment. Its
   * output is never forwarded: a failure is the fixed message below, because
   * DuckDB's stderr can quote the rows (a paste). The extension auto-install
   * and auto-load settings are turned off before any other statement and the
   * configuration is locked before the copy, so the tool cannot fetch or load
   * code. The tool is killed after `timeoutMs` (default 10 minutes). The
   * scratch directory is removed afterwards, also on failure; if it cannot be
   * removed the call fails with `CLEANUP_MESSAGE` unless it failed already
   * (the first error is never replaced). `dir` is checked by the caller.
   */
  isolated?: { dir: string; timeoutMs?: number };
}

/** What a private-path failure says; DuckDB's own text is dropped. */
/** What a private-path call says when its scratch directory could not be removed. */
export const CLEANUP_MESSAGE = "the scratch directory could not be removed";
export const DEFAULT_TOOL_TIMEOUT_MS = 600_000;
export const WITHHELD_MESSAGE = "DuckDB failed; its output is withheld because it can quote a paste";
export const DEFAULT_ROW_GROUP_SIZE = 131072;

const literal = (text: string): string => {
  if (/[\0\n\r]/.test(text)) return refuse("a path or name with a control character is not passed to DuckDB");
  return `'${text.replaceAll("'", "''")}'`;
};

export class DuckdbWriter {
  readonly identity: WriterIdentity;
  readonly #binary: string;
  readonly #runner: Runner;

  private constructor(binary: string, identity: WriterIdentity, runner: Runner) {
    this.#binary = binary;
    this.identity = identity;
    this.#runner = runner;
  }

  /** Open a binary: run `--version` and hash the file. Refuses a binary that does not answer as DuckDB. */
  static open(binary: string, runner: Runner = run): DuckdbWriter {
    const answer = runner(binary, ["--version"], "", { PATH: "" });
    const version = /^v([0-9]+\.[0-9]+\.[0-9]+)\b/.exec(answer.stdout.trim())?.[1];
    if (answer.status !== 0 || version === undefined) return refuse(`${binary} did not answer --version as DuckDB: ${answer.stderr.trim() || answer.stdout.trim()}`);
    const sha = createHash("sha256").update(readFileSync(binary)).digest("hex");
    return new DuckdbWriter(binary, { name: "duckdb", version, binary_sha256: sha }, runner);
  }

  /** Refuse unless this binary is the one `pin` names for this platform (version and SHA-256). */
  requirePinned(pin: DuckdbPin, platform: string = platformKey()): void {
    const asset = pin.assets[platform];
    if (asset === undefined) return refuse(`the pin has no binary for ${platform}`);
    if (this.identity.version !== pin.version) refuse(`DuckDB ${this.identity.version} is not the pinned ${pin.version}`);
    if (this.identity.binary_sha256 !== asset.binary_sha256) refuse(`the DuckDB binary's SHA-256 ${this.identity.binary_sha256} is not the pinned ${asset.binary_sha256}`);
  }

  /**
   * Run a SQL script against a database file (created when absent). Relative
   * paths in the script resolve against `cwd`. The tool gets a scratch HOME
   * (no `~/.duckdbrc`, no extension directory) and an empty PATH. Public data
   * only: on failure the tool's own text is forwarded, which can quote rows.
   */
  runScript(sql: string, database: string, cwd: string): void {
    const home = mkdtempSync(join(tmpdir(), "duckdb-home-"));
    try {
      const result = this.#runner(this.#binary, ["-bail", "-batch", database], sql, { PATH: "", HOME: home }, cwd);
      if (result.status !== 0) refuse(`DuckDB failed (${result.status}): ${result.stderr.trim()}`);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  }

  /** Convert a JSON Lines file of rows (keys = column names) into one Parquet file, keeping the row order. */
  writeShard(rowsFile: string, out: string, columns: readonly Column[], options: WriteOptions = {}): void {
    const types = {
      VARCHAR: "VARCHAR",
      NULLABLE_VARCHAR: "VARCHAR",
      BIGINT: "BIGINT",
      BOOLEAN: "BOOLEAN",
      VARCHAR_LIST: "VARCHAR[]",
      LABELS: "STRUCT(label_id VARCHAR, value VARCHAR)[]",
    } as const;
    const group = options.rowGroupSize ?? DEFAULT_ROW_GROUP_SIZE;
    if (!Number.isSafeInteger(group) || group < 1) return refuse("the row group size must be a positive integer");
    const compression = options.compression ?? "zstd";
    const level = options.compressionLevel;
    if (level !== undefined && (!Number.isInteger(level) || level < 1 || level > 22)) return refuse("the compression level must be between 1 and 22");
    const shape = columns.map((c) => `${literal(c.name)}: '${types[c.type]}'`).join(", ");
    const isolated = options.isolated;
    const sql = [
      ...(isolated === undefined ? [] : ["SET autoinstall_known_extensions = false;", "SET autoload_known_extensions = false;"]),
      "SET threads = 1;",
      "SET preserve_insertion_order = true;",
      ...(isolated === undefined ? [] : ["SET lock_configuration = true;"]),
      `COPY (SELECT * FROM read_json(${literal(rowsFile)}, format = 'newline_delimited', columns = {${shape}})) TO ${literal(out)}`,
      `(FORMAT parquet, COMPRESSION ${compression}${level === undefined ? "" : `, COMPRESSION_LEVEL ${level}`}, ROW_GROUP_SIZE ${group}, KV_METADATA {${SCHEMA_VERSION_KEY}: '${PARQUET_SCHEMA_VERSION}'});`,
    ].join("\n");
    // A scratch HOME keeps the tool away from any ~/.duckdbrc and extension directory.
    const home = mkdtempSync(join(isolated?.dir ?? tmpdir(), "duckdb-home-"));
    let done = false;
    try {
      const result =
        isolated === undefined
          ? this.#runner(this.#binary, ["-bail", "-batch"], sql, { PATH: "", HOME: home })
          : this.#runner(this.#binary, ["-bail", "-batch"], sql, { PATH: "", HOME: home, TMPDIR: home, DUCKDB_HISTORY: "/dev/null" }, home, isolated.timeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS);
      if (result.status !== 0) refuse(isolated === undefined ? `DuckDB failed (${result.status}): ${result.stderr.trim()}` : WITHHELD_MESSAGE);
      done = true;
    } finally {
      if (isolated === undefined) rmSync(home, { recursive: true, force: true });
      else if (!removeTree(home) && done) refuse(CLEANUP_MESSAGE);
    }
  }
}
