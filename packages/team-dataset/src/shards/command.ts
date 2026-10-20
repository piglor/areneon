// The command line of the shard tooling:
//
//   write   --dataset DIR --labels FILE --out DIR --shard-key HEX --snapshot-id ID
//           [--splits dev,validation,training] [--parent HASH] [--duckdb PATH]
//           [--require-pinned] [--shard-rows N] [--row-group-size N]
//   verify  --snapshot FILE --shards DIR --dataset DIR --labels FILE [--shard-rows N]
//
// `write` converts a dataset directory (JSON Lines records and manifest.json)
// into Parquet shards under OUT/pub/v1/shards/<sha256> and OUT/snapshot.json,
// with the DuckDB command-line tool as the writer (`--duckdb`, else the
// `DUCKDB_BIN` variable, else the PATH). Without `--require-pinned` any
// DuckDB version may write (the manifest records which); with it the binary
// must be the one `scripts/duckdb-pin.json` names for this platform.
// `verify` reads the shards back and checks them against the dataset's records
// byte for byte. Exit codes: 0 done, 1 refused, 2 usage or input error.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { readTeamLabelSet } from "@areneon/showdown-bridge";
import { ShardError } from "./columns.ts";
import { convertDataset, verifySnapshot } from "./convert.ts";
import { DuckdbWriter, locateDuckdb, readPin, type Runner } from "./duckdb.ts";

export interface ShardsIo {
  out(line: string): void;
  err(line: string): void;
  /** Environment for locating the writer; `process.env` when omitted. */
  env?: Record<string, string | undefined>;
  /** How the writer's child process runs; a real one when omitted. */
  runner?: Runner;
}

class UsageError extends Error {
  override name = "UsageError";
}

const OPTIONS = {
  dataset: { type: "string" },
  labels: { type: "string" },
  out: { type: "string" },
  "shard-key": { type: "string" },
  "snapshot-id": { type: "string" },
  splits: { type: "string" },
  parent: { type: "string" },
  duckdb: { type: "string" },
  "require-pinned": { type: "boolean" },
  "shard-rows": { type: "string" },
  "row-group-size": { type: "string" },
  snapshot: { type: "string" },
  shards: { type: "string" },
} as const;
type Values = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>["values"];

const need = (values: Values, name: keyof Values & string): string => {
  const value = values[name];
  if (typeof value !== "string") throw new UsageError(`--${name} is required`);
  return value;
};

function whole(text: string | undefined, name: string): number | undefined {
  if (text === undefined) return undefined;
  if (!/^[1-9][0-9]*$/.test(text) || !Number.isSafeInteger(Number(text))) throw new UsageError(`--${name} must be a positive whole number`);
  return Number(text);
}

const SPLITS = ["dev", "validation", "training"] as const;

function splitsOf(text: string | undefined): (typeof SPLITS)[number][] {
  const names = (text ?? "dev,validation,training").split(",");
  const known = names.filter((n): n is (typeof SPLITS)[number] => (SPLITS as readonly string[]).includes(n));
  if (known.length !== names.length || new Set(names).size !== names.length) throw new UsageError("--splits is a comma-separated list of dev, validation, training, each once");
  return known;
}

const labelSetOf = (values: Values) => readTeamLabelSet(JSON.parse(readFileSync(need(values, "labels"), "utf8")));

async function write(values: Values, io: ShardsIo): Promise<void> {
  const binary = values.duckdb ?? locateDuckdb(io.env);
  if (binary === undefined) throw new UsageError("no DuckDB command-line tool: give --duckdb, set DUCKDB_BIN, or put duckdb on the PATH");
  const writer = DuckdbWriter.open(binary, io.runner);
  if (values["require-pinned"] === true) writer.requirePinned(readPin());
  const rowGroupSize = whole(values["row-group-size"], "row-group-size");
  const parent = values.parent;
  const { snapshot, snapshotFile } = await convertDataset({
    datasetDir: need(values, "dataset"),
    outDir: need(values, "out"),
    labelSet: labelSetOf(values),
    shardKey: need(values, "shard-key"),
    snapshotId: need(values, "snapshot-id"),
    splits: splitsOf(values.splits),
    writer,
    ...(parent === undefined ? {} : { parentSnapshotHash: parent }),
    ...(rowGroupSize === undefined ? {} : { write: { rowGroupSize } }),
    ...(values["shard-rows"] === undefined ? {} : { shardRows: whole(values["shard-rows"], "shard-rows") as number }),
  });
  io.out(JSON.stringify({ snapshot_file: snapshotFile, logical_snapshot_hash: snapshot.logical_snapshot_hash, shards: snapshot.shards.length, writer: snapshot.writer }));
}

async function verify(values: Values, io: ShardsIo): Promise<void> {
  const rows = await verifySnapshot({
    snapshotFile: need(values, "snapshot"),
    shardsDir: need(values, "shards"),
    datasetDir: need(values, "dataset"),
    labelSet: labelSetOf(values),
    ...(values["shard-rows"] === undefined ? {} : { shardRows: whole(values["shard-rows"], "shard-rows") as number }),
  });
  io.out(JSON.stringify({ verified: rows }));
}

/** Run the command line; returns the exit code. */
export async function runShardsCli(argv: readonly string[], io: ShardsIo): Promise<number> {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true });
    const [command, ...rest] = positionals;
    if (rest.length > 0 || (command !== "write" && command !== "verify")) throw new UsageError("usage: <write|verify> [options]; see src/shards/command.ts");
    await (command === "write" ? write(values, io) : verify(values, io));
    return 0;
  } catch (error) {
    if (error instanceof ShardError) {
      io.err(`refused: ${error.message}`);
      return 1;
    }
    io.err(`${error instanceof UsageError ? "usage" : "error"}: ${(error as Error).message}`);
    return 2;
  }
}
