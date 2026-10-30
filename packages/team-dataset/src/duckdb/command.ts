// The `teams` command line (so far one subcommand):
//
//   duckdb --snapshot DIR|FILE --out FILE.duckdb [--labels FILE] [--dataset DIR]
//          [--shards DIR] [--duckdb PATH] [--require-pinned] [--shard-rows N]
//
// Builds the derived working database of a PUBLIC snapshot (src/duckdb/working-database.ts).
// `--snapshot` is a snapshot directory (holding `snapshot.json` and
// `pub/v1/shards/`, as `duckdb-writer.ts write` leaves it) or a `snapshot.json`
// (its shards are then looked for beside it, or in `--shards`). `--labels` is
// the label set the records were made with; by default the committed one of the
// snapshot's format. `--dataset` is the dataset directory whose manifest names
// the merged strata; a snapshot of a dataset with merged strata needs it to
// verify. The DuckDB tool is `--duckdb`, else `DUCKDB_BIN`, else the PATH.
// Exit codes: 0 done, 1 refused, 2 usage or input error.

import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadEngine, readTeamDatasetManifest, readTeamLabelSet, readTeamSnapshotManifest } from "@areneon/showdown-bridge";
import { TEAMS_CONFIG_DIR } from "../check-labels.ts";
import { ShardError } from "../shards/columns.ts";
import { DuckdbWriter, locateDuckdb, readPin, type Runner } from "../shards/duckdb.ts";
import { buildWorkingDatabase } from "./working-database.ts";

export interface TeamsIo {
  out(line: string): void;
  err(line: string): void;
  env?: Record<string, string | undefined>;
  runner?: Runner;
}

class UsageError extends Error {
  override name = "UsageError";
}

const OPTIONS = {
  snapshot: { type: "string" },
  out: { type: "string" },
  labels: { type: "string" },
  dataset: { type: "string" },
  shards: { type: "string" },
  duckdb: { type: "string" },
  "require-pinned": { type: "boolean" },
  "shard-rows": { type: "string" },
} as const;
type Values = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>["values"];

const need = (values: Values, name: keyof Values & string): string => {
  const value = values[name];
  if (typeof value !== "string") throw new UsageError(`--${name} is required`);
  return value;
};

async function duckdb(values: Values, io: TeamsIo): Promise<void> {
  const binary = values.duckdb ?? locateDuckdb(io.env);
  if (binary === undefined) throw new UsageError("no DuckDB command-line tool: give --duckdb, set DUCKDB_BIN, or put duckdb on the PATH");
  const out = need(values, "out");
  if (!out.endsWith(".duckdb")) throw new UsageError("--out must be a file name ending in .duckdb");
  const given = need(values, "snapshot");
  const isDirectory = statSync(given).isDirectory();
  const snapshotFile = isDirectory ? join(given, "snapshot.json") : given;
  const shardsDir = values.shards ?? (isDirectory ? given : join(snapshotFile, ".."));
  const snapshot = readTeamSnapshotManifest(JSON.parse(readFileSync(snapshotFile, "utf8")));
  const labelSet = readTeamLabelSet(JSON.parse(readFileSync(values.labels ?? join(TEAMS_CONFIG_DIR, snapshot.manifest.format_id, "labels.json"), "utf8")));
  const datasetManifest = values.dataset === undefined ? undefined : readTeamDatasetManifest(JSON.parse(readFileSync(join(values.dataset, "manifest.json"), "utf8")), { labelSet });
  const shardRows = values["shard-rows"];
  if (shardRows !== undefined && !/^[1-9][0-9]*$/.test(shardRows)) throw new UsageError("--shard-rows must be a positive whole number");
  const writer = DuckdbWriter.open(binary, io.runner);
  if (values["require-pinned"] === true) writer.requirePinned(readPin());
  const result = await buildWorkingDatabase({
    snapshotFile,
    shardsDir,
    labelSet,
    engine: loadEngine(),
    writer,
    outFile: out,
    ...(datasetManifest === undefined ? {} : { datasetManifest }),
    ...(shardRows === undefined ? {} : { shardRows: Number(shardRows) }),
  });
  io.out(JSON.stringify({ database: result.database, teams: result.teams, sets: result.sets, moves: result.moves }));
}

/** Errors that mean the inputs are wrong: exit 2. */
const INPUT_ERRORS = new Set(["UsageError", "SyntaxError", "TeamRecordError", "EngineError"]);
const INPUT_CODES = new Set(["ENOENT", "ENOTDIR"]);

/** Run the command line; returns the exit code. */
export async function runTeamsCli(argv: readonly string[], io: TeamsIo): Promise<number> {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true });
    const [command, ...rest] = positionals;
    if (rest.length > 0 || command !== "duckdb") throw new UsageError("usage: duckdb --snapshot DIR|FILE --out FILE.duckdb [options]; see src/duckdb/command.ts");
    await duckdb(values, io);
    return 0;
  } catch (error) {
    if (error instanceof ShardError) {
      io.err(`refused: ${error.message}`);
      return 1;
    }
    const code = String((error as { code?: unknown }).code);
    const input = INPUT_ERRORS.has((error as Error).name) || INPUT_CODES.has(code) || code.startsWith("ERR_PARSE_ARGS");
    if (!input) throw error;
    io.err(`${error instanceof UsageError ? "usage" : "error"}: ${(error as Error).message}`);
    return 2;
  }
}
