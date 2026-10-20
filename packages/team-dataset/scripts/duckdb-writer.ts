// Convert a dataset directory into Parquet shards with the pinned DuckDB
// command-line tool, or verify shards against a dataset (src/shards/command.ts).
//   node packages/team-dataset/scripts/duckdb-writer.ts write  --dataset DIR --labels FILE --out DIR --shard-key HEX --snapshot-id ID [--require-pinned]
//   node packages/team-dataset/scripts/duckdb-writer.ts verify --snapshot FILE --shards DIR --dataset DIR --labels FILE
// The tool is user-installed (DUCKDB_BIN or the PATH); scripts/duckdb-pin.json names the
// version and SHA-256 CI installs. It is never added to a lockfile.
import { runShardsCli } from "../src/shards/command.ts";

process.exitCode = await runShardsCli(process.argv.slice(2), { out: (line) => console.log(line), err: (line) => console.error(line) });
