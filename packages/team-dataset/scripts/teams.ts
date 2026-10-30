// The team tooling's command line.
//   node packages/team-dataset/scripts/teams.ts duckdb --snapshot DIR|FILE --out teams.duckdb [--labels FILE] [--dataset DIR]
// Builds the derived DuckDB working database of a public snapshot; the DuckDB
// tool is user-installed (DUCKDB_BIN or the PATH). See packages/team-dataset/README.md.
import { runTeamsCli } from "../src/duckdb/index.ts";

process.exitCode = await runTeamsCli(process.argv.slice(2), { out: (line) => console.log(line), err: (line) => console.error(line) });
