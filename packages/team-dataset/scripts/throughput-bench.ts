// Decisions per host-core-hour of an arm of the throughput comparison (the Showdown path, a native engine, a GPU engine): the harness
// measures wall clock, the CPU of the arm's whole process tree and the machine's conditions, and writes one record per run.
//   node packages/team-dataset/scripts/throughput-bench.ts run --arm showdown --out DIR --config C --run-dir R --preview P --battle B --engine-data E --worker-python W --ids-python PY
//   node packages/team-dataset/scripts/throughput-bench.ts run --arm command --out DIR --arm-id ID --arm-label TEXT --arm-report FILE -- COMMAND ARGS...
//   node packages/team-dataset/scripts/throughput-bench.ts guard | verify RECORD... | summarise RECORD_OR_DIR...
// Definitions, protocol and the record: docs/architecture/throughput.md
import { runThroughputCli } from "../src/throughput/cli.ts";

const root = new URL("../../..", import.meta.url).pathname;
process.exitCode = await runThroughputCli(process.argv.slice(2), { out: (line) => console.log(line), err: (line) => console.error(line), repoRoot: root });
