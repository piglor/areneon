// The full fidelity report: `node packages/showdown-bridge/test/fidelity/report-cli.ts [--subset N] [--no-candidates] [--json FILE] > report.md`.
// Test tooling, not shipped: it is the only entry point besides the tests that reaches the fork oracle.
import { writeFileSync } from "node:fs";
import { loadEngine } from "../../src/engine.ts";
import { GOLDEN_FORMAT } from "../../src/golden.ts";
import { loadWorldWriters } from "../../src/world-writers.ts";
import { buildReport, renderReport } from "./report.ts";
import { CORPORA, reportInputs, runFidelity } from "./run.ts";

const args = process.argv.slice(2);
const at = (flag: string): string | undefined => (args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined);
const engine = loadEngine();
const subset = at("--subset");
const data = await runFidelity({ corpora: CORPORA, battles: subset === undefined ? undefined : Number(subset), candidates: !args.includes("--no-candidates"), k: Number(at("--k") ?? 3), m: Number(at("--m") ?? 2) }, engine);
const writers = loadWorldWriters(engine, GOLDEN_FORMAT);
const report = buildReport(data, writers, reportInputs(engine, writers));
const json = at("--json");
if (json !== undefined) writeFileSync(json, JSON.stringify(report, null, 2) + "\n");
process.stdout.write(renderReport(report));
