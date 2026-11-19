// The committed self-differential report of the differential harness (ADR-030 groundwork G1, #1461): the pinned probe against
// itself over every query the feature builders issue at every request of the committed random, search-fidelity and unavailable
// corpora. Run after a pin bump, a builder or probe change, a corpus change or a harness change (a test fails on stale inputs):
//
//   node packages/team-dataset/scripts/differential-report.ts                          # every corpus in this process
//   node packages/team-dataset/scripts/differential-report.ts --part search-fidelity --parts DIR   # one corpus's tally into DIR
//   node packages/team-dataset/scripts/differential-report.ts --merge DIR                          # the report from the parts
//
// The parts form lets the corpora run in separate processes (about half an hour of probe work in all on one thread).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type CorpusTally, differentialReport, type EntityTally, renderReport } from "@areneon/differential";
import { loadEngine } from "@areneon/showdown-bridge";
import { COMMITTED_CORPORA, differentialCorpus, harnessConfigs, reportInputs, secondProbe, SELF_REPORT_DIR, SELF_REPORT_TITLE } from "../src/differential-mechanics.ts";

const engine = loadEngine();
const configs = harnessConfigs(engine);

function tallyOf(id: string): CorpusTally {
  const corpus = COMMITTED_CORPORA.find((c) => c.id === id);
  if (corpus === undefined) throw new Error(`unknown corpus ${id}`);
  const started = Date.now();
  const run = differentialCorpus(engine, configs, secondProbe(loadEngine(), configs), corpus);
  console.error(`${id}: ${run.builds} builds, ${run.tally.items} queries, ${run.tally.differences.length} differences, refused ${JSON.stringify(run.refused)}, ${Math.round((Date.now() - started) / 1000)} s`);
  return run.tally;
}

const toJson = (t: CorpusTally) => ({ spec: t.spec, items: t.items, entities: [...t.entities], differences: t.differences, count_mismatches: t.count_mismatches });
const fromJson = (value: ReturnType<typeof toJson>): CorpusTally => ({ ...value, entities: new Map(value.entities as [string, EntityTally][]), pairs: new Map(), keys: [] });

function write(tallies: CorpusTally[]): void {
  const report = differentialReport(reportInputs(engine, configs), tallies);
  mkdirSync(SELF_REPORT_DIR, { recursive: true });
  writeFileSync(join(SELF_REPORT_DIR, "self-report.json"), `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(join(SELF_REPORT_DIR, "self-report.md"), renderReport(report, SELF_REPORT_TITLE));
  console.error(`wrote ${SELF_REPORT_DIR}self-report.{json,md}`);
}

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const at = args.indexOf(name);
  return at < 0 ? undefined : args[at + 1];
};
const part = flag("--part");
const merge = flag("--merge");
if (part !== undefined) {
  const dir = flag("--parts") ?? ".";
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, `${part}.json`), JSON.stringify(toJson(tallyOf(part))));
} else if (merge !== undefined) {
  write(COMMITTED_CORPORA.map((c) => fromJson(JSON.parse(readFileSync(join(merge, `${c.id}.json`), "utf8")) as ReturnType<typeof toJson>)));
} else {
  write(COMMITTED_CORPORA.map((c) => tallyOf(c.id)));
}
