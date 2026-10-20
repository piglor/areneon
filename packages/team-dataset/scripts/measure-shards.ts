// Time and memory of writing and reading shards (ADR-013 section 4, budget
// about 2 s to rebuild a 2^18-row shard).
//   DUCKDB_BIN=... node packages/team-dataset/scripts/measure-shards.ts DATASET_DIR OUT_DIR [--split dev] [--shard-key HEX] [--read-only]
// (--read-only reads an existing OUT_DIR/snapshot.json and measures the read alone.)
// DATASET_DIR is a generator output (manifest.json, records.<split>.jsonl). Prints one JSON object:
// the write (stream and stage, sort, DuckDB, read-back check), then a second, separate read
// of every shard (decode, rebuild, hash and verify). Peak RSS is the process's maximum so far (it only grows).
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { readTeamDatasetManifest, readTeamLabelSet, readTeamSnapshotManifest } from "@areneon/showdown-bridge";
import { convertDataset, datasetPlan, DuckdbWriter, eachSnapshotShard, locateDuckdb, type Runner, shardsInDirectory } from "../src/index.ts";

const { values, positionals } = parseArgs({ args: process.argv.slice(2), allowPositionals: true, options: { "read-only": { type: "boolean" }, split: { type: "string", default: "dev" }, "shard-key": { type: "string", default: "ab".repeat(32) } } });
const [datasetDir, outDir] = positionals;
if (datasetDir === undefined || outDir === undefined) throw new Error("usage: measure-shards.ts DATASET_DIR OUT_DIR [--split dev] [--shard-key HEX]");
const split = values.split as "dev" | "validation" | "training";
const root = new URL("../../..", import.meta.url).pathname;
const manifest = readTeamDatasetManifest(JSON.parse(readFileSync(join(datasetDir, "manifest.json"), "utf8")));
const labelSet = readTeamLabelSet(JSON.parse(readFileSync(join(root, "config/teams", manifest.generator.manifest.format_id, "labels.json"), "utf8")));
const binary = locateDuckdb();
if (binary === undefined) throw new Error("no DuckDB command-line tool (set DUCKDB_BIN)");
let duckdbMs = 0;
const timed: Runner = (file, args, input, env) => {
  const start = performance.now();
  const result = spawnSync(file, [...args], { input, env, encoding: "utf8" });
  if (args[0] !== "--version") duckdbMs += performance.now() - start;
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
};
const writer = DuckdbWriter.open(binary, timed);
const maxRssMb = (): number => Math.round(process.resourceUsage().maxRSS / 1024);

const writeStart = performance.now();
const snapshot =
  values["read-only"] === true
    ? readTeamSnapshotManifest(JSON.parse(readFileSync(join(outDir, "snapshot.json"), "utf8")))
    : (await convertDataset({ datasetDir, outDir, labelSet, shardKey: values["shard-key"] as string, snapshotId: "measure", splits: [split], writer })).snapshot;
const writeMs = values["read-only"] === true ? 0 : performance.now() - writeStart;
const writePeak = maxRssMb();
const plan = datasetPlan(manifest, labelSet);
const readStart = performance.now();
const perShard: { rows: number; ms: number }[] = [];
let last = performance.now();
for await (const { entry } of eachSnapshotShard(snapshot, shardsInDirectory(outDir), plan)) {
  const now = performance.now();
  perShard.push({ rows: entry.rows, ms: Math.round(now - last) });
  last = now;
}
const readMs = performance.now() - readStart;
const rows = snapshot.shards.reduce((n, s) => n + s.rows, 0);
const bytes = snapshot.shards.reduce((n, s) => n + s.bytes, 0);
console.log(
  JSON.stringify(
    {
      writer: snapshot.writer,
      rows,
      shards: snapshot.shards.length,
      parquet_bytes: bytes,
      bytes_per_team: Math.round(bytes / rows),
      write_ms: Math.round(writeMs),
      write_duckdb_ms: Math.round(duckdbMs),
      peak_rss_mb_after_write: writePeak,
      read_ms: Math.round(readMs),
      read_us_per_record: Number(((readMs * 1000) / rows).toFixed(1)),
      peak_rss_mb_after_read: maxRssMb(),
      per_shard: perShard,
    },
    null,
    1,
  ),
);
