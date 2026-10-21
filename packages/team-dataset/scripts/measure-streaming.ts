// Measure the streaming dataset I/O on a synthetic split (#883). No engine and
// no validation: only the primitives `writeDataset`, `recordHashes` and the
// known-hash build are made of. Each record is a padded JCS line, so the file
// is large even where the record is small.
//   node --max-old-space-size=2048 packages/team-dataset/scripts/measure-streaming.ts [--records N] [--bytes B] [--old]
// `--old` runs the previous whole-string code (joined output, readFileSync) to show where it fails.
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { jcs } from "@areneon/showdown-bridge";
import { recordHashes, writeLines } from "../src/dataset-io.ts";
import { KnownHashes, referencePoolHash, sortedUnique } from "../src/dedupe.ts";

const arg = (name: string, fallback: number): number => {
  const at = process.argv.indexOf(`--${name}`);
  return at === -1 ? fallback : Number(process.argv[at + 1]);
};
const records = arg("records", 1_200_000);
const width = arg("bytes", 700);
const old = process.argv.includes("--old");

let peak = 0;
const rss = (): number => {
  peak = Math.max(peak, process.memoryUsage().rss);
  return peak;
};
const mb = (n: number): string => `${Math.round(n / 1048576)} MB`;
const sample = setInterval(rss, 20);

const line = (k: number): string => jcs({ canonical_team_hash: createHash("sha256").update(`team ${k}`).digest("hex"), pad: "x".repeat(width), index: k } as never);
const dir = mkdtempSync(join(tmpdir(), "measure-streaming-"));
const file = join(dir, "records.training.jsonl");
const report = (step: string, started: number): void => console.log(`${step}: ${((Date.now() - started) / 1000).toFixed(1)} s, peak RSS so far ${mb(rss())}`);

try {
  let t = Date.now();
  if (old) {
    const all: string[] = [];
    for (let k = 0; k < records; k++) all.push(`${line(k)}\n`);
    writeFileSync(file, all.join(""));
  } else {
    writeLines(file, (function* () { for (let k = 0; k < records; k++) yield line(k); })());
  }
  console.log(`${records} records, file ${mb(statSync(file).size)}`);
  report("write", t);
  t = Date.now();
  const hashes = old
    ? readFileSync(file, "utf8").split("\n").filter((l) => l !== "").map((l) => (JSON.parse(l) as { canonical_team_hash: string }).canonical_team_hash)
    : recordHashes([file]);
  report("recordHashes", t);
  t = Date.now();
  const sorted = sortedUnique(hashes);
  const known = new KnownHashes(sorted);
  const pool = referencePoolHash(sorted);
  report("known-hash build", t);
  console.log(`${known.size} known hashes, pool ${pool.slice(0, 12)}, has(first)=${known.has(sorted[0] as string)}`);
} catch (error) {
  console.log(`failed: ${(error as Error).name}: ${(error as Error).message}`);
  process.exitCode = 1;
} finally {
  clearInterval(sample);
  rmSync(dir, { recursive: true, force: true });
}
