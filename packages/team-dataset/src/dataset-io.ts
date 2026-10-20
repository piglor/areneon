// Writing and reading a generated dataset directory (JSON Lines and JSON, no
// Parquet: shards are a later slice). Every file is the canonical (JCS) form
// of its value, so the same inputs give the same bytes:
//   records.<split>.jsonl   one TeamRecord per line
//   manifest.json           the TeamDatasetManifest
//   coverage.json           the coverage report
//   pilot.json              the recorded pilot
//   rejections.jsonl        the rejection log (written during the run)

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { jcs } from "@areneon/showdown-bridge";
import type { GenerateResult } from "./generate.ts";
import { SPLITS } from "./plan.ts";

/** Indented JSON with the keys in JCS order. */
export const pretty = (value: unknown): string => `${JSON.stringify(JSON.parse(jcs(value as never)), null, 2)}\n`;

export const recordsFile = (split: string): string => `records.${split}.jsonl`;

/** Write the dataset into `dir` (which holds `rejections.jsonl` already). */
export function writeDataset(dir: string, result: GenerateResult): string[] {
  const files: string[] = [];
  for (const split of SPLITS) {
    const records = result.records[split];
    if (records === undefined) continue;
    writeFileSync(join(dir, recordsFile(split)), records.map((r) => `${jcs(r as never)}\n`).join(""));
    files.push(recordsFile(split));
  }
  writeFileSync(join(dir, "manifest.json"), pretty(result.manifest));
  writeFileSync(join(dir, "coverage.json"), pretty(result.coverage));
  writeFileSync(join(dir, "pilot.json"), pretty(result.pilot));
  return [...files, "manifest.json", "coverage.json", "pilot.json", "rejections.jsonl"];
}

/** The `canonical_team_hash` of every record of JSON Lines files: the hashes a later dataset dedupes against. */
export function recordHashes(paths: readonly string[]): string[] {
  return paths.flatMap((path) =>
    readFileSync(path, "utf8")
      .split("\n")
      .filter((line) => line !== "")
      .map((line, n) => {
        const hash = (JSON.parse(line) as { canonical_team_hash?: unknown }).canonical_team_hash;
        if (typeof hash !== "string") throw new Error(`${path} line ${n + 1} has no canonical_team_hash`);
        return hash;
      }),
  );
}
