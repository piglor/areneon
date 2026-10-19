import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { FIXTURE_FORMAT, RECORDINGS } from "../src/recordings.ts";
import type { HarnessFixture } from "../src/harness.ts";

export const A2_DIR = fileURLToPath(new URL(`../../../fixtures/a2/${FIXTURE_FORMAT}/`, import.meta.url));

export function committed(file: string): HarnessFixture {
  return JSON.parse(readFileSync(`${A2_DIR}${file}`, "utf8")) as HarnessFixture;
}

export const COMMITTED = RECORDINGS.map((recording) => ({ recording, fixture: committed(recording.file) }));
