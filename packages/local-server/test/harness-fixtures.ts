import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { FIXTURE_FORMAT, RECORDINGS } from "../src/recordings.ts";
import { type HarnessFixture, type HarnessFixtureV2, upgradeFixture } from "../src/harness.ts";

export const A2_DIR = fileURLToPath(new URL(`../../../fixtures/a2/${FIXTURE_FORMAT}/`, import.meta.url));

/** A committed file as it is stored: schema version 2 (one connection per seat) or 3. */
export function stored(file: string): HarnessFixture | HarnessFixtureV2 {
  return JSON.parse(readFileSync(`${A2_DIR}${file}`, "utf8")) as HarnessFixture | HarnessFixtureV2;
}

/** A committed fixture as the current schema stores it. */
export function committed(file: string): HarnessFixture {
  return upgradeFixture(stored(file));
}

export const COMMITTED = RECORDINGS.map((recording) => ({ recording, fixture: committed(recording.file) }));
