import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { WorldFactsData } from "@areneon/public-state";
import { GOLDEN_DIR } from "../src/golden.ts";
import { type FactsTable, readWorldFactsData as read } from "../src/world-facts-data.ts";

// The reviewed engine data the world facts take (`WorldFactsData`), read from the pinned engine's own source
// files and the committed engine-data table (`src/world-facts-data.ts`). Shared by the world-facts parity test and the world builder tests.

export const FORMAT_DIR = GOLDEN_DIR;
export const engineData = JSON.parse(readFileSync(join(FORMAT_DIR, "engine-data.json"), "utf8")) as FactsTable;

export function readWorldFactsData(): WorldFactsData {
  return read(engineData);
}
