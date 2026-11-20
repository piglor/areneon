// The BattleTrace corpus samples (fixtures/contracts/v1/battle-trace/), read by this package's reader: valid ones are read back
// unchanged, schema-invalid ones are refused by the schema and each reader-invalid one for its rule.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { endStep, readBattleTrace, TraceError } from "../src/battle-trace.ts";

const SAMPLES = fileURLToPath(new URL("../../../fixtures/contracts/", import.meta.url));
const DIR = "v1/battle-trace";
const load = (path: string): unknown => JSON.parse(readFileSync(join(SAMPLES, path), "utf8"));
const names = (kind: string): string[] => readdirSync(join(SAMPLES, DIR, kind)).sort();

const READER_INVALID: Record<string, string> = {
  "ends-early": "only it, ends the battle",
  "never-ends": "only it, ends the battle",
  "request-gap": "numbered 1, 2",
  "start-with-action": "only the start step has no action",
  "step-without-action": "only the start step has no action",
  "team-hash-mismatch": "does not hash to its team_hashes entry",
  "wrong-winner": "is not the one the last step's",
};

describe("the BattleTrace samples", () => {
  it.each(names("valid"))("reads %s unchanged", (file) => {
    const value = load(`${DIR}/valid/${file}`);
    const trace = readBattleTrace(structuredClone(value));
    expect(trace).toEqual(value);
    expect(endStep(trace)).toBe(trace.steps.length - 1);
  });
  it.each(names("invalid"))("refuses %s by the schema", (file) => {
    expect(() => readBattleTrace(load(`${DIR}/invalid/${file}`))).toThrow("invalid BattleTrace");
  });
  it("lists every reader-invalid sample", () => {
    expect(names("reader-invalid")).toEqual(Object.keys(READER_INVALID).map((n) => `${n}.json`));
  });
  it.each(Object.entries(READER_INVALID))("refuses %s: %s", (name, message) => {
    const read = () => readBattleTrace(load(`${DIR}/reader-invalid/${name}.json`));
    expect(read).toThrow(TraceError);
    expect(read).toThrow(message);
  });
});
