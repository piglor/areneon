import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";
import { jcs, type Json } from "../src/canonical-json.ts";

// The committed Smogon prior of 2026-09, cutoff 1630 (#1120): read against the golden team vocabulary by the TS reader
// (the Python reader and the byte-for-byte re-derivation are tests/scripts/test_usage_prior_record.py).
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const FORMAT = "gen9championsvgc2026regmc";
const text = readFileSync(join(ROOT, "config", "opponent-model", FORMAT, "usage-prior.2026-09-1630.json"), "utf8");
const vocabulary = JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "team-vocabulary.json"), "utf8"));

describe("the committed usage prior record", () => {
  it("passes the reader, is its own JCS bytes and has the pinned prior hash", () => {
    const prior = readUsagePrior(JSON.parse(text), vocabulary);
    expect(`${jcs(prior as unknown as Json)}\n`).toBe(text);
    expect(usagePriorHash(prior)).toBe("1949b487e7685c94f0ec9d96ee39889d04396d64c1665483e18ed18906ebc846");
    expect(prior.source.kind).toBe("smogon-chaos");
    expect(prior.species).toHaveLength(231);
  });
});
