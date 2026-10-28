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
const text2 = readFileSync(join(ROOT, "config", "opponent-model", FORMAT, "usage-prior.2026-09-1630.v2.json"), "utf8");
const vocabulary = JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "team-vocabulary.json"), "utf8"));

describe("the committed usage prior record", () => {
  it("passes the reader, is its own JCS bytes and has the pinned prior hash", () => {
    const prior = readUsagePrior(JSON.parse(text), vocabulary);
    expect(`${jcs(prior as unknown as Json)}\n`).toBe(text);
    expect(usagePriorHash(prior)).toBe("1949b487e7685c94f0ec9d96ee39889d04396d64c1665483e18ed18906ebc846");
    expect(prior.source.kind).toBe("smogon-chaos");
    expect(prior.species).toHaveLength(231);
  });

  it("reads the second record (#1209): the engine's ids for Farfetch'd and Sirfetch'd, 67 species from the generator-matched prior", () => {
    const prior = readUsagePrior(JSON.parse(text2), vocabulary);
    expect(`${jcs(prior as unknown as Json)}\n`).toBe(text2);
    expect(usagePriorHash(prior)).toBe("910736ec4d738cb022b7aee64110028210c6d31338c926f9579434d3304eeda7");
    expect(prior.species).toHaveLength(300);
    const fallback = prior.source.kind === "smogon-chaos" ? prior.source.fallback : undefined;
    expect(fallback?.kind).toBe("generator-matched");
    expect(fallback?.species).toHaveLength(67);
    for (const id of ["farfetchd", "sirfetchd"]) expect(fallback?.species).not.toContain(id);
    for (const id of ["farfetchd", "sirfetchd"]) expect(prior.species.some((s) => s.id === id)).toBe(true);
  });
});
