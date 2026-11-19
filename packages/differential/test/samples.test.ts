// The BackendAdmission corpus samples (fixtures/contracts/v1/backend-admission/), read by this package's reader: valid ones
// are read back unchanged and hash to the digests the corpus records; each reader-invalid one is refused for its rule (the
// Python reader refuses the same samples with the same words, tests/contracts/test_backend_admission.py).
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AdmissionError, admissionHash, readBackendAdmission } from "../src/admission.ts";
import { PIN } from "./support.ts";

const SAMPLES = fileURLToPath(new URL("../../../fixtures/contracts/", import.meta.url));
const DIR = "v1/backend-admission";
const hashes = JSON.parse(readFileSync(join(SAMPLES, "hashes.json"), "utf8")) as Record<string, string>;
const load = (path: string): unknown => JSON.parse(readFileSync(join(SAMPLES, path), "utf8"));
const names = (kind: string): string[] => readdirSync(join(SAMPLES, DIR, kind)).sort();

const READER_INVALID: Record<string, string> = {
  "count-mismatch": "count mismatch",
  "counts-not-nested": "must nest",
  "fresh-below-minimum": "minimum size",
  "fresh-repeated-items": "each item once",
  "kinds-not-query-kind": "sorted and unique",
  "kinds-on-query-kind": "only an entity the fresh corpus holds",
  "missing-revocation": "revoked exactly when",
  "nineteen-exercising": "admitted flag",
  "other-pin": "another pinned engine",
  "pair-flag": "pair's admitted flag",
  "pair-not-admitted-entity": "two admitted entities",
  "rare-pair": "1 in 10^4",
  "regression-evidence": "must come from the fresh corpus",
  "revocation-past-corpus": "within its corpus",
  "two-fresh": "at most one fresh",
  "unsorted-entities": "entities must be sorted",
  "wrong-fresh-seed": "fresh seed",
};

describe("the BackendAdmission samples", () => {
  it.each(names("valid"))("reads %s unchanged, with the corpus's digest", (file) => {
    const value = load(`${DIR}/valid/${file}`);
    const ledger = readBackendAdmission(value, { showdown_commit: PIN });
    expect(ledger).toEqual(value);
    expect(admissionHash(ledger)).toBe(hashes[`${DIR}/valid/${file}`]);
  });
  it.each(names("invalid"))("refuses %s by the schema", (file) => {
    expect(() => readBackendAdmission(load(`${DIR}/invalid/${file}`), { showdown_commit: PIN })).toThrow("invalid BackendAdmission");
  });
  it("lists every reader-invalid sample", () => {
    expect(names("reader-invalid")).toEqual(Object.keys(READER_INVALID).map((n) => `${n}.json`));
  });
  it.each(Object.entries(READER_INVALID))("refuses %s: %s", (name, message) => {
    const read = () => readBackendAdmission(load(`${DIR}/reader-invalid/${name}.json`), { showdown_commit: PIN });
    expect(read).toThrow(AdmissionError);
    expect(read).toThrow(message);
  });
});
