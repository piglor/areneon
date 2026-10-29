import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { RejectionLog } from "../src/rejection-log.ts";

const dir = mkdtempSync(join(tmpdir(), "rejection-log-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const hash = (n: number): string => n.toString(16).padStart(64, "0");

describe("the rejection log (ADR-013 section 2)", () => {
  it("writes one canonical line per entry, counts each kind and names the file by its SHA-256", () => {
    const path = join(dir, "small.jsonl");
    const log = new RejectionLog(path);
    log.write({ k: 0, reason: "invalid", canonical_team_hash: hash(1), detail: "Species Clause" });
    log.write({ k: 1, reason: "round_trip", canonical_team_hash: hash(2), detail: "not reproduced" });
    log.write({ k: 2, reason: "duplicate", canonical_team_hash: hash(3), stratum: 4 });
    log.write({ k: 3, reason: "overflow", canonical_team_hash: hash(4), stratum: 4 });
    log.write({ k: 4, reason: "overflow", canonical_team_hash: hash(5), stratum: 5 });
    log.write({ k: 5, reason: "near_duplicate", canonical_team_hash: hash(6), stratum: 5 });
    log.write({ k: 6, reason: "near_reference", canonical_team_hash: hash(7), stratum: 5 });
    const summary = log.close();
    expect(summary).toMatchObject({ rejections: 2, duplicates: 1, near_duplicates: 1, near_references: 1, overflows: 2 });
    const text = readFileSync(path, "utf8");
    expect(summary.log_sha256).toBe(createHash("sha256").update(text).digest("hex"));
    expect(text.split("\n")[0]).toBe(`{"canonical_team_hash":"${hash(1)}","detail":"Species Clause","k":0,"reason":"invalid"}`);
    expect(text.endsWith("\n")).toBe(true);
  });

  it("hashes what it wrote when it flushes in pieces", () => {
    const path = join(dir, "big.jsonl");
    const log = new RejectionLog(path);
    const detail = "x".repeat(900);
    for (let k = 0; k < 3000; k++) log.write({ k, reason: "invalid", canonical_team_hash: hash(k), detail });
    const summary = log.close();
    const text = readFileSync(path, "utf8");
    expect(text.length).toBeGreaterThan(2 ** 21);
    expect(summary.log_sha256).toBe(createHash("sha256").update(text).digest("hex"));
    expect(summary.rejections).toBe(3000);
  });

  it("is an empty file with the hash of nothing when nothing was logged", () => {
    const path = join(dir, "empty.jsonl");
    expect(new RejectionLog(path).close()).toEqual({ log_sha256: createHash("sha256").update("").digest("hex"), rejections: 0, duplicates: 0, near_duplicates: 0, near_references: 0, overflows: 0 });
    expect(readFileSync(path, "utf8")).toBe("");
  });
});
