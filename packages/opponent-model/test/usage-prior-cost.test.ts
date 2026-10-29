// The belief under the committed Smogon usage prior (#1241). Under it the lists are long and the residuals cover the whole
// vocabulary, so the numbers the belief multiplies are thousands of bits long, and the cost was dominated by gcds of such
// numbers (a belief took 9 to 35 s on the four requests below; now about a second). The digests were taken from the computation
// BEFORE the change (Euclid gcd, a reduction per cell): the cost work may not move a byte of a belief. The time gate is
// loose (a shared CI machine) and only catches a return to the old cost.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { describe, expect, it } from "vitest";
import { beliefDigest } from "../src/index.ts";
import { beliefAt, games, points } from "./corpus.ts";
import { ROOT, realVocabulary } from "./real.ts";

const prior = JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", realVocabulary.format_id, "usage-prior.2026-09-1630.v2.json"), "utf8")) as UsagePrior;

/**
 * game, seat, request, belief digest. The digests of the cost change were taken before it; belief/5 (#1132) moved them by the
 * format string and the new `damage` block alone: the belief with those two put back (`format` belief/4, no `damage`) digests to the
 * old values (d025959a..., 9fe9ac4f..., 53533000..., b4038298...).
 */
const CASES: readonly (readonly [string, "p1" | "p2", number, string])[] = [
  ["random/b00", "p1", 18, "ef221a47dc560d49862e3cb7299f2a0966572783c6b8d72917c522b330bfbcaa"],
  ["baton-pass-volatiles", "p2", 14, "4bc10d82231dde3cddd39c1031eb5721a21d4e696c14bdd79776f2aebdb01ab2"],
  ["transform", "p2", 23, "aac9c819da130b135cffad68f389aab631f80a53b7614c6f3b35f7b698ee9b1b"],
  ["revival", "p1", 18, "1beefd91c0db38c4eeeb2a9da2824589232ef08a5dc83cd0fd28af0cb5558a61"],
];

describe("the belief under the usage prior on its slowest committed requests", () => {
  it.each(CASES)("%s %s request %i keeps its digest and stays fast", { timeout: 120_000 }, (game, seat, request, digest) => {
    const point = points(games().find((g) => g.id === game)!).find((p) => p.seat === seat && p.request === request)!;
    const t0 = performance.now();
    const b = beliefAt(point, prior);
    const ms = performance.now() - t0;
    expect(beliefDigest(b)).toBe(digest);
    // 8 s: the old cost was 9 to 35 s on this machine and the new one is about 1 s.
    expect(ms).toBeLessThan(8000);
  });
});
