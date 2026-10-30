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
 * game, seat, request, belief digest. The digests of the cost change were taken before it; belief/5 (#1132) and belief/6 (#1133) moved them by the
 * format string and the new `damage` and `speed` blocks alone: the belief with those put back (`format` belief/4, no `damage`, no `speed`) digests to the
 * old values (d025959a..., 9fe9ac4f..., 53533000..., b4038298...).
 */
const CASES: readonly (readonly [string, "p1" | "p2", number, string])[] = [
  ["random/b00", "p1", 18, "d9c7c7567ee7e51bfa377810eb14ee09783ee1f6865ec778edb1c7172156981c"],
  ["baton-pass-volatiles", "p2", 14, "adb4d7aecf1995edda642cfd109e61a7627a8393da12a9e7e66d11ffd6cbdff0"],
  ["transform", "p2", 23, "d8d7b9d12d19e490457bc10233c8a9b3184e8475c371ab751f75e82fdc8d90ea"],
  ["revival", "p1", 18, "fbf38a6260fbc2032785f0567dec33789d1556564a40fcb4f8a633130bd91e3f"],
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
