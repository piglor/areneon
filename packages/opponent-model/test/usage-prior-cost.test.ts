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

/** game, seat, request, belief digest before the change. */
const CASES: readonly (readonly [string, "p1" | "p2", number, string])[] = [
  ["random/b00", "p1", 18, "d025959ab3e960c6c28d061ca35be75d5833c8e373e3e3735f833cfbc447bf9e"],
  ["baton-pass-volatiles", "p2", 14, "9fe9ac4ff1f55df5209cd73bf4c02c09cdf23f501e9431558b480cd6b7f41b2b"],
  ["transform", "p2", 23, "53533000d41737680edb884b43d1f1cbe680abab3a4061ad2f48e16bdcf88f7d"],
  ["revival", "p1", 18, "b4038298fd2222bb6c3632b255d0908b3602516e3d67ad19d9728cb38f2fb78f"],
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
