// The cost measurement on a small run and its gate (ADR-025 slice 2 part 4): per decision (one belief) and per world (one
// sample) on committed channels, one thread. The gate is deliberately loose (a shared CI machine): it catches a change of
// kind (an exponential blow-up), not a few percent; the journaled figures come from `scripts/measure.ts`.

import { beforeAll, describe, expect, it } from "vitest";
import { createSetLegality } from "../../showdown-bridge/src/set-legality.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import { type Sampler, createSampler } from "../src/index.ts";
import { games, skewedPrior } from "./corpus.ts";
import { measure, percentiles } from "./measure.ts";
import { realVocabulary, uniformPrior } from "./real.ts";

/** Per decision and per world, in milliseconds. */
const GATE = { decision_p99_ms: 2000, world_p99_ms: 2000 };

let sampler: Sampler;
beforeAll(() => {
  sampler = createSampler({
    legality: createSetLegality({ engine: loadEngine(), vocabulary: realVocabulary }),
    spreads: { other: (_s, _f, below) => ({ nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id, stat_points: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: Number(below(2n)) } }) },
  });
});

describe("percentiles", () => {
  it("are nearest-rank values of the sorted list", () => {
    const p = percentiles([5, 1, 3, 2, 4, 10, 9, 8, 7, 6]);
    expect(p).toEqual({ n: 10, median_ms: 5, p95_ms: 10, p99_ms: 10, max_ms: 10 });
    expect(percentiles([7])).toEqual({ n: 1, median_ms: 7, p95_ms: 7, p99_ms: 7, max_ms: 7 });
  });
});

describe("cost per decision and per world on the committed channels", () => {
  const list = games({ random: 2 }).slice(0, 4);
  it.each([
    ["uniform", uniformPrior],
    ["skewed", skewedPrior],
  ] as const)("is within the gate under the %s prior, and a world needs no legality redraw", (_name, prior) => {
    const cost = measure(list, prior, sampler, { every: 3, perBelief: 3 });
    expect(cost.decision.n).toBeGreaterThan(40);
    expect(cost.worlds).toBeGreaterThan(20);
    expect(cost.decision.p99_ms).toBeLessThan(GATE.decision_p99_ms);
    expect(cost.first_world.p99_ms).toBeLessThan(GATE.world_p99_ms);
    expect(cost.later_world.p99_ms).toBeLessThan(GATE.world_p99_ms);
    expect(cost.later_world.n).toBeGreaterThan(0);
    expect(cost.redraw_ppm).toBeLessThan(500_000);
    expect(cost.refused).toBeGreaterThanOrEqual(0);
  });
});
