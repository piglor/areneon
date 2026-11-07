// The pure search decision's own pieces (ADR-026 erratum E2, SD-b): the q_ppm arithmetic and the label's refusals, with stand-in ports
// (no engine). The searched decisions themselves are covered with the real engine in the search actor tests.
import { describe, expect, it } from "vitest";
import type { Candidate } from "../src/choices.ts";
import { qPpm, searchLabel, type LabelInput, type SearchDecisionDeps } from "../src/search-decision.ts";
import { loadSearchConfig } from "../src/search-config.ts";
import type { DecisionWorlds, WorldSampler } from "../src/world-sampler.ts";

const label = loadSearchConfig("gen9championsvgc2026regmc", "rerank-1-label.v1");
const cand = (choice: string, pending = false): Candidate => ({ choice, pending });
const input = (candidates: Candidate[]): LabelInput => ({ world: {} as never, channel: [], candidates, prior_ppm: candidates.map(() => 500_000), stream: new Uint8Array(32) });
const depsWith = (sampled: DecisionWorlds): SearchDecisionDeps => ({ config: label, sampler: { sample: () => sampled } as unknown as WorldSampler } as unknown as SearchDecisionDeps);

describe("qPpm", () => {
  it("is the floored weighted mean of the leaf values, clamped to 0..10^6", () => {
    expect(qPpm(2 * 1_000_000 * 500_000, 2)).toBe(500_000);
    expect(qPpm(2 * 1_000_000 * 500_000 + 1_999_999, 2)).toBe(500_000);
    expect(qPpm(3 * 1_000_000 * 1_000_000, 3)).toBe(1_000_000);
    expect(qPpm(4 * 1_000_000 * 1_200_000, 4)).toBe(1_000_000);
    expect(qPpm(-5, 2)).toBe(0);
    expect(qPpm(0, 2)).toBe(0);
  });
});

describe("searchLabel refusals", () => {
  const two = [cand("move 1 1, move 1 2"), cand("move 2 1, move 1 2")];

  it("refuses a request with fewer than two non-pending candidates, naming the prior", async () => {
    const lone = [cand("move 1 1"), cand("move 2 1", true)];
    expect(await searchLabel(input(lone), depsWith({ kind: "refused" } as DecisionWorlds))).toEqual({ kind: "refused", reason: "too_few_candidates", prior_ppm: [500_000, 500_000] });
  });

  it("refuses with the belief's refusal and with too few valid worlds", async () => {
    expect(await searchLabel(input(two), depsWith({ kind: "refused" } as DecisionWorlds))).toMatchObject({ kind: "refused", reason: "belief_refused" });
    expect(await searchLabel(input(two), depsWith({ kind: "sampled", searched: false } as unknown as DecisionWorlds))).toMatchObject({ kind: "refused", reason: "below_min_valid" });
  });
});
