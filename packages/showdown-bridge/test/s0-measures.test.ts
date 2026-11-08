import { describe, expect, it } from "vitest";
import type { Candidate } from "../src/choices.ts";
import { aucPpm, bootstrapInterval, doubledRanks, exhaustiveBest, leafQuality, meanFloor, rolloutSums, siblingCalibration, spearmanBrownPpm, spearmanPpm, splitHalf, top1AgreementPpm, top1HitPpm, topKRecall } from "../src/s0-measures.ts";

const cand = (n: number): Candidate[] => Array.from({ length: n }, (_, i) => ({ choice: `move ${i + 1}, move 1`, pending: false })) as Candidate[];

describe("ranks and Spearman", () => {
  it("gives tied values the average rank, doubled", () => {
    expect(doubledRanks([10, 20, 20, 30])).toEqual([2, 5, 5, 8]);
    expect(doubledRanks([5, 5, 5])).toEqual([4, 4, 4]);
  });
  it("is 1, -1 and a hand-worked value", () => {
    expect(spearmanPpm([1, 2, 3, 4], [10, 20, 30, 40])).toBe(1_000_000);
    expect(spearmanPpm([1, 2, 3, 4], [4, 3, 2, 1])).toBe(-1_000_000);
    // ranks 1,2,3,4 against 2,1,4,3: sum d^2 = 4, rho = 1 - 6*4/(4*15) = 0.6
    expect(spearmanPpm([1, 2, 3, 4], [2, 1, 4, 3])).toBe(600_000);
    // ties: x ranks 1,2.5,2.5,4 against y ranks 1,2,3,4 -> 0.9486833 (Pearson of the ranks)
    expect(spearmanPpm([1, 2, 2, 3], [1, 2, 3, 4])).toBe(948_683);
  });
  it("is null with no spread or fewer than two items, and refuses unequal lengths", () => {
    expect(spearmanPpm([1, 1, 1], [1, 2, 3])).toBeNull();
    expect(spearmanPpm([1, 2, 3], [4, 4, 4])).toBeNull();
    expect(spearmanPpm([1], [2])).toBeNull();
    expect(() => spearmanPpm([1], [1, 2])).toThrow(RangeError);
  });
  it("corrects a half correlation by Spearman-Brown", () => {
    expect(spearmanBrownPpm(600_000)).toBe(750_000);
    expect(spearmanBrownPpm(0)).toBe(0);
    expect(spearmanBrownPpm(-1_000_000)).toBeNull();
  });
  it("means round down and an empty list has none", () => {
    expect(meanFloor([1, 2])).toBe(1);
    expect(meanFloor([])).toBeNull();
  });
});

describe("bootstrap interval", () => {
  it("is the mean and a bracket for a constant, deterministic for a label, different for another", () => {
    expect(bootstrapInterval([300_000, 300_000, 300_000], "a")).toEqual({ mean_ppm: 300_000, low_ppm: 300_000, high_ppm: 300_000, n: 3 });
    const v = [100_000, -50_000, 400_000, 250_000, 0, 120_000, 90_000, 310_000];
    const a = bootstrapInterval(v, "x");
    expect(bootstrapInterval(v, "x")).toEqual(a);
    expect(a.low_ppm).toBeLessThanOrEqual(a.mean_ppm);
    expect(a.high_ppm).toBeGreaterThanOrEqual(a.mean_ppm);
    expect(a.low_ppm).toBeGreaterThanOrEqual(-50_000);
    expect(a.high_ppm).toBeLessThanOrEqual(400_000);
    expect(bootstrapInterval(v, "y")).not.toEqual(a);
    expect(() => bootstrapInterval([], "x")).toThrow(RangeError);
  });
});

describe("top-k recall", () => {
  // five candidates in canonical order, prior favours 4,3,2,1,0 (index 4 first)
  const prior = [100, 200, 300, 400, 500];
  const mk = (sums: number[]) => ({ candidates: cand(5), prior_ppm: prior, scores: sums.map((sum, candidate) => ({ candidate, sum })) });
  it("picks the highest sum, the first in prior rank on a tie", () => {
    expect(exhaustiveBest(mk([9, 1, 1, 1, 1]))).toBe(0);
    expect(exhaustiveBest(mk([5, 1, 1, 1, 5]))).toBe(4); // tie: index 4 has the higher prior
    expect(() => exhaustiveBest({ candidates: cand(1), prior_ppm: [1], scores: [] })).toThrow(RangeError);
  });
  it("counts the decisions whose best candidate the rule searches", () => {
    // best at prior rank 0, 3 and 4 (indices 4, 1, 0)
    const decisions = [mk([0, 0, 0, 0, 9]), mk([0, 9, 0, 0, 0]), mk([9, 0, 0, 0, 0])];
    expect(topKRecall(decisions, "prior-top-k", 1)).toEqual({ rule: "prior-top-k", k: 1, decisions: 3, hits: 1, recall_ppm: 333_333 });
    expect(topKRecall(decisions, "prior-top-k", 4).hits).toBe(2);
    expect(topKRecall(decisions, "prior-top-k", 5).recall_ppm).toBe(1_000_000);
    expect(topKRecall(decisions, "distinct-slot", 4).hits).toBe(2);
    expect(topKRecall([], "prior-top-k", 4).recall_ppm).toBeNull();
  });
  it("reads the slot-cover rule through the same function", () => {
    const row = topKRecall([mk([0, 0, 0, 0, 9])], "slot-cover", 4);
    expect(row.decisions).toBe(1);
    expect(row.hits).toBe(1);
  });
});

describe("top-1", () => {
  it("credits a tie by the expected hit", () => {
    expect(top1HitPpm(0, [1, 5, 5])).toBe(0);
    expect(top1HitPpm(1, [1, 5, 5])).toBe(500_000);
    expect(top1HitPpm(0, [9, 5, 5])).toBe(1_000_000);
    expect(top1AgreementPpm([1, 5, 5], [5, 5, 1])).toBe(250_000); // {1,2} against {0,1}: one common of 2*2
    expect(top1AgreementPpm([1, 2], [1, 2])).toBe(1_000_000);
    expect(top1AgreementPpm([1, 2], [2, 1])).toBe(0);
  });
});

describe("split-half reliability", () => {
  it("is 1 for halves that agree, the Spearman-Brown value for a hand case, and counts undefined decisions", () => {
    const same = [[1, 1, 1, 1, 0, 0, 0, 0].map((x) => x * 1_000_000), [0, 0, 0, 0, 1, 1, 1, 1].map((x) => x * 1_000_000)];
    // halves: sums (4,0) and (0,4): perfectly reversed with two items
    expect(splitHalf([same]).half_correlation_ppm).toBe(-1_000_000);
    expect(splitHalf([same]).corrected_ppm).toBeNull();
    const agree = [[2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]];
    expect(splitHalf([agree])).toEqual({ decisions: 1, defined: 1, half_correlation_ppm: 1_000_000, corrected_ppm: 1_000_000, top1_agreement_ppm: 1_000_000 });
    // three candidates; first half sums 8,4,0 -> ranks 3,2,1; second half 4,8,0 -> ranks 2,3,1; rho = 0.5
    const half = [[2, 2, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]];
    const r = splitHalf([half, [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]]);
    expect(r).toMatchObject({ decisions: 2, defined: 1, half_correlation_ppm: 500_000, corrected_ppm: 666_667 });
    expect(r.top1_agreement_ppm).toBe(250_000); // (0 + 500000) / 2
  });
  it("needs an even number of rollouts", () => {
    expect(() => splitHalf([[[1, 2, 3], [3, 2, 1]]])).toThrow(RangeError);
    expect(rolloutSums([[1, 2, 3, 4]], 0, 2)).toEqual([3]);
  });
});

describe("AUC", () => {
  it("is the Mann-Whitney share with ties counted half", () => {
    expect(aucPpm([0.9, 0.8, 0.3, 0.1].map((x) => x * 10), [true, true, false, false])).toBe(1_000_000);
    expect(aucPpm([1, 2, 1, 2], [true, false, false, true])).toBe(500_000); // pairs: (1,2) lose, (1,1) half, (2,2) half, (2,1) win
    expect(aucPpm([1, 2], [true, true])).toBeNull();
    expect(() => aucPpm([1], [true, false])).toThrow(RangeError);
  });
});

describe("sibling calibration", () => {
  it("bins pairs by the predicted difference and compares the oracle's sign and size", () => {
    const bins = siblingCalibration(
      [
        // pairs: (0,1) predicted +100k oracle +40k; (0,2) predicted +400k oracle -10k (wrong sign); (1,2) predicted +300k oracle -50k... oracle diff 40k-(-10k)?? see below
        { predicted: [500_000, 400_000, 100_000], oracle: [300_000, 260_000, 310_000] },
        { predicted: [1, 1], oracle: [5, 9] }, // tied prediction: skipped
      ],
      [0, 200_000],
    );
    // (0,1): p=100k o=40k -> bin 0, same sign. (0,2): p=400k o=-10k -> bin 1, wrong. (1,2): p=300k o=-50k -> bin 1, wrong.
    expect(bins).toEqual([
      { from_ppm: 0, pairs: 1, mean_predicted_ppm: 100_000, mean_oracle_ppm: 40_000, same_sign: 1, oracle_zero: 0 },
      { from_ppm: 200_000, pairs: 2, mean_predicted_ppm: 350_000, mean_oracle_ppm: -30_000, same_sign: 0, oracle_zero: 0 },
    ]);
  });
  it("flips a negative prediction to the preferred candidate and counts a zero oracle difference", () => {
    const [bin] = siblingCalibration([{ predicted: [0, 50_000], oracle: [7, 7] }], [0]);
    expect(bin).toEqual({ from_ppm: 0, pairs: 1, mean_predicted_ppm: 50_000, mean_oracle_ppm: 0, same_sign: 0, oracle_zero: 1 });
    expect(siblingCalibration([], [0])[0]?.mean_predicted_ppm).toBeNull();
  });
});

describe("leaf quality", () => {
  it("reports the search's and the prior's correlation, their paired difference and the top-1s", () => {
    const decisions = [
      { search: [3, 2, 1], prior: [1, 2, 3], oracle: [30, 20, 10] }, // search rho 1, prior rho -1
      { search: [1, 2, 3, 4], prior: [4, 3, 2, 1], oracle: [1, 2, 4, 3] }, // search 0.8, prior -0.8
      { search: [1, 1], prior: [1, 2], oracle: [1, 2] }, // search undefined
    ];
    const q = leafQuality(decisions, "t");
    expect(q).toMatchObject({ decisions: 3, paired: 2, search_spearman_ppm: 900_000, prior_spearman_ppm: -266_667 });
    expect(q.difference?.mean_ppm).toBe(1_800_000);
    expect(q.difference?.low_ppm).toBeGreaterThan(0);
    // top-1: search picks index 0 (oracle 30: best), index 3 (oracle 3, not best 4), index 0 of a tie (oracle 1 vs 2: miss)
    expect(q.search_top1_ppm).toBe(Math.floor((1_000_000 + 0 + 0) / 3));
    expect(leafQuality([{ search: [1, 1], prior: [1, 1], oracle: [1, 2] }], "t").difference).toBeNull();
  });
});
