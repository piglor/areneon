import { describe, expect, it } from "vitest";
import { elementary, moveModel } from "../src/moves.ts";
import { ratio } from "../src/ratio.ts";
import { elementaryWithout } from "../src/moves.ts";

// The worked case: 10 sets of 4 slots. `ma` is in every set (forced), `mb` and `mc` in half, `md`
// (not named) takes the residual 5 over its single option: p = 5 / (1 * 10) = 1/2. Inclusions
// 10 + 5 + 5 plus 15 empty slots plus the residual 5 fill 40. Every open move has odds 1.
const list = { denominator: 10, slots: 4, entries: [{ id: "ma", inclusion: 10 }, { id: "mb", inclusion: 5 }, { id: "mc", inclusion: 5 }], empty_slots: 15, residual: 5 };
const pool = ["ma", "mb", "mc", "md"];

describe("elementary symmetric polynomials", () => {
  it("computes e_0..e_d of the weights", () => {
    expect(elementary([2n, 3n, 5n], 3)).toEqual([1n, 10n, 31n, 30n]);
    expect(elementary([], 2)).toEqual([1n, 0n, 0n]);
  });
  it("removes a weight by the exact recurrence", () => {
    // without 3: weights 2 and 5 give 1, 7, 10
    expect(elementaryWithout(elementary([2n, 3n, 5n], 3), 3n, 2)).toEqual([1n, 7n, 10n]);
  });
});

describe("the conditional Poisson move model", () => {
  it("the two-point count and the inclusions of the worked case", () => {
    const got = moveModel(list, pool, []);
    expect(got).not.toBeNull();
    const { model, likelihood } = got!;
    expect(likelihood).toEqual(ratio(1n));
    expect(model.forced).toEqual(["ma"]);
    expect(model.revealed).toEqual([]);
    // e = 15/10 = 1.5 empty slots: 2 moves with probability 1/2 and 3 moves with probability 1/2.
    expect(model.outcomes).toEqual([{ size: 2, weight: ratio(1n, 2n) }, { size: 3, weight: ratio(1n, 2n) }]);
    expect(model.candidates).toEqual([{ id: "mb", odds: 1n }, { id: "mc", odds: 1n }, { id: "md", odds: 1n }]);
    // size 2: one of three, 1/3 each; size 3: two of three, 2/3 each; mixed 1/2.
    expect(model.inclusion).toEqual([{ id: "ma", p: ratio(1n) }, { id: "mb", p: ratio(1n, 2n) }, { id: "mc", p: ratio(1n, 2n) }, { id: "md", p: ratio(1n, 2n) }]);
  });

  it("conditions on a revealed move: the likelihood is its inclusion and the rest follow", () => {
    const { model, likelihood } = moveModel(list, pool, ["mb"])!;
    expect(likelihood).toEqual(ratio(1n, 2n)); // P(mb in the set) = 1/2
    expect(model.revealed).toEqual(["mb"]);
    // size 2: P(mb) = 1/3 -> weight 1/6; size 3: P(mb) = 2/3 -> weight 1/3; posterior 1/3 and 2/3.
    expect(model.outcomes).toEqual([{ size: 2, weight: ratio(1n, 3n) }, { size: 3, weight: ratio(2n, 3n) }]);
    // mc: size 2 -> 0 (the second slot is mb), size 3 -> 1/2; mixed 2/3 * 1/2 = 1/3.
    expect(model.inclusion).toEqual([{ id: "ma", p: ratio(1n) }, { id: "mb", p: ratio(1n) }, { id: "mc", p: ratio(1n, 3n) }, { id: "md", p: ratio(1n, 3n) }]);
  });

  it("a revealed forced move changes nothing", () => {
    const a = moveModel(list, pool, [])!;
    const b = moveModel(list, pool, ["ma"])!;
    expect(b.likelihood).toEqual(ratio(1n));
    expect(b.model.inclusion).toEqual(a.model.inclusion);
  });

  it("conditions away counts below the revealed moves", () => {
    // mb and mc revealed with forced ma: at least 3 moves, so only the count 3 remains.
    const { model, likelihood } = moveModel(list, pool, ["mb", "mc"])!;
    expect(model.outcomes).toEqual([{ size: 3, weight: ratio(1n) }]);
    expect(likelihood).toEqual(ratio(1n, 6n)); // P(count 3) = 1/2, then both mb and mc among two draws of three: 1/3
    expect(model.inclusion).toEqual([{ id: "ma", p: ratio(1n) }, { id: "mb", p: ratio(1n) }, { id: "mc", p: ratio(1n) }]);
  });

  it("never lets the count fall below the revealed moves, even when the mean says fewer", () => {
    // Three revealed moves with forced ma: 4 moves, above both counts (2 and 3): the count is the minimum.
    const full = moveModel(list, pool, ["mb", "mc", "md"])!;
    expect(full.model.outcomes).toEqual([{ size: 4, weight: ratio(1n) }]);
    expect(full.likelihood).toEqual(ratio(1n));
  });

  it("weights unequal odds exactly", () => {
    // p = 1/2 (odds 1) and p = 3/4 (odds 3): one draw of two candidates picks the second with 3/4.
    const l = { denominator: 4, slots: 2, entries: [{ id: "x", inclusion: 2 }, { id: "y", inclusion: 3 }], empty_slots: 4, residual: 0 };
    const { model } = moveModel(l, ["x", "y"], [])!;
    // empty_slots / D = 1 -> size 2 - 1 = 1 move exactly.
    expect(model.outcomes).toEqual([{ size: 1, weight: ratio(1n) }]);
    expect(model.inclusion).toEqual([{ id: "x", p: ratio(1n, 4n) }, { id: "y", p: ratio(3n, 4n) }]);
  });

  it("a residual of a single option that fills every set is forced", () => {
    const l = { denominator: 2, slots: 2, entries: [], empty_slots: 0, residual: 4 };
    // pool of two moves, residual 4 over n = 2: each p = 4 / (2 * 2) = 1: both forced.
    const { model } = moveModel(l, ["x", "y"], [])!;
    expect(model.forced).toEqual(["x", "y"]);
  });

  it("refuses revealed moves no set can hold", () => {
    expect(moveModel(list, pool, ["mb", "mc", "md", "me"])).toBeNull(); // me is not a move of the pool: it has no weight
    const noResidual = { ...list, residual: 0, empty_slots: 20 };
    // md has no probability at all (no entry, no residual): revealing it has no support.
    expect(moveModel(noResidual, pool, ["md"])).toBeNull();
  });

  it("caps the count at the moves that can be drawn", () => {
    // Only ma is possible (the rest have zero probability): a count of 3 is capped to 1.
    const l = { denominator: 2, slots: 4, entries: [{ id: "ma", inclusion: 2 }], empty_slots: 0, residual: 0 };
    const { model } = moveModel(l, ["ma", "mb"], [])!;
    expect(model.outcomes).toEqual([{ size: 1, weight: ratio(1n) }]);
    expect(model.inclusion).toEqual([{ id: "ma", p: ratio(1n) }]);
  });
});
