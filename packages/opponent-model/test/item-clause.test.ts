import { describe, expect, it } from "vitest";
import { MAX_CLAUSE_SLOTS, solveItemClause } from "../src/item-clause.ts";

describe("the Item Clause subset DP", () => {
  // Two slots, two items. Slot 0: x 1, y 2, none 3. Slot 1: x 4, y 5, none 6.
  const w = { items: [[1n, 2n], [4n, 5n]], none: [3n, 6n] };

  it("the normaliser and marginals of the worked case", () => {
    const r = solveItemClause(w, true);
    // 6 * 15 = 90 assignments by weight, minus (x,x) = 1*4 and (y,y) = 2*5; "no item" twice is allowed.
    expect(r.normaliser).toBe(76n);
    expect(r.items).toEqual([[11n, 20n], [20n, 20n]]);
    expect(r.none).toEqual([45n, 36n]);
  });

  it("without the clause the slots are independent", () => {
    const r = solveItemClause(w, false);
    expect(r.normaliser).toBe(90n);
    expect(r.items).toEqual([[15n, 30n], [24n, 30n]]);
    expect(r.none).toEqual([45n, 36n]);
  });

  it("matches brute force on seeded weights with four slots", () => {
    let seed = 12345;
    const next = (): bigint => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return BigInt(seed % 7);
    };
    const slots = 4;
    const m = 5;
    const items = Array.from({ length: slots }, () => Array.from({ length: m }, next));
    const none = Array.from({ length: slots }, () => next() + 1n);
    const got = solveItemClause({ items, none }, true);
    // Brute force over every assignment: -1 is "no item".
    let total = 0n;
    const marg = items.map((row) => row.map(() => 0n));
    const noneMarg = none.map(() => 0n);
    const pick: number[] = [];
    const walk = (i: number): void => {
      if (i === slots) {
        const used = pick.filter((x) => x >= 0);
        if (new Set(used).size !== used.length) return;
        let weight = 1n;
        pick.forEach((x, s) => (weight *= x < 0 ? (none[s] as bigint) : ((items[s] as bigint[])[x] as bigint)));
        total += weight;
        pick.forEach((x, s) => {
          if (x < 0) noneMarg[s] = (noneMarg[s] as bigint) + weight;
          else ((marg[s] as bigint[])[x] as bigint) += weight;
        });
        return;
      }
      for (let x = -1; x < m; x += 1) {
        pick[i] = x;
        walk(i + 1);
      }
    };
    walk(0);
    expect(got.normaliser).toBe(total);
    expect(got.items).toEqual(marg);
    expect(got.none).toEqual(noneMarg);
  });

  it("a slot that must hold the same item as another has no allowed world", () => {
    const r = solveItemClause({ items: [[1n], [1n]], none: [0n, 0n] }, true);
    expect(r.normaliser).toBe(0n);
  });

  it("handles no items at all and refuses more slots than it is built for", () => {
    expect(solveItemClause({ items: [[], []], none: [2n, 3n] }, true)).toEqual({ normaliser: 6n, items: [[], []], none: [6n, 6n] });
    const many = MAX_CLAUSE_SLOTS + 1;
    expect(() => solveItemClause({ items: Array.from({ length: many }, () => []), none: Array.from({ length: many }, () => 1n) }, true)).toThrow(RangeError);
  });

  it("works above 2^64 and stays exact", () => {
    const big = 10n ** 30n;
    const r = solveItemClause({ items: [[big], [big]], none: [big, big] }, true);
    expect(r.normaliser).toBe(3n * big * big); // (big+big)(big+big) - big*big
    expect(r.items[0]).toEqual([2n * big * big - big * big]);
  });
});
