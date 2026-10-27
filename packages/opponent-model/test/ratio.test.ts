import { describe, expect, it } from "vitest";
import { ONE, ZERO, add, div, gcd, greater, integerise, isZero, lcm, mul, ratio, sum } from "../src/ratio.ts";

describe("exact ratios", () => {
  it("reduces to lowest terms with a positive denominator", () => {
    expect(ratio(6n, -4n)).toEqual({ num: -3n, den: 2n });
    expect(ratio(0n, 7n)).toEqual(ZERO);
    expect(() => ratio(1n, 0n)).toThrow(RangeError);
  });
  it("adds, multiplies and divides exactly", () => {
    expect(add(ratio(1n, 3n), ratio(1n, 6n))).toEqual(ratio(1n, 2n));
    expect(mul(ratio(2n, 3n), ratio(3n, 4n))).toEqual(ratio(1n, 2n));
    expect(div(ratio(1n, 2n), ratio(1n, 4n))).toEqual(ratio(2n));
    expect(sum([ratio(1n, 4n), ratio(1n, 4n), ratio(1n, 2n)])).toEqual(ONE);
    expect(isZero(ZERO)).toBe(true);
    expect(greater(ratio(2n, 3n), ratio(3n, 5n))).toBe(true);
    expect(greater(ratio(3n, 5n), ratio(2n, 3n))).toBe(false);
  });
  it("gcd and lcm treat zero and negatives", () => {
    expect(gcd(-12n, 18n)).toBe(6n);
    expect(gcd(0n, 0n)).toBe(0n);
    expect(lcm(4n, 6n)).toBe(12n);
    expect(lcm(0n, 6n)).toBe(0n);
  });
  it("scales a list to integers in the same proportion", () => {
    expect(integerise([ratio(1n, 2n), ratio(1n, 3n), ratio(5n, 6n)])).toEqual({ scale: 6n, weights: [3n, 2n, 5n] });
    expect(integerise([])).toEqual({ scale: 1n, weights: [] });
  });
});
