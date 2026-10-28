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

describe("gcd of large numbers (Lehmer)", () => {
  const euclid = (a: bigint, b: bigint): bigint => {
    let x = a < 0n ? -a : a;
    let y = b < 0n ? -b : b;
    while (y !== 0n) [x, y] = [y, x % y];
    return x;
  };
  // A fixed xorshift stream, so the operands are the same on every run.
  let state = 0x9e3779b97f4a7c15n;
  const next = (): bigint => {
    state ^= (state << 13n) & 0xffffffffffffffffn;
    state ^= state >> 7n;
    state ^= (state << 17n) & 0xffffffffffffffffn;
    return state;
  };
  const big = (bits: number): bigint => {
    let v = 1n;
    for (let k = 0; k < bits; k += 64) v = (v << 64n) | next();
    return v >> BigInt(64 * Math.ceil(bits / 64) - bits);
  };

  it("equals Euclid on random pairs, shared factors, very unequal sizes and powers of two", () => {
    for (const bits of [100, 200, 500, 1500, 4000]) {
      for (let k = 0; k < 12; k += 1) {
        const a = big(bits);
        const b = big(bits + (k % 3) * 40 - 30);
        const f = big(60 + 20 * k);
        for (const [x, y] of [
          [a, b],
          [a * f, b * f],
          [a, b * f],
          [a * a, b],
          [a, big(10 + k * 9)],
          [-a, b],
        ] as const) {
          expect(gcd(x, y)).toBe(euclid(x, y));
        }
      }
    }
    expect(gcd(1n << 400n, 1n << 300n)).toBe(1n << 300n);
    expect(gcd(7n << 200n, 7n << 200n)).toBe(7n << 200n);
    const v = big(300);
    expect(gcd(v, 0n)).toBe(v);
    expect(gcd(0n, v)).toBe(v);
  });
  it("equals Euclid on consecutive Fibonacci numbers (the worst case, every quotient 1) and on huge cofactor runs", () => {
    let [a, b] = [1n, 1n];
    for (let k = 0; k < 3000; k += 1) [a, b] = [b, a + b];
    expect(gcd(b, a)).toBe(1n);
    expect(gcd(b * 12345n, a * 12345n)).toBe(12345n);
    const m = (1n << 521n) - 1n;
    expect(gcd(m * 3n ** 90n, m * 5n ** 60n)).toBe(m);
  });
});

describe("add, mul and div equal the naive reduced fraction", () => {
  const naiveAdd = (a: { num: bigint; den: bigint }, b: { num: bigint; den: bigint }) => ratio(a.num * b.den + b.num * a.den, a.den * b.den);
  const naiveMul = (a: { num: bigint; den: bigint }, b: { num: bigint; den: bigint }) => ratio(a.num * b.num, a.den * b.den);
  const values = [0n, 1n, -1n, 2n, -3n, 6n, 35n, 1n << 70n, 3n ** 60n, -(5n ** 50n), 2n ** 100n * 3n ** 20n];
  const pool = values.flatMap((n) => values.filter((d) => d > 0n).map((d) => ratio(n, d)));
  it("on every pair of a pool with shared factors, zeros, ones and negatives", () => {
    for (const a of pool) {
      for (const b of pool) {
        expect(add(a, b)).toEqual(naiveAdd(a, b));
        expect(mul(a, b)).toEqual(naiveMul(a, b));
        if (b.num !== 0n) expect(div(a, b)).toEqual(ratio(a.num * b.den, a.den * b.num));
      }
    }
  });
  it("refuses to divide by zero and cancels to exactly zero", () => {
    expect(() => div(ONE, ZERO)).toThrow(RangeError);
    expect(add(ratio(1n, 6n), ratio(-1n, 6n))).toEqual(ZERO);
    expect(add(ratio(1n, 6n), ratio(-1n, 4n))).toEqual(ratio(-1n, 12n));
  });
});
