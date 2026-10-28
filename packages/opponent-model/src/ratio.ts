// Exact rational arithmetic on BigInt. The belief is computed with no floating point: every
// probability is a reduced `Ratio` and every normaliser an integer (ADR-025 section 3).

/** A reduced fraction with a positive denominator. */
export interface Ratio {
  readonly num: bigint;
  readonly den: bigint;
}

/** Plain Euclid; fast once the operands are small. */
function euclid(a: bigint, b: bigint): bigint {
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}

// Below this size Euclid's remainders are cheap; above it Lehmer's steps pay off.
const LEHMER_FROM = 1n << 96n;
const LEHMER_BITS = 40;

/**
 * The greatest common divisor, exact. Lehmer's algorithm (Knuth 4.5.2, Algorithm L): the leading
 * 37 to 40 bits of both operands drive a run of Euclid steps whose cofactors are applied to the full numbers
 * at once, so a gcd of two thousand-bit numbers costs a few hundred long multiplications and not
 * a long division per quotient. Every quotient of the run is the one the full numbers have (the
 * double test of the algorithm), so the result is the Euclid result and the belief's bytes do not move.
 * The cofactors stay below 2^40, so every Number below is an exact integer; `%` is exact on them and
 * `(x - x % y) / y` is their exact quotient (no rounding, no `Math`).
 */
export const gcd = (a: bigint, b: bigint): bigint => {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  if (x < y) [x, y] = [y, x];
  while (y >= LEHMER_FROM) {
    const shift = BigInt(x.toString(16).length * 4 - LEHMER_BITS);
    let xh = Number(x >> shift);
    let yh = Number(y >> shift);
    let A = 1;
    let B = 0;
    let C = 0;
    let D = 1;
    for (;;) {
      if (yh + C === 0 || yh + D === 0) break;
      const num1 = xh + A;
      const den1 = yh + C;
      const num2 = xh + B;
      const den2 = yh + D;
      const q = (num1 - (num1 % den1)) / den1;
      if (q !== (num2 - (num2 % den2)) / den2) break;
      [A, C] = [C, A - q * C];
      [B, D] = [D, B - q * D];
      [xh, yh] = [yh, xh - q * yh];
    }
    if (B === 0) {
      [x, y] = [y, x % y];
    } else {
      const bx = BigInt(A) * x + BigInt(B) * y;
      const by = BigInt(C) * x + BigInt(D) * y;
      x = bx;
      y = by;
    }
  }
  return euclid(x, y);
};

export const lcm = (a: bigint, b: bigint): bigint => (a === 0n || b === 0n ? 0n : (a / gcd(a, b)) * b);

/** `num / den` in lowest terms; a zero denominator is a programming error. */
export function ratio(num: bigint, den: bigint = 1n): Ratio {
  if (den === 0n) throw new RangeError("a ratio needs a non-zero denominator");
  const sign = den < 0n ? -1n : 1n;
  const g = gcd(num, den);
  return g === 0n ? ZERO : { num: (sign * num) / g, den: (sign * den) / g };
}

export const ZERO: Ratio = { num: 0n, den: 1n };
export const ONE: Ratio = { num: 1n, den: 1n };

/**
 * a + b of two reduced ratios without a gcd of the products (Knuth 4.5.1): with d = gcd(a.den, b.den) the sum
 * a.num (b.den / d) + b.num (a.den / d) over (a.den / d) b.den is in lowest terms when d = 1, and otherwise
 * only the common part of the numerator and d is left to cancel. The result is the reduced fraction `ratio` would give.
 */
export const add = (a: Ratio, b: Ratio): Ratio => {
  if (a.num === 0n) return b;
  if (b.num === 0n) return a;
  const d = a.den === b.den ? a.den : gcd(a.den, b.den);
  if (d === 1n) return { num: a.num * b.den + b.num * a.den, den: a.den * b.den };
  const t = a.num * (b.den / d) + b.num * (a.den / d);
  if (t === 0n) return ZERO;
  const e = gcd(t, d);
  return { num: t / e, den: (a.den / d) * (b.den / e) };
};

/** a * b of two reduced ratios: only the cross factors can cancel (a.num with b.den, b.num with a.den). */
export const mul = (a: Ratio, b: Ratio): Ratio => {
  if (a.num === 0n || b.num === 0n) return ZERO;
  if (b.num === b.den) return a;
  if (a.num === a.den) return b;
  const g1 = gcd(a.num, b.den);
  const g2 = gcd(b.num, a.den);
  return { num: (a.num / g1) * (b.num / g2), den: (a.den / g2) * (b.den / g1) };
};

export const div = (a: Ratio, b: Ratio): Ratio => {
  if (b.num === 0n) throw new RangeError("a ratio needs a non-zero denominator");
  return mul(a, b.num < 0n ? { num: -b.den, den: -b.num } : { num: b.den, den: b.num });
};
export const isZero = (a: Ratio): boolean => a.num === 0n;

/** Strictly greater, by cross-multiplication (denominators are positive). */
export const greater = (a: Ratio, b: Ratio): boolean => a.num * b.den > b.num * a.den;

/** The sum of a list of ratios. */
export const sum = (values: readonly Ratio[]): Ratio => values.reduce(add, ZERO);

/** Scale a list of ratios by the least common multiple of their denominators: integers in the same proportion. */
export function integerise(values: readonly Ratio[]): { scale: bigint; weights: bigint[] } {
  const scale = values.reduce((l, v) => lcm(l, v.den), 1n);
  return { scale, weights: values.map((v) => v.num * (scale / v.den)) };
}
