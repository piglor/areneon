// Exact rational arithmetic on BigInt. The belief is computed with no floating point: every
// probability is a reduced `Ratio` and every normaliser an integer (ADR-025 section 3).

/** A reduced fraction with a positive denominator. */
export interface Ratio {
  readonly num: bigint;
  readonly den: bigint;
}

export const gcd = (a: bigint, b: bigint): bigint => {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) [x, y] = [y, x % y];
  return x;
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

export const add = (a: Ratio, b: Ratio): Ratio => ratio(a.num * b.den + b.num * a.den, a.den * b.den);
export const mul = (a: Ratio, b: Ratio): Ratio => ratio(a.num * b.num, a.den * b.den);
export const div = (a: Ratio, b: Ratio): Ratio => ratio(a.num * b.den, a.den * b.num);
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
