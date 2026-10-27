// The move model of one forme (ADR-025 section 3): conditional Poisson with exact rational odds.
//
// A Pokemon carries k moves. Forced moves (inclusion 1) fill slots first; the other moves are drawn
// by the conditional Poisson design, P(S) proportional to the product of the odds w = p / (1 - p)
// over the subsets S of the remaining candidates of a fixed size. The normalising constants are the
// elementary symmetric polynomials of the odds, computed by an exact integer DP (the odds are
// scaled to integers by their common denominator, which cancels in every ratio). Conditioning on
// revealed moves is the same DP over the remaining candidates and slots. The count k comes from
// the empty-slot mass by the two-point rule: the mean e of the empty slots is the only thing the
// data gives, so k is `slots - floor(e) - 1` with probability `e - floor(e)` and `slots - floor(e)`
// otherwise (a stated approximation), never below the forced moves, never above the number of moves that
// can be drawn, and, given revealed moves, conditioned on holding them (a count below the revealed
// moves is impossible; if every count is, the count is the number of revealed and forced moves). The design's inclusion probabilities are close to, not
// equal to, the source marginals.

import { type Ratio, ONE, ZERO, add, div, integerise, isZero, mul, ratio } from "./ratio.ts";

/** The part of a prior move list the model reads. */
export interface MoveListInput {
  readonly denominator: number;
  readonly slots: number;
  readonly entries: readonly { readonly id: string; readonly inclusion: number }[];
  readonly empty_slots: number;
  readonly residual: number;
}

export interface MoveOutcome {
  /** The number of moves the set holds (forced and revealed ones included). */
  readonly size: number;
  /** The posterior probability of that count, given the revealed moves. */
  readonly weight: Ratio;
}

export interface MoveModel {
  /** Moves every set holds (inclusion 1), sorted. */
  readonly forced: readonly string[];
  /** Revealed moves that are not forced, sorted. */
  readonly revealed: readonly string[];
  /** The candidates the design draws the remaining slots from, sorted, with their integer odds. */
  readonly candidates: readonly { readonly id: string; readonly odds: bigint }[];
  readonly outcomes: readonly MoveOutcome[];
  /** P(move in the set) for every move with a positive probability, sorted by id. */
  readonly inclusion: readonly { readonly id: string; readonly p: Ratio }[];
}

/** e_0 .. e_degree of the weights: e_m is the sum over m-subsets of the product of their weights. */
export function elementary(weights: readonly bigint[], degree: number): bigint[] {
  const e: bigint[] = [1n, ...Array.from({ length: degree }, () => 0n)];
  for (const w of weights) for (let m = degree; m >= 1; m -= 1) e[m] = (e[m] as bigint) + w * (e[m - 1] as bigint);
  return e;
}

/** e_0 .. e_degree of the weights without one of them (weight `w`), by the exact recurrence e_m - w e'_(m-1). */
export function elementaryWithout(e: readonly bigint[], w: bigint, degree: number): bigint[] {
  const out: bigint[] = [1n];
  for (let m = 1; m <= degree; m += 1) out[m] = (e[m] as bigint) - w * (out[m - 1] as bigint);
  return out;
}

const compareIds = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** One candidate move: its inclusion probability a / b. */
interface Inclusion {
  readonly id: string;
  readonly a: bigint;
  readonly b: bigint;
}

/** The inclusion of every pool move: named moves as given, the residual spread equally over the unnamed ones. */
function inclusions(list: MoveListInput, pool: readonly string[]): Inclusion[] {
  const named = new Map(list.entries.map((e) => [e.id, BigInt(e.inclusion)]));
  const D = BigInt(list.denominator);
  const unnamed = pool.filter((id) => !named.has(id)).length;
  return [...pool].sort(compareIds).map((id) => {
    const a = named.get(id);
    return a === undefined ? { id, a: BigInt(list.residual), b: BigInt(unnamed) * D } : { id, a, b: D };
  });
}

/**
 * The model of one forme given its revealed moves, or null when the revealed moves cannot all be in
 * a set (more than `slots`, or a move with no probability at all). `revealed` holds pool moves only.
 * The likelihood is P(all revealed moves are in the set | forme).
 */
export function moveModel(list: MoveListInput, pool: readonly string[], revealed: readonly string[]): { likelihood: Ratio; model: MoveModel } | null {
  const all = inclusions(list, pool);
  const forced = all.filter((m) => m.a >= m.b && m.b > 0n).map((m) => m.id);
  const forcedSet = new Set(forced);
  // Odds a / (b - a) as fractions, scaled to integers by their common denominator.
  const open = all.filter((m) => !forcedSet.has(m.id) && m.a > 0n);
  const odds = open.map((m) => ratio(m.a, m.b - m.a));
  const { weights } = integerise(odds);
  const weightOf = new Map(open.map((m, k) => [m.id, weights[k] as bigint]));

  const revealedSet = [...new Set(revealed)].sort(compareIds);
  const targets = revealedSet.filter((id) => !forcedSet.has(id));
  if (targets.some((id) => !weightOf.has(id))) return null;
  const minimum = forced.length + targets.length;
  if (minimum > list.slots) return null;
  const highest = Math.min(list.slots, forced.length + open.length);

  // The two-point count: the mean number of empty slots is empty_slots / denominator.
  const D = BigInt(list.denominator);
  const empty = BigInt(list.empty_slots);
  const whole = Number(empty / D);
  const lower = ratio(empty % D, D);
  const counts: { size: number; p: Ratio }[] = [];
  if (!isZero(lower)) counts.push({ size: list.slots - whole - 1, p: lower });
  counts.push({ size: list.slots - whole, p: isZero(lower) ? ONE : add(ONE, mul(lower, ratio(-1n))) });

  const rest = open.filter((m) => !targets.includes(m.id));
  const restWeights = rest.map((m) => weightOf.get(m.id) as bigint);
  const targetWeight = targets.reduce((product, id) => product * (weightOf.get(id) as bigint), 1n);
  // A count below the revealed moves is impossible given them (conditioned away); when every count is,
  // the Pokemon has shown more moves than the data's mean allows and the count is the minimum (never below).
  const sizes = new Map<number, Ratio>();
  for (const c of counts) {
    const size = Math.min(Math.max(c.size, forced.length), highest);
    if (size >= minimum) sizes.set(size, add(sizes.get(size) ?? ZERO, c.p));
  }
  if (sizes.size === 0) sizes.set(minimum, ONE);

  // For each distinct size: P(targets in the set | size) and the inclusion of every other candidate given it.
  const full = elementary(open.map((m) => weightOf.get(m.id) as bigint), highest - forced.length);
  const draws = new Map<number, { weight: Ratio; chosen: Ratio[] }>();
  for (const [size, prior] of sizes) {
    const r = size - forced.length;
    const t = targets.length;
    const eRest = elementary(restWeights, r - t);
    const given = ratio(targetWeight * (eRest[r - t] as bigint), full[r] as bigint);
    const chosen = rest.map((_, k) => {
      if (r - t === 0) return ZERO;
      const without = elementaryWithout(eRest, restWeights[k] as bigint, r - t - 1);
      return ratio((restWeights[k] as bigint) * (without[r - t - 1] as bigint), eRest[r - t] as bigint);
    });
    draws.set(size, { weight: mul(prior, given), chosen });
  }
  const likelihood = [...draws.values()].reduce((s, d) => add(s, d.weight), ZERO);

  const outcomes: MoveOutcome[] = [...draws].map(([size, d]) => ({ size, weight: div(d.weight, likelihood) })).sort((a, b) => a.size - b.size);
  const inclusion: { id: string; p: Ratio }[] = [];
  for (const id of new Set([...forced, ...targets])) inclusion.push({ id, p: ONE });
  rest.forEach((m, k) => {
    const p = [...draws.values()].reduce((s, d) => add(s, mul(div(d.weight, likelihood), d.chosen[k] as Ratio)), ZERO);
    if (!isZero(p)) inclusion.push({ id: m.id, p });
  });
  inclusion.sort((a, b) => compareIds(a.id, b.id));
  return {
    likelihood,
    model: {
      forced,
      revealed: targets,
      candidates: open.map((m) => ({ id: m.id, odds: weightOf.get(m.id) as bigint })),
      outcomes,
      inclusion,
    },
  };
}
