// The arithmetic of the S0 measures (ADR-026 section 8, #1147 measures 3, 4 and 8; #1308). Pure and integer in, integer ppm out:
// no engine, battle, worker or oracle is read here, only the numbers a measurement tool hands over, so this module ships and
// is tested on hand-built cases. Where the oracle values come from (rollouts from a test-only fork) is the test tooling's business.
//
// Conventions. A "ppm" is parts per million. A correlation is rounded to the nearest ppm. Ties in a ranking get the average rank
// (Spearman). A decision whose values have no spread has no correlation (null), counted apart and never read as 0. A "best
// candidate" that is tied is credited by the expected hit under a uniform tie-break (1 / number tied, in ppm), as the prediction
// harness does. Intervals are the percentile bootstrap over decisions of the progress evaluation (`stats.pair_interval`):
// 4,000 resamples drawn from SHA-256(key || counter) as big-endian uint32 words (a word at or above the largest multiple of n is
// skipped), the 2.5% tails cut at ranks 100 and 3,899, the mean rounded down and the bounds outward.

import { createHash } from "node:crypto";
import type { Candidate } from "./choices.ts";
import { searchedIndices } from "./search-set.ts";
import type { SearchConfig } from "./search-config.ts";

const PPM = 1_000_000;
const B_BOOT = 4000;
const TAIL = 100;

/** Twice the average rank (1-based) of each value, so every rank is an integer. Equal values share the average of their positions. */
export function doubledRanks(values: readonly number[]): number[] {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v || a.i - b.i);
  const out = new Array<number>(values.length).fill(0);
  let start = 0;
  while (start < order.length) {
    let end = start;
    while (end + 1 < order.length && (order[end + 1] as { v: number }).v === (order[start] as { v: number }).v) end += 1;
    // positions start+1 .. end+1: twice their average is (start + 1) + (end + 1)
    for (let k = start; k <= end; k += 1) out[(order[k] as { i: number }).i] = start + end + 2;
    start = end + 1;
  }
  return out;
}

/** Spearman's rho of two equal-length lists in ppm (nearest), or null for fewer than two items or no spread in either list. */
export function spearmanPpm(a: readonly number[], b: readonly number[]): number | null {
  if (a.length !== b.length) throw new RangeError("the two lists need the same length");
  const n = a.length;
  if (n < 2) return null;
  const ra = doubledRanks(a);
  const rb = doubledRanks(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i += 1) {
    const x = (ra[i] as number) - (n + 1); // the doubled mean rank is n + 1
    const y = (rb[i] as number) - (n + 1);
    num += x * y;
    da += x * x;
    db += y * y;
  }
  if (da === 0 || db === 0) return null;
  return Math.round((num / Math.sqrt(da * db)) * PPM);
}

/** The Spearman-Brown correction of a half-to-half correlation to the full length: 2r / (1 + r), in ppm. Null when r is at or below -1. */
export function spearmanBrownPpm(rPpm: number): number | null {
  if (rPpm <= -PPM) return null;
  return Math.round((2 * rPpm * PPM) / (PPM + rPpm));
}

/** Mean of integers in ppm units, rounded down; null for an empty list. */
export const meanFloor = (values: readonly number[]): number | null => (values.length === 0 ? null : Math.floor(values.reduce((a, b) => a + b, 0) / values.length));

function draws(key: Buffer, n: number, count: number): number[] {
  const limit = Math.floor(2 ** 32 / n) * n;
  const out: number[] = [];
  for (let counter = 0; out.length < count; counter += 1) {
    const counterBytes = Buffer.alloc(8);
    counterBytes.writeBigUInt64BE(BigInt(counter));
    const block = createHash("sha256").update(key).update(counterBytes).digest();
    for (let i = 0; i < 32 && out.length < count; i += 4) {
      const word = block.readUInt32BE(i);
      if (word < limit) out.push(word % n);
    }
  }
  return out;
}

export interface Interval {
  mean_ppm: number;
  low_ppm: number;
  high_ppm: number;
  n: number;
}

/** Mean and 95% percentile-bootstrap interval of per-decision integer values (ppm units): the unit of resampling is the decision. */
export function bootstrapInterval(values: readonly number[], label: string): Interval {
  const n = values.length;
  if (n < 1) throw new RangeError("an interval needs at least one decision");
  const pick = draws(Buffer.from(label, "utf8"), n, n * B_BOOT);
  const sums: number[] = [];
  for (let b = 0; b < B_BOOT; b += 1) {
    let s = 0;
    for (let j = 0; j < n; j += 1) s += values[pick[b * n + j] as number] as number;
    sums.push(s);
  }
  sums.sort((x, y) => x - y);
  return {
    mean_ppm: Math.floor(values.reduce((a, b) => a + b, 0) / n),
    low_ppm: Math.floor((sums[TAIL] as number) / n),
    high_ppm: -Math.floor(-(sums[B_BOOT - 1 - TAIL] as number) / n),
    n,
  };
}

/** One decision's exhaustive re-rank: every non-pending candidate scored (`sum` over the scored worlds, so sums compare as means do). */
export interface ExhaustiveDecision {
  candidates: readonly Candidate[];
  prior_ppm: readonly number[];
  /** Scores by index into `candidates`, in any order. */
  scores: readonly { candidate: number; sum: number }[];
}

/** The index the exhaustive re-rank picks: the highest sum, the first in prior rank on a tie. */
export function exhaustiveBest(d: ExhaustiveDecision): number {
  const rank = new Map(searchedIndices({ top_k: d.candidates.length, top_k_rule: "prior-top-k" }, d.candidates, d.prior_ppm).map((index, r) => [index, r]));
  let best: { candidate: number; sum: number } | null = null;
  for (const s of d.scores) {
    if (best === null || s.sum > best.sum || (s.sum === best.sum && (rank.get(s.candidate) as number) < (rank.get(best.candidate) as number))) best = s;
  }
  if (best === null) throw new RangeError("a decision needs at least one score");
  return best.candidate;
}

export interface RecallRow {
  rule: SearchConfig["top_k_rule"];
  k: number;
  decisions: number;
  hits: number;
  recall_ppm: number | null;
}

/** Top-k recall (measure 4): the share of decisions whose exhaustive best candidate is in the set the rule searches. */
export function topKRecall(decisions: readonly ExhaustiveDecision[], rule: SearchConfig["top_k_rule"], k: number): RecallRow {
  let hits = 0;
  for (const d of decisions) {
    const set = searchedIndices({ top_k: k, top_k_rule: rule, ...(rule === "slot-cover" ? { k_max: k } : {}) }, d.candidates, d.prior_ppm);
    if (set.includes(exhaustiveBest(d))) hits += 1;
  }
  return { rule, k, decisions: decisions.length, hits, recall_ppm: decisions.length === 0 ? null : Math.floor((hits * PPM) / decisions.length) };
}

/** The indices that tie for the highest value. */
const argmaxSet = (values: readonly number[]): number[] => {
  const top = Math.max(...values);
  return values.flatMap((v, i) => (v === top ? [i] : []));
};

/** Expected top-1 hit in ppm of `chosen` against the best values: 1 / (number tied) if it is among the tied best, else 0. */
export function top1HitPpm(chosen: number, values: readonly number[]): number {
  const best = argmaxSet(values);
  return best.includes(chosen) ? Math.floor(PPM / best.length) : 0;
}

/** Agreement of two value lists on their best candidate under uniform tie-breaks: |A and B| / (|A| |B|) in ppm. */
export function top1AgreementPpm(a: readonly number[], b: readonly number[]): number {
  const [sa, sb] = [argmaxSet(a), argmaxSet(b)];
  const common = sa.filter((i) => sb.includes(i)).length;
  return Math.floor((common * PPM) / (sa.length * sb.length));
}

/** Per-candidate sums of rollout scores over a range of rollouts. `rollouts[c][r]` is candidate c's score in rollout r. */
export const rolloutSums = (rollouts: readonly (readonly number[])[], from = 0, to?: number): number[] => rollouts.map((row) => row.slice(from, to).reduce((a, b) => a + b, 0));

export interface SplitHalf {
  decisions: number;
  /** Decisions whose halves have a correlation (both halves have a spread), and those dropped from the mean for lack of spread. */
  defined: number;
  dropped: number;
  /** Mean over defined decisions of the Spearman between the half-sums, ppm. */
  half_correlation_ppm: number | null;
  /** The same corrected to the full 8 rollouts (Spearman-Brown), ppm: the noise ceiling on any attainable correlation. */
  corrected_ppm: number | null;
  /** Mean over decisions of the two halves' best-candidate agreement, ppm. */
  top1_agreement_ppm: number | null;
}

/** Split-half reliability of the oracle values (measure 8): the first half of the rollouts against the second. Each decision is a matrix [candidate][rollout]. */
export function splitHalf(decisions: readonly (readonly (readonly number[])[])[]): SplitHalf {
  const rhos: number[] = [];
  const agree: number[] = [];
  for (const matrix of decisions) {
    const width = (matrix[0] as readonly number[]).length;
    if (width < 2 || width % 2 !== 0) throw new RangeError("a split half needs an even number of rollouts, at least 2");
    const [a, b] = [rolloutSums(matrix, 0, width / 2), rolloutSums(matrix, width / 2)];
    const rho = spearmanPpm(a, b);
    if (rho !== null) rhos.push(rho);
    agree.push(top1AgreementPpm(a, b));
  }
  const mean = meanFloor(rhos);
  return { decisions: decisions.length, defined: rhos.length, dropped: decisions.length - rhos.length, half_correlation_ppm: mean, corrected_ppm: mean === null ? null : spearmanBrownPpm(mean), top1_agreement_ppm: meanFloor(agree) };
}

/** Area under the ROC curve in ppm (Mann-Whitney; a positive-negative pair with equal scores counts half). Null when a class is empty. */
export function aucPpm(scores: readonly number[], positive: readonly boolean[]): number | null {
  if (scores.length !== positive.length) throw new RangeError("scores and labels need the same length");
  let [wins2, pairs] = [0, 0];
  for (let i = 0; i < scores.length; i += 1) {
    if (!positive[i]) continue;
    for (let j = 0; j < scores.length; j += 1) {
      if (positive[j]) continue;
      pairs += 1;
      wins2 += (scores[i] as number) > (scores[j] as number) ? 2 : scores[i] === scores[j] ? 1 : 0;
    }
  }
  return pairs === 0 ? null : Math.floor((wins2 * PPM) / (2 * pairs));
}

export interface AucInterval {
  auc_ppm: number | null;
  games: number;
  decisions: number;
  /** Percentile bounds over the resamples of GAMES (all of a game's decisions together) that had both classes; null when none did. */
  low_ppm: number | null;
  high_ppm: number | null;
  resamples_used: number;
}

/**
 * The value head's AUC with a game-clustered bootstrap interval (S0 measure 3): decisions of one game share one result and are
 * correlated, so the unit of resampling is the game. A resample with no winner or no loser has no AUC and is skipped, counted.
 */
export function aucGameBootstrap(rows: readonly { game: string; score: number; positive: boolean }[], label: string): AucInterval {
  const byGame = new Map<string, { score: number; positive: boolean }[]>();
  for (const r of rows) byGame.set(r.game, [...(byGame.get(r.game) ?? []), { score: r.score, positive: r.positive }]);
  const games = [...byGame.keys()].sort().map((g) => byGame.get(g) as { score: number; positive: boolean }[]);
  const point = aucPpm(rows.map((r) => r.score), rows.map((r) => r.positive));
  if (games.length === 0) return { auc_ppm: null, games: 0, decisions: 0, low_ppm: null, high_ppm: null, resamples_used: 0 };
  const pick = draws(Buffer.from(label, "utf8"), games.length, games.length * 2000);
  const values: number[] = [];
  for (let b = 0; b < 2000; b += 1) {
    const sample = Array.from({ length: games.length }, (_, j) => games[pick[b * games.length + j] as number] as { score: number; positive: boolean }[]).flat();
    const auc = aucPpm(sample.map((x) => x.score), sample.map((x) => x.positive));
    if (auc !== null) values.push(auc);
  }
  values.sort((x, y) => x - y);
  const tail = Math.floor(values.length / 40);
  return { auc_ppm: point, games: games.length, decisions: rows.length, low_ppm: values.length === 0 ? null : (values[tail] as number), high_ppm: values.length === 0 ? null : (values[values.length - 1 - tail] as number), resamples_used: values.length };
}

export interface CalibrationBin {
  /** Lower edge of |predicted difference| in ppm (the next bin's edge is its upper bound, the last is open). */
  from_ppm: number;
  pairs: number;
  /** Mean predicted and mean oracle difference (ppm, signed so the predicted one is positive). */
  mean_predicted_ppm: number | null;
  mean_oracle_ppm: number | null;
  /** Pairs where the oracle difference has the predicted sign, among those where it is not zero; and the pairs where it is zero. */
  same_sign: number;
  oracle_zero: number;
}

/**
 * Calibration of value differences between siblings (measure 3): over every pair of candidates of a decision, the predicted
 * difference (search score or prior, in ppm units) against the oracle difference (the same sign convention: the candidate
 * predicted better first), binned by the predicted difference's size. `edges` ascend and start at 0.
 */
export function siblingCalibration(decisions: readonly { predicted: readonly number[]; oracle: readonly number[] }[], edges: readonly number[]): CalibrationBin[] {
  const bins = edges.map((from_ppm) => ({ from_ppm, pairs: 0, p: 0, o: 0, same: 0, zero: 0 }));
  for (const d of decisions) {
    for (let i = 0; i < d.predicted.length; i += 1) {
      for (let j = i + 1; j < d.predicted.length; j += 1) {
        let p = (d.predicted[i] as number) - (d.predicted[j] as number);
        let o = (d.oracle[i] as number) - (d.oracle[j] as number);
        if (p === 0) continue; // no predicted preference: nothing to calibrate
        if (p < 0) [p, o] = [-p, -o];
        let bin = bins[0] as (typeof bins)[number];
        for (const b of bins) if (p >= b.from_ppm) bin = b;
        bin.pairs += 1;
        bin.p += p;
        bin.o += o;
        if (o === 0) bin.zero += 1;
        else if (o > 0) bin.same += 1;
      }
    }
  }
  return bins.map((b) => ({ from_ppm: b.from_ppm, pairs: b.pairs, mean_predicted_ppm: b.pairs === 0 ? null : Math.floor(b.p / b.pairs), mean_oracle_ppm: b.pairs === 0 ? null : Math.floor(b.o / b.pairs), same_sign: b.same, oracle_zero: b.zero }));
}

export interface LeafQuality {
  /** Decisions that entered, those with a correlation for both the search and the prior, and the search's top-1 against the oracle. */
  decisions: number;
  paired: number;
  /** Decisions left out of the paired set because the search, the prior or the oracle values have no spread (no correlation). */
  dropped: number;
  search_spearman_ppm: number | null;
  prior_spearman_ppm: number | null;
  /** The same two means over the paired set only, the set the difference is taken on. */
  search_spearman_paired_ppm: number | null;
  prior_spearman_paired_ppm: number | null;
  /** Per-decision (search minus prior) with its bootstrap interval; the go rule reads the lower end. */
  difference: Interval | null;
  search_top1_ppm: number | null;
  prior_top1_ppm: number | null;
}

/**
 * Leaf quality within a decision (measure 3 and the top-1 of measure 8). Per decision, over the searched candidates:
 * `search` the weighted sums, `prior` the prior's ppm, `oracle` the oracle sums. The paired difference is taken on decisions where
 * both correlations exist.
 */
export function leafQuality(decisions: readonly { search: readonly number[]; prior: readonly number[]; oracle: readonly number[] }[], label: string): LeafQuality {
  const [s, p, diffs, st, pt, sp, pp] = [[] as number[], [] as number[], [] as number[], [] as number[], [] as number[], [] as number[], [] as number[]];
  for (const d of decisions) {
    const [rs, rp] = [spearmanPpm(d.search, d.oracle), spearmanPpm(d.prior, d.oracle)];
    if (rs !== null) s.push(rs);
    if (rp !== null) p.push(rp);
    if (rs !== null && rp !== null) {
      diffs.push(rs - rp);
      sp.push(rs);
      pp.push(rp);
    }
    st.push(top1HitPpm(argmaxSet(d.search)[0] as number, d.oracle));
    pt.push(top1HitPpm(argmaxSet(d.prior)[0] as number, d.oracle));
  }
  return { decisions: decisions.length, paired: diffs.length, dropped: decisions.length - diffs.length, search_spearman_ppm: meanFloor(s), prior_spearman_ppm: meanFloor(p), search_spearman_paired_ppm: meanFloor(sp), prior_spearman_paired_ppm: meanFloor(pp), difference: diffs.length === 0 ? null : bootstrapInterval(diffs, label), search_top1_ppm: meanFloor(st), prior_top1_ppm: meanFloor(pt) };
}
