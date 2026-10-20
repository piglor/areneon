// Range provisioning (ADR-013 section 2, "CI cost"). A bin of quota q for a
// stratum whose pilot rate is p = hits / pilot_draws fills when the q-th
// member of the stratum arrives, a negative-binomial number of draws. Planning
// uses a high quantile of that fill time rather than its mean, takes the
// largest over the strata as the planned stream length, adds a margin and
// rounds up to whole ranges of 2^range_draws_log2 draws.
//
// P(fill time <= t) = P(Binomial(t, p) >= q) = I_p(q, t - q + 1), the
// regularised incomplete beta function, evaluated by Lentz's continued
// fraction. This is a plan report, not an identity: it is rounded to whole
// draws and enters no hash.

import type { QuotaPlan } from "./plan.ts";

export class ProvisionError extends Error {
  override name = "ProvisionError";
}

/** ln Gamma(x) for x > 0 (Lanczos, g = 7). */
function lnGamma(x: number): number {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  const y = x - 1;
  let sum = c[0] as number;
  const t = y + 7.5;
  for (let k = 1; k < 9; k++) sum += (c[k] as number) / (y + k);
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** The continued fraction of the incomplete beta function (modified Lentz). */
export function betaFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  const guard = (v: number): number => (Math.abs(v) < tiny ? tiny : v);
  let c = 1;
  let d = 1 / guard(1 - ((a + b) * x) / (a + 1));
  let h = d;
  for (let m = 1; m <= 200_000; m++) {
    const m2 = 2 * m;
    const even = (m * (b - m) * x) / ((a - 1 + m2) * (a + m2));
    d = 1 / guard(1 + even * d);
    c = guard(1 + even / c);
    h *= d * c;
    const odd = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + 1 + m2));
    d = 1 / guard(1 + odd * d);
    c = guard(1 + odd / c);
    const change = d * c;
    h *= change;
    if (Math.abs(change - 1) < 1e-15) break;
  }
  return h;
}

/** The regularised incomplete beta function I_x(a, b). */
export function regularisedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (front * betaFraction(a, b, x)) / a : 1 - (front * betaFraction(b, a, 1 - x)) / b;
}

/** P(the q-th success comes on trial t or earlier) for success probability p. */
export const fillProbability = (q: number, p: number, t: number): number => (t < q ? 0 : regularisedBeta(p, q, t - q + 1));

/**
 * The smallest number of draws t with P(fill time <= t) >= quantile, for a bin
 * of `quota` filled at rate `hits` / `draws`. A rate of zero never fills.
 */
export function fillQuantile(quota: number, hits: number, draws: number, quantilePermille: number): number {
  if (hits <= 0) throw new ProvisionError("a stratum with no pilot hits never fills");
  if (quota < 1 || draws < 1 || hits > draws) throw new ProvisionError("quota, hits and draws are not a rate");
  const p = hits / draws;
  const target = quantilePermille / 1000;
  let high = Math.ceil(quota / p);
  while (fillProbability(quota, p, high) < target) high *= 2;
  let low = quota;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (fillProbability(quota, p, mid) >= target) high = mid;
    else low = mid + 1;
  }
  return low;
}

export interface Provision {
  /** The largest fill-time quantile over the strata. */
  planned_draws: number;
  /** `planned_draws` plus the margin. */
  provisioned_draws: number;
  /** Whole ranges of `range_draws` draws that cover `provisioned_draws`. */
  ranges: number;
  range_draws: number;
}

export interface ProvisionParameters {
  pilotDraws: number;
  fillQuantilePermille: number;
  rangeMarginPermille: number;
  rangeDrawsLog2: number;
}

export function provision(plan: QuotaPlan, p: ProvisionParameters): Provision {
  const planned = plan.strata.reduce((most, s) => Math.max(most, fillQuantile(s.total, s.pilot_hits, p.pilotDraws, p.fillQuantilePermille)), 0);
  const withMargin = Math.ceil((planned * (1000 + p.rangeMarginPermille)) / 1000);
  const rangeDraws = 2 ** p.rangeDrawsLog2;
  return { planned_draws: planned, provisioned_draws: withMargin, ranges: Math.ceil(withMargin / rangeDraws), range_draws: rangeDraws };
}
