// The cost measurement of the belief and the sampler (ADR-025 slice 2 part 4): wall-clock per decision (one belief) and per
// world (one `sample`), one thread, on the committed channels. Used by `scripts/measure.ts` (the journaled figures) and by
// `cost.test.ts` (a small run and the gate). Timings are measurements of one machine at one moment and are never compared
// for equality; the legality redraw count is exact and deterministic.

import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { BeliefError, type Sampler } from "../src/index.ts";
import { type Game, beliefAt, idsOf, points } from "./corpus.ts";

export interface Percentiles {
  readonly n: number;
  readonly median_ms: number;
  readonly p95_ms: number;
  readonly p99_ms: number;
  readonly max_ms: number;
}

/** Nearest-rank percentiles of a list of durations in milliseconds. */
export function percentiles(ms: readonly number[]): Percentiles {
  const sorted = [...ms].sort((a, b) => a - b);
  const at = (p: number): number => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))] as number;
  return { n: sorted.length, median_ms: at(0.5), p95_ms: at(0.95), p99_ms: at(0.99), max_ms: sorted[sorted.length - 1] as number };
}

const now = (): number => Number(process.hrtime.bigint()) / 1e6;

export interface Cost {
  readonly decision: Percentiles;
  /** The first world of a belief: it builds the belief's tables (item DP, move suffix tables). */
  readonly first_world: Percentiles;
  /** Every later world of the same belief. */
  readonly later_world: Percentiles;
  readonly worlds: number;
  readonly redraws: number;
  /** Legality redraws per world drawn, in ppm. */
  readonly redraw_ppm: number;
  readonly refused: number;
}

/**
 * Time a belief at every request of `list` and, at every `every`-th belief, `perBelief` worlds drawn one at a time.
 * `sampler` is bound to the legality port by the caller.
 */
export function measure(list: readonly Game[], prior: (ids: string[]) => UsagePrior, sampler: Sampler, options: { every?: number; perBelief?: number } = {}): Cost {
  const every = options.every ?? 4;
  const perBelief = options.perBelief ?? 5;
  const decision: number[] = [];
  const first: number[] = [];
  const later: number[] = [];
  let redraws = 0;
  let worlds = 0;
  let refused = 0;
  let k = 0;
  for (const game of list) {
    for (const point of points(game)) {
      const p = prior(idsOf(point));
      const t0 = now();
      let b;
      try {
        b = beliefAt(point, p);
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        refused += 1;
        continue;
      }
      decision.push(now() - t0);
      k += 1;
      if (k % every !== 0) continue;
      for (let w = 0; w < perBelief; w += 1) {
        const stream = Uint8Array.from({ length: 32 }, (_, i) => (k * 131 + w * 17 + i * 7) & 255);
        const t1 = now();
        const out = sampler.sample(b, stream);
        (w === 0 ? first : later).push(now() - t1);
        redraws += out.redraws;
        worlds += 1;
      }
    }
  }
  return { decision: percentiles(decision), first_world: percentiles(first), later_world: percentiles(later), worlds, redraws, redraw_ppm: worlds === 0 ? 0 : Math.floor((redraws * 1_000_000) / worlds), refused };
}
