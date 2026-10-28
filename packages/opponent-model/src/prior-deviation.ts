// How far the move model's conditional Poisson design is from the marginals a usage prior states (ADR-025
// section 3: "the design's inclusion probabilities are close to, not equal to, the source marginals").
// A report of the derivation (#1120), not a consumer input: for each forme, the design is built with no
// revealed move and each named move's design inclusion is compared with its stated inclusion / denominator.
// Exact integers; the deviation is reported in parts per million, rounded down.

import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { moveModel } from "./moves.ts";

const PPM = 1_000_000n;

export interface FormeDeviation {
  readonly species: string;
  readonly forme: string;
  /** The largest |design - stated| over the forme's named moves, in parts per million. */
  readonly max_ppm: number;
  /** The move where it is largest (the first by id on a tie). */
  readonly move: string | null;
}

/** The largest deviation per forme; `pools` maps a forme id to its roster move pool. */
export function designDeviation(prior: UsagePrior, pools: ReadonlyMap<string, readonly string[]>): FormeDeviation[] {
  const out: FormeDeviation[] = [];
  for (const species of prior.species) {
    for (const forme of species.formes) {
      const pool = pools.get(forme.id);
      if (pool === undefined) throw new RangeError(`forme ${forme.id} has no move pool`);
      const result = moveModel(forme.moves, pool, []);
      if (result === null) throw new RangeError(`forme ${forme.id}: the move model refuses the empty evidence`);
      const design = new Map(result.model.inclusion.map((m) => [m.id, m.p]));
      let worst = -1n;
      let move: string | null = null;
      for (const entry of forme.moves.entries) {
        const p = design.get(entry.id) ?? { num: 0n, den: 1n };
        const D = BigInt(forme.moves.denominator);
        // |p.num / p.den - inclusion / D| in ppm, by cross multiplication.
        const left = p.num * D;
        const right = BigInt(entry.inclusion) * p.den;
        const gap = left > right ? left - right : right - left;
        const ppm = (gap * PPM) / (p.den * D);
        if (ppm > worst) {
          worst = ppm;
          move = entry.id;
        }
      }
      out.push({ species: species.id, forme: forme.id, max_ppm: worst < 0n ? 0 : Number(worst), move });
    }
  }
  return out;
}
