// The bounded redraw interface (ADR-026 Evidence 1, V2 as evidence, N6): a world that fails a check because
// of the opponent's unrevealed sets (a revealed fact it contradicts, an Item Clause break, a `maybeTrapped`
// request nothing in the world explains, a condition that has run out under the sampled item) is not a
// defect; redrawing the opponent's sets until the evidence holds is exactly conditioning the belief on the
// own request. The builder reports pass or the failing check and class and whether a redraw can fix it; this
// module is the loop around it with the bound R and the counts by check and class. It draws nothing itself:
// `draw` hands over the world input of one attempt (slice B makes it from the sampler's seeded stream), so the
// loop is a pure function of the builder, the draws and the bound.

import type { WorldBuilder, WorldInput, WorldRefusal, WorldResult } from "./world-builder.ts";

export interface RedrawCounts {
  /** Worlds asked for (one `build` call each). */
  worlds: number;
  /** Builder calls made, the first draws included. */
  attempts: number;
  /** Draws after a refusal a redraw could fix. */
  redraws: number;
  /** Worlds still refused after the bound R. */
  exhausted: number;
  /** Worlds refused for a reason a redraw cannot fix. */
  final_refusals: number;
  /** Refusals that triggered a redraw, by check and class as `V2:-`. */
  redrawn: Record<string, number>;
}

export interface Redrawer {
  /**
   * Ask `draw(attempt)` for a world input (attempt 0 first) and build it; after a refusal that `redraw` marks
   * fixable, draw again, at most `max_redraws` more times. Returns the world, or the last refusal with the
   * number of draws used.
   */
  build(draw: (attempt: number) => WorldInput): { result: WorldResult; attempts: number; exhausted: boolean };
  counts(): RedrawCounts;
}

const key = (refusal: WorldRefusal): string => `${refusal.check ?? "-"}:${refusal.state_class ?? "-"}`;

export function createRedrawer(builder: WorldBuilder, max_redraws: number): Redrawer {
  if (!Number.isInteger(max_redraws) || max_redraws < 0) throw new RangeError("max_redraws is a non-negative integer");
  const counts: RedrawCounts = { worlds: 0, attempts: 0, redraws: 0, exhausted: 0, final_refusals: 0, redrawn: {} };
  return {
    build(draw) {
      counts.worlds += 1;
      for (let attempt = 0; ; attempt += 1) {
        counts.attempts += 1;
        const result = builder.build(draw(attempt));
        if (result.kind === "world") return { result, attempts: attempt + 1, exhausted: false };
        if (!result.refusal.redraw) {
          counts.final_refusals += 1;
          return { result, attempts: attempt + 1, exhausted: false };
        }
        if (attempt >= max_redraws) {
          counts.exhausted += 1;
          return { result, attempts: attempt + 1, exhausted: true };
        }
        counts.redraws += 1;
        counts.redrawn[key(result.refusal)] = (counts.redrawn[key(result.refusal)] ?? 0) + 1;
      }
    },
    counts: () => structuredClone(counts),
  };
}
