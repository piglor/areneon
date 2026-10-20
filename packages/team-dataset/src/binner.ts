// Binning in counter order (ADR-013 section 2). Each accepted draw is offered
// in the order of its counter k. It is a duplicate when its canonical hash is
// known (an earlier binned draw of this run, or any earlier released dataset);
// otherwise it is binned if its stratum's bin is below quota; otherwise it is
// an overflow. A full bin ignores later candidates, so the bin holds the first
// `quota` distinct candidates of an independent stream that land in the
// stratum: an exact sample from the proposal conditioned on the stratum. The
// order of the checks (full bin first) is part of the definition.

import type { Accepted } from "./draw.ts";
import type { KnownHashes } from "./dedupe.ts";

export class BinError extends Error {
  override name = "BinError";
}

const fail = (message: string): never => {
  throw new BinError(message);
};

export type Offer = "binned" | "duplicate" | "overflow";

export class Binner {
  readonly #quotas: ReadonlyMap<number, number>;
  readonly #known: Pick<KnownHashes, "has">;
  readonly #bins = new Map<number, Accepted[]>();
  readonly #seen = new Set<string>();
  #open: number;

  /** `quotas` maps each stratum to its bin size; every stratum has a non-empty bin. */
  constructor(quotas: ReadonlyMap<number, number>, known: Pick<KnownHashes, "has">) {
    this.#quotas = quotas;
    this.#known = known;
    this.#open = quotas.size;
    for (const stratum of quotas.keys()) this.#bins.set(stratum, []);
  }

  /** Whether every bin is at quota: the stream can stop. */
  get full(): boolean {
    return this.#open === 0;
  }

  /** The strata whose bin is not at quota, ascending. */
  get short(): number[] {
    return [...this.#bins].filter(([stratum, bin]) => bin.length < (this.#quotas.get(stratum) as number)).map(([stratum]) => stratum).sort((a, b) => a - b);
  }

  /** Offer one accepted draw whose labels fall in `stratum`. */
  offer(draw: Accepted, stratum: number): Offer {
    const bin = this.#bins.get(stratum) ?? fail(`stratum ${stratum} has no bin`);
    if (bin.length >= (this.#quotas.get(stratum) as number)) return "overflow";
    if (this.#seen.has(draw.canonical_team_hash) || this.#known.has(draw.canonical_team_hash)) return "duplicate";
    this.#seen.add(draw.canonical_team_hash);
    bin.push(draw);
    if (bin.length === this.#quotas.get(stratum)) this.#open--;
    return "binned";
  }

  /** What each stratum's bin holds, in counter order. */
  bin(stratum: number): readonly Accepted[] {
    return this.#bins.get(stratum) ?? fail(`stratum ${stratum} has no bin`);
  }
}
