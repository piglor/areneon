// The quota plan: the D20 cell merge and the per-split quotas (ADR-013
// section 2, ADR-010 E1), decided before a single team is generated, from the
// counts of a recorded pilot. Everything here is exact integer arithmetic
// (BigInt where a product can pass 2^53), so a plan is the same on every
// platform.
//
// Cells and strata. A base cell is a label vector (labeller.ts). The merge
// coarsens a cell by dropping label dimensions one at a time, the last label
// of the label set first, so a coarsened group is a run of consecutive base
// cell numbers: dropping the last `level` digits of a cell number groups the
// cells that share the rest. A group is a stratum, numbered by its lowest cell.
//
// A group is affordable when
//   - its pilot hits are at least `pilot_min_hits` (the count rule), and
//   - the draws it is expected to need, quota * pilot_draws / pilot_hits, are
//     at most `d_max` (the cost rule),
// where its quota is the sum over splits of max(n_stratum_min, the split's
// size times the group's share of the pilot).
//
// The merge runs from the base cells up. Every unaffordable group is replaced
// by the group one level up, which takes in all the cells under it, affordable
// or not (a stratum is a partition: no team can be in two). This repeats a
// level at a time until every group is affordable. If even the single group of
// all cells is not, the plan is infeasible.

export class PlanError extends Error {
  override name = "PlanError";
}

export const SPLITS = ["dev", "validation", "training"] as const;
export type Split = (typeof SPLITS)[number];

/** The number of teams asked for in each split; a missing or zero split is not generated. */
export type SplitSizes = Partial<Record<Split, number>>;

export interface PlanParameters {
  /** The pilot's hit count of every base cell, in cell order. */
  cellHits: readonly number[];
  /** Radix of each label, first label first. */
  radices: readonly number[];
  pilotDraws: number;
  pilotMinHits: number;
  dMax: number;
  nStratumMin: number;
  sizes: SplitSizes;
  /** Groups that must be coarsened at least once more, as `level:group` keys (a re-plan after a shortfall). */
  forced?: ReadonlySet<string>;
}

export interface Stratum {
  /** The stratum number: its lowest base cell. */
  stratum: number;
  /** How many label digits were dropped: 0 is a base cell. */
  level: number;
  /** The base cells, `first` to `first + cells - 1`. */
  cells: number;
  pilot_hits: number;
  /** The quota in each generated split. */
  quotas: Partial<Record<Split, number>>;
  /** The sum of the quotas: the bin size. */
  total: number;
}

export interface QuotaPlan {
  splits: Split[];
  /** Sorted by stratum. */
  strata: Stratum[];
  /** The stratum of every base cell. */
  stratumOf: Uint16Array;
}

/** The key a re-plan uses to name a group that must be coarsened. */
export const groupKey = (level: number, stratum: number): string => `${level}:${stratum}`;

function checkCount(value: number, what: string, min: number): void {
  if (!Number.isSafeInteger(value) || value < min) throw new PlanError(`${what} must be an integer of at least ${min}`);
}

/** round(a * b / c) for non-negative integers, halves up, exact. */
const scaled = (a: number, b: number, c: number): number => Number((2n * BigInt(a) * BigInt(b) + BigInt(c)) / (2n * BigInt(c)));

export function planQuotas(p: PlanParameters): QuotaPlan {
  const cells = p.radices.reduce((n, r) => n * r, 1);
  if (p.cellHits.length !== cells) throw new PlanError(`${p.cellHits.length} cell counts for ${cells} cells`);
  checkCount(p.pilotDraws, "pilot_draws", 1);
  checkCount(p.pilotMinHits, "pilot_min_hits", 1);
  checkCount(p.dMax, "d_max", 1);
  checkCount(p.nStratumMin, "n_stratum_min", 1);
  const splits = SPLITS.filter((s) => (p.sizes[s] ?? 0) > 0);
  if (splits.length === 0) throw new PlanError("no split has a team count");
  for (const s of splits) checkCount(p.sizes[s] as number, `the size of ${s}`, 1);
  const total = p.cellHits.reduce((a, b) => a + b, 0);
  if (total > p.pilotDraws) throw new PlanError(`the cells hold ${total} hits from ${p.pilotDraws} draws`);

  /** Cells under a group of `level`: the product of the radices of the last `level` labels. */
  const width = (level: number): number => p.radices.slice(p.radices.length - level).reduce((n, r) => n * r, 1);
  const levels = new Uint8Array(cells);
  const prefix = new Array<number>(cells + 1).fill(0);
  p.cellHits.forEach((h, c) => (prefix[c + 1] = (prefix[c] as number) + h));
  const hitsIn = (first: number, count: number): number => (prefix[first + count] as number) - (prefix[first] as number);

  const quotasOf = (hits: number): Partial<Record<Split, number>> =>
    Object.fromEntries(splits.map((s) => [s, Math.max(p.nStratumMin, scaled(p.sizes[s] as number, hits, p.pilotDraws))]));
  const affordable = (hits: number, quotas: Partial<Record<Split, number>>): boolean => {
    if (hits < p.pilotMinHits) return false;
    const bin = Object.values(quotas).reduce((a, b) => a + (b as number), 0);
    return BigInt(bin) * BigInt(p.pilotDraws) <= BigInt(p.dMax) * BigInt(hits);
  };

  for (let level = 0; level <= p.radices.length; level++) {
    const span = width(level);
    for (let first = 0; first < cells; first += span) {
      if (levels[first] !== level) continue;
      const hits = hitsIn(first, span);
      if (affordable(hits, quotasOf(hits)) && !p.forced?.has(groupKey(level, first))) continue;
      if (level === p.radices.length) {
        throw new PlanError(`even all ${cells} cells together are not affordable: ${hits} pilot hits for d_max ${p.dMax}; raise d_max or the pilot, or lower n_stratum_min or pilot_min_hits`);
      }
      const parent = width(level + 1);
      const start = first - (first % parent);
      levels.fill(level + 1, start, start + parent);
    }
  }

  const strata: Stratum[] = [];
  const stratumOf = new Uint16Array(cells);
  for (let first = 0; first < cells; ) {
    const level = levels[first] as number;
    const span = width(level);
    const hits = hitsIn(first, span);
    const quotas = quotasOf(hits);
    strata.push({
      stratum: first,
      level,
      cells: span,
      pilot_hits: hits,
      quotas,
      total: Object.values(quotas).reduce((a, b) => a + (b as number), 0),
    });
    stratumOf.fill(first, first, first + span);
    first += span;
  }
  return { splits, strata, stratumOf };
}

export interface ManifestPlan {
  splits: { split: Split; teams: number; cells: { stratum: number; pilot_hits: number; quota: number }[] }[];
  merged_cells?: { stratum: number; cells: number[] }[];
}

/** The quota plan as the dataset manifest records it: per split, then the merged strata. */
export function manifestPlan(plan: QuotaPlan): ManifestPlan {
  const splits = plan.splits.map((split) => {
    const cells = plan.strata.map((s) => ({ stratum: s.stratum, pilot_hits: s.pilot_hits, quota: s.quotas[split] as number }));
    return { split, teams: cells.reduce((n, c) => n + c.quota, 0), cells };
  });
  const merged = plan.strata
    .filter((s) => s.level > 0)
    .map((s) => ({ stratum: s.stratum, cells: Array.from({ length: s.cells }, (_, k) => s.stratum + k) }));
  return merged.length === 0 ? { splits } : { splits, merged_cells: merged };
}
