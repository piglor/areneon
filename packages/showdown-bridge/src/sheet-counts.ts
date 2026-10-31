// The counts of opponent sheet blocks (ADR-025 erratum E4 point 5), pure and engine-free so the record readers can use them: the
// blocks a builder built, how many were refused (the `unknown` block with a fixed code, never a rule fallback) and the refusals by
// code. Several builders' counts (a game's seats, a match's games) are summed with the refusals sorted by code.

export interface SheetCounts {
  /** Blocks built. */
  readonly requests: number;
  /** Blocks that were refused. */
  readonly refused: number;
  /** Refused blocks by their fixed code, sorted by code and unique. */
  readonly refusals: readonly { readonly code: string; readonly count: number }[];
}

/** Counts by code, sorted by code (codes are unique, so two are never equal). */
export const sortedByCode = (counts: ReadonlyMap<string, number>): { code: string; count: number }[] =>
  [...counts].sort(([a], [b]) => (a < b ? -1 : 1)).map(([code, count]) => ({ code, count }));

/** Counts of several builders (a game's seats) or games as one: requests and refusals summed, refusals sorted by code. */
export function sumSheetCounts(counts: readonly SheetCounts[]): { requests: number; refused: number; refusals: { code: string; count: number }[] } {
  const byCode = new Map<string, number>();
  for (const c of counts) for (const { code, count } of c.refusals) byCode.set(code, (byCode.get(code) ?? 0) + count);
  return { requests: counts.reduce((s, c) => s + c.requests, 0), refused: counts.reduce((s, c) => s + c.refused, 0), refusals: sortedByCode(byCode) };
}

/** The rule the profile cannot say about one counts object: refusals sorted by code and unique, adding up to `refused`, at most `requests`. */
export function sheetCountsProblem(c: SheetCounts): string | null {
  for (let i = 1; i < c.refusals.length; i++) {
    if ((c.refusals[i - 1] as { code: string }).code >= (c.refusals[i] as { code: string }).code) return "the refusals must be sorted by code and unique";
  }
  if (c.refused > c.requests) return "more blocks refused than built";
  if (c.refusals.reduce((s, r) => s + r.count, 0) !== c.refused) return "the refusals do not add up to refused";
  return null;
}

/** E4 point 5: a refused share above 1% of the requests invalidates the experiment (in parts per million: 10,000). */
export const MAX_REFUSED_PPM = 10_000;

/** The refused share of the requests in parts per million, rounded down; 0 for no request. */
export const refusedSharePpm = (counts: Pick<SheetCounts, "requests" | "refused">): number => (counts.requests === 0 ? 0 : Math.floor((counts.refused * 1_000_000) / counts.requests));
