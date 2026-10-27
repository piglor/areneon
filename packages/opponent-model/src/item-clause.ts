// The Item Clause joint (ADR-025 section 3, review finding N2): the slots couple only through the
// rule that no two of them hold the same item, and "no item" is exempt (several slots may hold
// none). With integer weights a[i][x] for slot i holding item x (everything else about the slot
// already summed in), the joint weight of an assignment is the product of the a[i][x_i] over
// assignments in which no item appears twice. An exact DP over the items in order, with the
// subset of slots that already hold an item as its state (2^N subsets, 64 for six slots),
// computes the normaliser; a forward and a backward table give every marginal. The DP decides
// only which slots hold an item: every slot outside that subset takes its "no item" weight.

export interface ClauseWeights {
  /** weights[i][j]: the weight of slot i holding item j (items in a fixed order, the same for every slot). */
  readonly items: readonly (readonly bigint[])[];
  /** none[i]: the weight of slot i holding no item. */
  readonly none: readonly bigint[];
}

export interface ClauseResult {
  /** The sum of the joint weights of all allowed assignments. */
  readonly normaliser: bigint;
  /** items[i][j]: the joint weight of the assignments in which slot i holds item j; divide by the normaliser. */
  readonly items: readonly (readonly bigint[])[];
  /** none[i]: the same for slot i holding no item. */
  readonly none: readonly bigint[];
}

/** Without the clause the slots are independent: the normaliser is the product of the slot totals. */
function independent(w: ClauseWeights): ClauseResult {
  const totals = w.none.map((n, i) => (w.items[i] as readonly bigint[]).reduce((s, x) => s + x, n));
  const normaliser = totals.reduce((p, t) => p * t, 1n);
  const others = (i: number): bigint => totals.reduce((p, t, l) => (l === i ? p : p * t), 1n);
  return {
    normaliser,
    items: w.items.map((row, i) => row.map((x) => x * others(i))),
    none: w.none.map((n, i) => n * others(i)),
  };
}

/** The most slots the subset DP is built for (2^N states); the format has six. */
export const MAX_CLAUSE_SLOTS = 10;

/** The DP tables of one set of weights (the sampler walks `backward` to draw an assignment). */
export interface ClauseTables {
  readonly slots: number;
  /** forward[k][mask]: the weight of giving items 0..k-1 to the slots of mask, one item each. */
  readonly forward: readonly (readonly bigint[])[];
  /** backward[k][open]: the weight of giving items k..m-1 to some of the open slots, the rest taking no item. */
  readonly backward: readonly (readonly bigint[])[];
}

export function clauseTables(w: ClauseWeights): ClauseTables {
  const slots = w.none.length;
  if (slots > MAX_CLAUSE_SLOTS) throw new RangeError(`the Item Clause DP is built for at most ${MAX_CLAUSE_SLOTS} slots`);
  const m = (w.items[0] as readonly bigint[] | undefined)?.length ?? 0;
  const size = 1 << slots;
  const at = (i: number, j: number): bigint => (w.items[i] as readonly bigint[])[j] as bigint;

  // forward[k][mask]: the weight of giving items 0..k-1 to the slots of mask, one item each.
  const forward: bigint[][] = [Array.from({ length: size }, (_, mask) => (mask === 0 ? 1n : 0n))];
  for (let k = 0; k < m; k += 1) {
    const before = forward[k] as bigint[];
    const next = before.slice();
    for (let mask = 1; mask < size; mask += 1) {
      for (let i = 0; i < slots; i += 1) if (mask & (1 << i)) next[mask] = (next[mask] as bigint) + (before[mask & ~(1 << i)] as bigint) * at(i, k);
    }
    forward.push(next);
  }
  // backward[k][open]: the weight of giving items k..m-1 to some of the open slots, the rest taking no item.
  const backward: bigint[][] = Array.from({ length: m + 1 }, () => []);
  backward[m] = Array.from({ length: size }, (_, open) => {
    let product = 1n;
    for (let i = 0; i < slots; i += 1) if (open & (1 << i)) product *= w.none[i] as bigint;
    return product;
  });
  for (let k = m - 1; k >= 0; k -= 1) {
    const after = backward[k + 1] as bigint[];
    const row = after.slice();
    for (let open = 1; open < size; open += 1) {
      for (let i = 0; i < slots; i += 1) if (open & (1 << i)) row[open] = (row[open] as bigint) + (after[open & ~(1 << i)] as bigint) * at(i, k);
    }
    backward[k] = row;
  }
  return { slots, forward, backward };
}

export function solveItemClause(w: ClauseWeights, enforce: boolean): ClauseResult {
  if (!enforce) return independent(w);
  const { slots, forward, backward } = clauseTables(w);
  const m = (w.items[0] as readonly bigint[] | undefined)?.length ?? 0;
  const size = 1 << slots;
  const full = size - 1;
  const at = (i: number, j: number): bigint => (w.items[i] as readonly bigint[])[j] as bigint;
  const normaliser = (backward[0] as bigint[])[full] as bigint;

  const items = Array.from({ length: slots }, (_, i) =>
    Array.from({ length: m }, (_, k) => {
      let total = 0n;
      const after = backward[k + 1] as bigint[];
      const before = forward[k] as bigint[];
      for (let mask = 0; mask < size; mask += 1) {
        if (mask & (1 << i)) continue;
        total += (before[mask] as bigint) * at(i, k) * (after[full & ~mask & ~(1 << i)] as bigint);
      }
      return total;
    }),
  );
  const none = items.map((row) => row.reduce((rest, x) => rest - x, normaliser));
  return { normaliser, items, none };
}
