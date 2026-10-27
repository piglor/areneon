// The most likely joint (forme, item) worlds of a belief: a DIAGNOSTIC only (ADR-025 section 1,
// section 4), for reports and tests, never a consumer's input. In a joint space this large the K
// most likely worlds carry a negligible share of the mass, so an average over them is biased
// toward modal sets. A world here fixes each slot's forme and item (abilities, spreads and moves
// are not enumerated); its weight is the product of the slot cells under the Item Clause.

import { type Ratio, greater, ratio } from "./ratio.ts";
import type { OpponentBelief } from "./types.ts";

export interface TopWorld {
  /** Per roster slot: the forme and the item (null: no item). */
  readonly slots: readonly { readonly forme: string; readonly item: string | null }[];
  /** The probability of the (forme, item) combination in this assignment within the belief (the assignment's share included); the same combination can appear once per assignment. */
  readonly p: Ratio;
  /** The index in `belief.assignments` of the disguise assignment the combination is in. */
  readonly assignment: number;
}

interface Node {
  readonly assignment: number;
  /** An upper bound on `p` for every completion of the node. */
  readonly bound: Ratio;
  readonly weight: bigint;
  readonly chosen: readonly { readonly forme: string; readonly item: string | null }[];
}

/** The `k` most likely combinations, best first (ties in the order of the assignments and the slots' cells), by best-first search with an exact bound. */
export function topWorlds(b: OpponentBelief, k: number): TopWorld[] {
  const total = b.disguise.total;
  const best = b.assignments.map((a) => a.slots.map((s) => s.cells.reduce((m, c) => (c.weight > m ? c.weight : m), 0n)));
  const rest = (a: number, from: number): bigint => (best[a] as bigint[]).slice(from).reduce((p, w) => p * w, 1n);
  const share = (a: number, weight: bigint): Ratio => ratio((b.assignments[a]?.weight as bigint) * weight, total * (b.assignments[a]?.normaliser as bigint));
  const queue: Node[] = b.assignments.map((_, a) => ({ assignment: a, bound: share(a, rest(a, 0)), weight: 1n, chosen: [] }));
  const out: TopWorld[] = [];
  while (out.length < k && queue.length > 0) {
    queue.sort((x, y) => (greater(x.bound, y.bound) ? -1 : greater(y.bound, x.bound) ? 1 : 0));
    const node = queue.shift() as Node;
    const slots = (b.assignments[node.assignment] as OpponentBelief["assignments"][number]).slots;
    const i = node.chosen.length;
    if (i === slots.length) {
      out.push({ slots: node.chosen, p: share(node.assignment, node.weight), assignment: node.assignment });
      continue;
    }
    for (const c of (slots[i] as OpponentBelief["assignments"][number]["slots"][number]).cells) {
      if (b.itemClause.enforced && c.item !== null && node.chosen.some((o) => o.item === c.item)) continue;
      const weight = node.weight * c.weight;
      queue.push({ assignment: node.assignment, bound: share(node.assignment, weight * rest(node.assignment, i + 1)), weight, chosen: [...node.chosen, { forme: c.forme, item: c.item }] });
    }
  }
  return out;
}
