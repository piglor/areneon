// The most likely joint (forme, item) worlds of a belief: a DIAGNOSTIC only (ADR-025 section 1,
// section 4), for reports and tests, never a consumer's input. In a joint space this large the K
// most likely worlds carry a negligible share of the mass, so an average over them is biased
// toward modal sets. A world here fixes each slot's forme and item (abilities, spreads and moves
// are not enumerated); its weight is the product of the slot cells under the Item Clause.

import { type Ratio, ratio } from "./ratio.ts";
import type { OpponentBelief } from "./types.ts";

export interface TopWorld {
  /** Per roster slot: the forme and the item (null: no item). */
  readonly slots: readonly { readonly forme: string; readonly item: string | null }[];
  /** The probability of the (forme, item) combination within the belief. */
  readonly p: Ratio;
}

interface Node {
  readonly bound: bigint;
  readonly weight: bigint;
  readonly chosen: readonly { readonly forme: string; readonly item: string | null }[];
}

/** The `k` most likely combinations, best first (ties in the order of the slots' cells), by best-first search with an exact bound. */
export function topWorlds(b: OpponentBelief, k: number): TopWorld[] {
  const best = b.slots.map((s) => s.cells.reduce((m, c) => (c.weight > m ? c.weight : m), 0n));
  const rest = (from: number): bigint => best.slice(from).reduce((p, w) => p * w, 1n);
  const queue: Node[] = [{ bound: rest(0), weight: 1n, chosen: [] }];
  const out: TopWorld[] = [];
  while (out.length < k && queue.length > 0) {
    queue.sort((x, y) => (x.bound === y.bound ? 0 : x.bound > y.bound ? -1 : 1));
    const node = queue.shift() as Node;
    const i = node.chosen.length;
    if (i === b.slots.length) {
      out.push({ slots: node.chosen, p: ratio(node.weight, b.itemClause.normaliser) });
      continue;
    }
    for (const c of (b.slots[i] as OpponentBelief["slots"][number]).cells) {
      if (b.itemClause.enforced && c.item !== null && node.chosen.some((o) => o.item === c.item)) continue;
      const weight = node.weight * c.weight;
      queue.push({ bound: weight * rest(i + 1), weight, chosen: [...node.chosen, { forme: c.forme, item: c.item }] });
    }
  }
  return out;
}

