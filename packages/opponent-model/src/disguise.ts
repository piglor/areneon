// Disguise assignments (ADR-025 section 3, review finding N-B). While a preview species that can hide
// its identity is unsettled, a Pokemon shown on a position may be that member in disguise. A
// disguise assignment maps each shown position to the slot its evidence belongs to: the position's
// own slot, or an unsettled identity-hiding slot, each hiding slot used by at most one position
// (an injective map, a matching of positions into slots). Hall's condition decides whether any
// assignment exists. This module is the combinatorics only: which assignments exist, and refusing
// above the reviewed cap. The evidence per assignment is the belief's.

/** For each shown position (in order), the slots its evidence may be attributed to, own slot first. */
export type Options = readonly (readonly number[])[];

/**
 * Whether every position can be given a slot of its own list with no slot used twice (a complete
 * matching; Kuhn's augmenting paths). By Hall's theorem this holds exactly when every set of
 * positions has at least as many distinct slots among its options as positions.
 */
export function hasMatching(options: Options): boolean {
  const owner = new Map<number, number>();
  const place = (position: number, seen: Set<number>): boolean => {
    for (const slot of options[position] as readonly number[]) {
      if (seen.has(slot)) continue;
      seen.add(slot);
      const holder = owner.get(slot);
      if (holder === undefined || place(holder, seen)) {
        owner.set(slot, position);
        return true;
      }
    }
    return false;
  };
  return options.every((_, position) => place(position, new Set()));
}

/**
 * Every complete assignment, in the order of the options (the first option of every position first),
 * where no slot of `hiding` is used twice. A position's own slot is its own and never clashes.
 * Returns null as soon as there are more than `limit` of them.
 */
export function enumerate(options: Options, hiding: ReadonlySet<number>, limit: number): number[][] | null {
  const out: number[][] = [];
  const used = new Set<number>();
  const chosen: number[] = [];
  let exceeded = false;
  const walk = (position: number): void => {
    if (exceeded) return;
    if (position === options.length) {
      if (out.length >= limit) exceeded = true;
      else out.push([...chosen]);
      return;
    }
    for (const slot of options[position] as readonly number[]) {
      if (hiding.has(slot) && used.has(slot)) continue;
      chosen.push(slot);
      used.add(slot);
      walk(position + 1);
      used.delete(slot);
      chosen.pop();
    }
  };
  walk(0);
  return exceeded ? null : out;
}
