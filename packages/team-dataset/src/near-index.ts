// The L2 distance between teams and the index that answers "is a candidate within d_team of any known
// team" (evaluation protocol L2, ADR-013). It lives here so the sealed generator and the public generator's
// evaluation-only exclusion and cross-split report share one definition; `sealed.ts` re-exports it unchanged.

import type { PokemonSet } from "@areneon/showdown-bridge";
import type { Split } from "./plan.ts";

export class SealedError extends Error {
  override name = "SealedError";
}

const fail = (message: string): never => {
  throw new SealedError(message);
};

const FIELDS = 6;
const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The six compared fields of one set. */
function setFields(set: PokemonSet): string[] {
  const evs = set.evs as unknown as Record<string, number | undefined>;
  return [toId(set.species), toId(set.item), toId(set.ability), toId(set.nature), set.moves.map(toId).sort().join(","), STATS.map((s) => evs[s] ?? 0).join(",")];
}

const setKey = (fields: readonly string[]): string => fields.join("|");
const MAX_TEAM = 12;

/**
 * The distance of two teams (L2): the least number of differing fields over
 * all pairings of their sets. Of unequal teams, the sets left unpaired count
 * as differing in every field.
 */
export function teamDistance(a: readonly PokemonSet[], b: readonly PokemonSet[]): number {
  return distanceOfFields(a.map(setFields), b.map(setFields));
}

function distanceOfFields(a: readonly string[][], b: readonly string[][]): number {
  const n = Math.max(a.length, b.length);
  if (n > MAX_TEAM) fail(`a team of ${n} sets is too large to match`);
  const cost = (i: number, j: number): number => {
    const x = a[i];
    const y = b[j];
    if (x === undefined || y === undefined) return FIELDS;
    return x.reduce((d, field, f) => d + (field === y[f] ? 0 : 1), 0);
  };
  // best[mask]: the least cost of pairing the first popcount(mask) sets of `a` with the sets of `b` in `mask`
  const best = new Array<number>(1 << n).fill(Infinity);
  best[0] = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    const i = popcount(mask);
    if (i >= n || best[mask] === Infinity) continue;
    for (let j = 0; j < n; j++) {
      if ((mask & (1 << j)) !== 0) continue;
      const next = mask | (1 << j);
      best[next] = Math.min(best[next] as number, (best[mask] as number) + cost(i, j));
    }
  }
  return best[(1 << n) - 1] as number;
}

function popcount(x: number): number {
  let n = 0;
  for (let v = x; v !== 0; v &= v - 1) n++;
  return n;
}

/**
 * The known teams, indexed to answer "is a candidate within `bound` of any of
 * them" without comparing every pair. A pairing with at most `bound` differing
 * fields leaves at most `bound` sets that are not identical, so a team that
 * shares fewer than (sets - bound) identical sets with the candidate is out; an
 * inverted index on whole sets finds the others, and only those are measured.
 * In memory: the scale-up to the shard store is a later slice.
 */
export class NearIndex {
  readonly #teams: string[][][] = [];
  readonly #postings = new Map<string, Map<number, number>>();
  readonly #bound: number;

  constructor(teams: Iterable<readonly PokemonSet[]>, bound: number) {
    if (!Number.isSafeInteger(bound) || bound < 0) fail("d_team must be a non-negative integer");
    this.#bound = bound;
    for (const team of teams) {
      const id = this.#teams.length;
      const fields = team.map(setFields);
      this.#teams.push(fields);
      for (const f of fields) {
        const key = setKey(f);
        const posting = this.#postings.get(key) ?? new Map<number, number>();
        posting.set(id, (posting.get(id) ?? 0) + 1);
        this.#postings.set(key, posting);
      }
    }
  }

  get size(): number {
    return this.#teams.length;
  }

  /** Whether some known team is within the bound of `team`. */
  near(team: readonly PokemonSet[]): boolean {
    const fields = team.map(setFields);
    const need = fields.length - this.#bound;
    if (need <= 0) return this.#teams.some((known) => distanceOfFields(fields, known) <= this.#bound);
    const wanted = new Map<string, number>();
    for (const f of fields) wanted.set(setKey(f), (wanted.get(setKey(f)) ?? 0) + 1);
    const shared = new Map<number, number>();
    for (const [key, count] of wanted) {
      for (const [id, have] of this.#postings.get(key) ?? []) shared.set(id, (shared.get(id) ?? 0) + Math.min(count, have));
    }
    for (const [id, count] of shared) {
      if (count >= need && distanceOfFields(fields, this.#teams[id] as string[][]) <= this.#bound) return true;
    }
    return false;
  }
}


// ---------------------------------------------------------------------------
// Cross-split near-duplicates (ADR-020 section 7): counts only.

export interface NearDuplicates {
  d_team: number;
  /** Teams of `of` within d_team of any team of `against`. */
  pairs: { of: Split; against: Split; teams: number; near: number }[];
}

/** The split pairs reported: a split is measured against the splits a policy could have trained on or tuned with. */
const NEAR_PAIRS = [
  ["validation", "dev"],
  ["validation", "training"],
  ["dev", "training"],
] as const;

/** For each split pair present, how many teams of the first are within `dTeam` of any team of the second. */
export function crossSplitNear(teams: Partial<Record<Split, readonly (readonly PokemonSet[])[]>>, dTeam: number): NearDuplicates {
  const indexes = new Map<Split, NearIndex>();
  const pairs: NearDuplicates["pairs"] = [];
  for (const [of, against] of NEAR_PAIRS) {
    const mine = teams[of];
    const theirs = teams[against];
    if (mine === undefined || theirs === undefined) continue;
    let index = indexes.get(against);
    if (index === undefined) {
      index = new NearIndex(theirs, dTeam);
      indexes.set(against, index);
    }
    const known = index;
    pairs.push({ of, against, teams: mine.length, near: mine.filter((team) => known.near(team)).length });
  }
  return { d_team: dTeam, pairs };
}
