// Counts of one corpus: per entity (present, exercised, equal, differing, the first difference that named it), per pair
// (present together, exercised together, equal), and the differences by reason and first-differing field. A tally is all the
// ledger and the report need of a corpus; items are added one at a time, so a corpus never has to be held in memory.
import type { Json } from "./canonical.ts";
import type { Difference, ItemResult } from "./compare.ts";
import { sha256Jcs } from "./hash.ts";

export type CorpusKind = "fresh" | "regression" | "shadow";

/** A corpus as the ledger names it. */
export interface CorpusSpec {
  id: string;
  kind: CorpusKind;
  /** Fresh: the backend's fresh seed; otherwise the hash naming the corpus's source (64 hex). */
  seed: string;
  /** The identity of what the items were drawn from (64 hex): the generator pool of a fresh corpus, the files of a committed one. */
  pool: string;
}

export interface EntityTally {
  present: number;
  exercised: number;
  equal: number;
  differing: number;
  /** The position of the first item with a difference that named the entity, or null. */
  first_difference: number | null;
}

export interface PairTally {
  co_present: number;
  co_exercised: number;
  co_equal: number;
}

export interface CorpusTally {
  spec: CorpusSpec;
  items: number;
  entities: Map<string, EntityTally>;
  /** Keyed `a|b` with `a < b`; kept for a fresh corpus only. */
  pairs: Map<string, PairTally>;
  /** Every difference, in item order (fixed words: reason, field, entities, position). */
  differences: Difference[];
  /** The batches the backend returned another number of outcomes than inputs for. */
  count_mismatches: number;
  /** The items' stable keys, in item order (the content hash is over them, sorted). */
  keys: string[];
  /** The distinct keys seen. A fresh corpus refuses a repeated one (see `addItem`). */
  seen: Set<string>;
}

const ID = /^[a-z0-9][a-z0-9._-]*$/;
const HEX64 = /^[0-9a-f]{64}$/;

export class TallyError extends Error {
  override name = "TallyError";
}

export function newTally(spec: CorpusSpec): CorpusTally {
  if (!ID.test(spec.id) || spec.id.length > 64) throw new TallyError("a corpus id is a lower-case token");
  if (!HEX64.test(spec.seed) || !HEX64.test(spec.pool)) throw new TallyError("a corpus seed and pool are 64 hex digits");
  return { spec: { ...spec }, items: 0, entities: new Map(), pairs: new Map(), differences: [], count_mismatches: 0, keys: [], seen: new Set() };
}

export const pairKey = (a: string, b: string): string => (a < b ? `${a}|${b}` : `${b}|${a}`);

const entityRow = (tally: CorpusTally, id: string): EntityTally => {
  let row = tally.entities.get(id);
  if (row === undefined) {
    row = { present: 0, exercised: 0, equal: 0, differing: 0, first_difference: null };
    tally.entities.set(id, row);
  }
  return row;
};

const pairRow = (tally: CorpusTally, key: string): PairTally => {
  let row = tally.pairs.get(key);
  if (row === undefined) {
    row = { co_present: 0, co_exercised: 0, co_equal: 0 };
    tally.pairs.set(key, row);
  }
  return row;
};

/**
 * Add one compared item. Its position must be the next one of the corpus. A fresh corpus holds each item once: a repeated key is
 * refused rather than counted once, so a draw with replacement (or copies of a few queries) fails where it is made instead of
 * quietly yielding a corpus smaller than it says, and every count of a fresh tally is a count of distinct items.
 */
export function addItem(tally: CorpusTally, item: ItemResult): void {
  if (item.position !== tally.items) throw new TallyError("items are added in corpus order");
  if (tally.seen.has(item.key) && tally.spec.kind === "fresh") throw new TallyError("a fresh corpus holds each item once: a repeated key (draw without replacement)");
  const exercised = new Set(item.exercised);
  for (const id of exercised) if (!item.entities.includes(id)) throw new TallyError("an exercised entity must be present");
  tally.items += 1;
  tally.keys.push(item.key);
  tally.seen.add(item.key);
  const equal = item.difference === null;
  if (item.difference !== null) tally.differences.push(item.difference);
  for (const id of item.entities) {
    const row = entityRow(tally, id);
    row.present += 1;
    if (exercised.has(id)) {
      row.exercised += 1;
      if (equal) row.equal += 1;
    }
    if (!equal) {
      row.differing += 1;
      row.first_difference ??= item.position;
    }
  }
  // Pairs are admission evidence only on the fresh corpus (ADR-030 rule 7), so only a fresh tally keeps them.
  if (tally.spec.kind !== "fresh") return;
  const list = item.entities;
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      const a = list[i] as string;
      const b = list[j] as string;
      const row = pairRow(tally, pairKey(a, b));
      row.co_present += 1;
      if (exercised.has(a) && exercised.has(b)) {
        row.co_exercised += 1;
        if (equal) row.co_equal += 1;
      }
    }
  }
}

/** Add a batch comparison's items (and note a count mismatch). */
export function addBatch(tally: CorpusTally, batch: { items: readonly ItemResult[]; count_mismatch: boolean }): void {
  for (const item of batch.items) addItem(tally, item);
  if (batch.count_mismatch) tally.count_mismatches += 1;
}

const byKey = <T>(entries: Iterable<[string, T]>): [string, T][] => [...entries].sort(([a], [b]) => (a < b ? -1 : 1));

/** `content_hash`: what the corpus holds, SHA-256 over the JCS of its pool and the sorted keys of its items. */
export const contentHash = (tally: Pick<CorpusTally, "spec" | "keys">): string => sha256Jcs({ keys: [...tally.keys].sort(), pool: tally.spec.pool });

/** `tally_hash`: every count of the corpus (entities, pairs, items, count mismatches), so a ledger edited by hand (a pair dropped) no longer matches its tally. */
export function tallyHash(tally: CorpusTally): string {
  const entities = byKey(tally.entities).map(([id, r]) => [id, r.present, r.exercised, r.equal, r.differing, r.first_difference]);
  const pairs = byKey(tally.pairs).map(([key, r]) => [key, r.co_present, r.co_exercised, r.co_equal]);
  return sha256Jcs({ count_mismatches: tally.count_mismatches, entities, items: tally.items, pairs } as unknown as Json);
}
