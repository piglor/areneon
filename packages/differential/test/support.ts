// A toy level for the core's own tests: an input names its entities and which of them its answer reads; an output names the
// input it answers and carries a value.
import type { Backend, Level, Settled } from "../src/backend.ts";
import type { Json } from "../src/canonical.ts";
import { addItem, type CorpusSpec, type CorpusTally, newTally } from "../src/tally.ts";
import { freshSeed, MIN_FRESH_ITEMS } from "../src/admission.ts";

export interface ToyIn {
  n: number;
  entities: string[];
  reads: string[];
}
export interface ToyOut {
  for: number;
  value: Json;
}

export const toyLevel: Level<ToyIn, ToyOut> = {
  name: "mechanics-query",
  entities: (input) => [...input.entities].sort(),
  exercised: (input, oracle) => (oracle.ok ? [...input.reads].sort() : []),
  key: (input) => String(input.n),
  answers: (input, output) => output.for === input.n,
  bytes: (output) => output as unknown as Json,
  errorKey: (error) => `${(error as Error).name}:${(error as Error).message}`,
};

export const truth = (input: ToyIn): Settled<ToyOut> => (input.n < 0 ? { ok: false, error: new RangeError("refused") } : { ok: true, result: { for: input.n, value: { rolls: [input.n, input.n + 1], kind: "computed" } } });

export const backendOf = (id: string, run: (batch: readonly ToyIn[]) => Settled<ToyOut>[]): Backend<ToyIn, ToyOut> => ({ id, backend_format: 1, backend_commit: "c".repeat(40), run });
export const oracle = backendOf("oracle", (batch) => batch.map(truth));

export const BACKEND = { kind: "toy-kernel", backend_format: 3, backend_commit: "b".repeat(40) };
export const PIN = "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b";
export const POOL = "9".repeat(64);
export const FRESH: CorpusSpec = { id: "fresh-x", kind: "fresh", seed: freshSeed(BACKEND), pool: POOL };
export const REGRESSION: CorpusSpec = { id: "committed", kind: "regression", seed: "a".repeat(64), pool: "a".repeat(64) };
/** A corpus spec of another id and kind (the seed doubles as the pool). */
export const corpus = (id: string, kind: CorpusSpec["kind"], seed: string): CorpusSpec => ({ id, kind, seed, pool: seed });

/** Items of a filler entity the answers never read: what tops a fresh tally up to the level's minimum size. */
export const FILLER = "filler:item";

/**
 * A tally with `items` items, each holding `entities` and exercising `reads`; the items listed in `differing` have a difference.
 * A fresh tally is topped up with filler items to the level-1 minimum unless `fill` is false.
 */
type Group = { count: number; entities: string[]; reads?: string[]; differing?: number[] };

export function tallyOf(spec: CorpusSpec, groups: Group[], fill = true): CorpusTally {
  const tally = newTally(spec);
  const total = groups.reduce((n, g) => n + g.count, 0);
  const filler: Group[] = spec.kind === "fresh" && fill && total < MIN_FRESH_ITEMS["mechanics-query"] ? [{ count: MIN_FRESH_ITEMS["mechanics-query"] - total, entities: [FILLER], reads: [] }] : [];
  for (const group of [...groups, ...filler]) {
    for (let k = 0; k < group.count; k++) {
      const position = tally.items;
      const entities = [...group.entities].sort();
      const differs = group.differing?.includes(k) === true;
      addItem(tally, { position, key: String(position), entities, exercised: [...(group.reads ?? group.entities)].sort(), difference: differs ? { position, reason: "value", field: "$.x", entities } : null });
    }
  }
  return tally;
}
