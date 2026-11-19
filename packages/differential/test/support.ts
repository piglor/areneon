// A toy level for the core's own tests: an input names its entities and which of them its answer reads; an output names the
// input it answers and carries a value.
import type { Backend, Level, Settled } from "../src/backend.ts";
import type { Json } from "../src/canonical.ts";
import { addItem, type CorpusSpec, type CorpusTally, newTally } from "../src/tally.ts";
import { freshSeed } from "../src/admission.ts";

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
  answers: (input, output) => output.for === input.n,
  bytes: (output) => output as unknown as Json,
  errorKey: (error) => `${(error as Error).name}:${(error as Error).message}`,
};

export const truth = (input: ToyIn): Settled<ToyOut> => (input.n < 0 ? { ok: false, error: new RangeError("refused") } : { ok: true, result: { for: input.n, value: { rolls: [input.n, input.n + 1], kind: "computed" } } });

export const backendOf = (id: string, run: (batch: readonly ToyIn[]) => Settled<ToyOut>[]): Backend<ToyIn, ToyOut> => ({ id, backend_format: 1, backend_commit: "c".repeat(40), run });
export const oracle = backendOf("oracle", (batch) => batch.map(truth));

export const BACKEND = { kind: "toy-kernel", backend_format: 3, backend_commit: "b".repeat(40) };
export const PIN = "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b";
export const FRESH: CorpusSpec = { id: "fresh-x", kind: "fresh", seed: freshSeed(BACKEND) };
export const REGRESSION: CorpusSpec = { id: "committed", kind: "regression", seed: "a".repeat(64) };

/** A tally with `items` items, each holding `entities` and exercising `reads`; the items listed in `differing` have a difference. */
export function tallyOf(spec: CorpusSpec, groups: { count: number; entities: string[]; reads?: string[]; differing?: number[] }[]): CorpusTally {
  const tally = newTally(spec);
  for (const group of groups) {
    for (let k = 0; k < group.count; k++) {
      const position = tally.items;
      const entities = [...group.entities].sort();
      const differs = group.differing?.includes(k) === true;
      addItem(tally, { position, entities, exercised: [...(group.reads ?? group.entities)].sort(), difference: differs ? { position, reason: "value", field: "$.x", entities } : null });
    }
  }
  return tally;
}
