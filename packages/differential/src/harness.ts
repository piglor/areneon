// One corpus through the harness: inputs in batches to the oracle and the backend, every item compared and tallied.
import type { Backend, Level } from "./backend.ts";
import { compareBatch } from "./compare.ts";
import { addBatch, type CorpusSpec, type CorpusTally, newTally } from "./tally.ts";

export class RunError extends Error {
  override name = "RunError";
}

/** Compare `inputs` (the corpus, in order) in batches of `batchSize`; returns the corpus's tally. */
export function runCorpus<In, Out>(level: Level<In, Out>, oracle: Backend<In, Out>, backend: Backend<In, Out>, spec: CorpusSpec, inputs: readonly In[], batchSize = 256): CorpusTally {
  if (!Number.isSafeInteger(batchSize) || batchSize < 1) throw new RunError("a batch holds at least one input");
  const tally = newTally(spec);
  for (let first = 0; first < inputs.length; first += batchSize) addBatch(tally, compareBatch(level, oracle, backend, inputs.slice(first, first + batchSize), first));
  return tally;
}
