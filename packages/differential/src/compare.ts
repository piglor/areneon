// Byte comparison of one batch: the backend's outcomes against the oracle's, item by item, with the first differing field.
import { type Json, jcs } from "./canonical.ts";
import type { Backend, Level, Settled } from "./backend.ts";

/** Why an item differs: `value` (bytes differ), `refusal` (both refused, differently), `settled` (one answered, one refused), `answer` (the backend's output answers another input), `missing` (the backend returned no outcome for it). */
export type DifferenceReason = "value" | "refusal" | "settled" | "answer" | "missing";

export interface Difference {
  /** The item's index in its corpus. */
  position: number;
  reason: DifferenceReason;
  /** JSON path of the first differing field (`$` for the whole outcome). */
  field: string;
  /** Every entity of the item (all of them are revoked by the difference). */
  entities: string[];
}

/** One compared item: its entities, which the oracle's answer exercised, and its difference if any. */
export interface ItemResult {
  position: number;
  entities: string[];
  exercised: string[];
  difference: Difference | null;
}

export class HarnessError extends Error {
  override name = "HarnessError";
}

const kindOf = (value: Json): string => (value === null ? "null" : Array.isArray(value) ? "array" : typeof value);

/** The JSON path of the first place where `a` and `b` differ (keys in sorted order), or null when they are equal. */
export function firstDifference(a: Json, b: Json, path = "$"): string | null {
  if (kindOf(a) !== kindOf(b)) return path;
  if (Array.isArray(a)) {
    const other = b as Json[];
    const n = Math.min(a.length, other.length);
    for (let i = 0; i < n; i++) {
      const found = firstDifference(a[i] as Json, other[i] as Json, `${path}[${i}]`);
      if (found !== null) return found;
    }
    return a.length === other.length ? null : `${path}[${n}]`;
  }
  if (a !== null && typeof a === "object") {
    const other = b as { [key: string]: Json };
    const keys = [...new Set([...Object.keys(a), ...Object.keys(other)])].sort();
    for (const key of keys) {
      if (!(key in a) || !(key in other)) return `${path}.${key}`;
      const found = firstDifference(a[key] as Json, other[key] as Json, `${path}.${key}`);
      if (found !== null) return found;
    }
    return null;
  }
  return a === b ? null : path;
}

/** Compare one item's two outcomes. The oracle's outcome must answer its input (else the harness itself is wrong). */
export function compareItem<In, Out>(level: Level<In, Out>, input: In, position: number, oracle: Settled<Out>, backend: Settled<Out> | undefined): ItemResult {
  if (oracle.ok && !level.answers(input, oracle.result)) throw new HarnessError(`the oracle's outcome of item ${position} does not answer it`);
  const entities = level.entities(input);
  const exercised = level.exercised(input, oracle);
  const differ = (reason: DifferenceReason, field: string): ItemResult => ({ position, entities, exercised, difference: { position, reason, field, entities } });
  if (backend === undefined) return differ("missing", "$");
  if (oracle.ok !== backend.ok) return differ("settled", "$");
  if (!oracle.ok || !backend.ok) {
    const same = level.errorKey((oracle as { error: unknown }).error) === level.errorKey((backend as { error: unknown }).error);
    return same ? { position, entities, exercised, difference: null } : differ("refusal", "$error");
  }
  if (!level.answers(input, backend.result)) return differ("answer", "$");
  const expected = level.bytes(oracle.result);
  const got = level.bytes(backend.result);
  if (jcs(expected) === jcs(got)) return { position, entities, exercised, difference: null };
  return differ("value", firstDifference(expected, got) as string);
}

/** The outcome of comparing a whole batch: one result per input, and whether the backend returned as many outcomes as inputs. */
export interface BatchComparison {
  items: ItemResult[];
  /** The backend returned another number of outcomes than inputs (a dropped or an extra result). */
  count_mismatch: boolean;
}

/**
 * Run `inputs` through both backends and compare. `first` is the corpus position of `inputs[0]`. The oracle must return one
 * outcome per input; the backend's are compared positionally, so a dropped result shows as answers to the wrong inputs from the
 * dropped item on (and the last as `missing`), a reordered batch as `answer` differences.
 */
export function compareBatch<In, Out>(level: Level<In, Out>, oracle: Backend<In, Out>, backend: Backend<In, Out>, inputs: readonly In[], first = 0): BatchComparison {
  const expected = oracle.run(inputs);
  if (expected.length !== inputs.length) throw new HarnessError("the oracle returned another number of outcomes than inputs");
  const got = backend.run(inputs);
  const items = inputs.map((input, i) => compareItem(level, input, first + i, expected[i] as Settled<Out>, got[i]));
  return { items, count_mismatch: got.length !== inputs.length };
}
