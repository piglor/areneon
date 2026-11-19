// The backend port and the level adapter (ADR-030 common rules 2 and 4, ADR-028 section 5). One core, two levels: level 1
// compares mechanics queries and their results (this package's `mechanics-level.ts`), level 2 a battle step and its player
// view (a later adapter on the same core). Showdown is always the oracle backend.
import type { Json } from "./canonical.ts";

/** One input's outcome: its result, or what the backend threw for it (a refusal is part of the answer). */
export type Settled<Out> = { ok: true; result: Out } | { ok: false; error: unknown };

/** A backend: a batch in, the outcomes in input order (as `BatchRunner` and `FeaturePool.run` return them). */
export interface Backend<In, Out> {
  /** A name for reports. */
  readonly id: string;
  /** Bumped when the backend computes anything differently. */
  readonly backend_format: number;
  /** The frozen commit of its code (40 hex). */
  readonly backend_commit: string;
  run(batch: readonly In[]): Settled<Out>[];
}

/**
 * What one level compares and how it names entities. `entities` and `exercised` return sorted, unique `<kind>:<id>` strings,
 * `exercised` a subset of `entities`; both are reviewed rules of the level (ADR-026 slice F's exercising rule: the entity is
 * present and the answer reads it). `exercised` is read from the ORACLE's outcome: the oracle defines what was read.
 */
export interface Level<In, Out> {
  readonly name: "mechanics-query" | "battle-step";
  entities(input: In): string[];
  exercised(input: In, oracle: Settled<Out>): string[];
  /** A stable key of an input (level 1: its query hash): what a corpus's content hash and the shadow sample are keyed by. */
  key(input: In): string;
  /** Whether `output` is an answer to `input` at all (it names it): a misattributed or reordered answer fails this. */
  answers(input: In, output: Out): boolean;
  /** The comparable form of an output (compared by its JCS bytes). */
  bytes(output: Out): Json;
  /** The comparable form of a thrown refusal (its class and fixed words). */
  errorKey(error: unknown): string;
}
